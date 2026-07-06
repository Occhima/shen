"""The two primitives: Lookup and @pricer.

The registry is an INDEX, not a router. Decorating a calculator makes
the function itself a self-contained pricer — fn.price(trades, mkt)
prices with that one function; plib.engine.price(...) is merely a
book-level combinator over an explicit iterable of pricers. REGISTRY
exists so you can *recover* pricers by name for dynamic use cases
(REGISTRY["bullet_swap"]), never so a central dispatcher can find them
for you. The user knows where things are.

Lookup — a declarative join against a market contract. The contract
carries its own geometry (_key, _at); the Lookup only maps instrument
columns onto it. interp="prev" is the fixing convention; "lerp"
interpolates pillars in log space. Scenario dimensions ride along.

@pricer — attaches (contract, lookups, pure calculator) to the
function and indexes it. price_legs(schema, leg, explode=...) composes
a leg pricer into a composite (explode in, reduce out) and returns a
first-class Pricer object. All drift checks at import time.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce as fold
from typing import Literal

import pandera.polars as pa
import polars as pl

from plib.contracts.market import MarketObject

type Calculator = Callable[..., pl.Expr]
type Frames = Mapping[type[MarketObject], pl.LazyFrame]

MEASURE = "value"
DIMS = ("scenario_id",)   # ride-along dimensions
INJECTED = ("ref_date",)  # columns the engine injects from Market


@dataclass(frozen=True, slots=True)
class Lookup:
    src: type[MarketObject]
    by: str                                  # instrument col -> src._key
    at: str                                  # instrument col -> src._at
    take: str | tuple[str, ...]              # value column(s) on src
    As: str | tuple[str, ...] | None = None  # aliases; defaults to `take`
    interp: Literal["prev", "lerp"] = "prev"

    @property
    def takes(self) -> tuple[str, ...]:
        return (self.take,) if isinstance(self.take, str) else tuple(self.take)

    @property
    def aliases(self) -> tuple[str, ...]:
        if self.As is None:
            return self.takes
        return (self.As,) if isinstance(self.As, str) else tuple(self.As)

    def __call__(self, lf: pl.LazyFrame, frames: Frames) -> pl.LazyFrame:
        takes, aliases = self.takes, self.aliases
        if len(takes) != len(aliases):
            raise TypeError(f"Lookup({self.src.__name__}): take/As length mismatch")
        rhs = frames[self.src]
        dims = [d for d in DIMS
                if d in rhs.collect_schema() and d in lf.collect_schema()]
        # private rhs names: the instrument's `at`/`by` columns may share
        # names with the market object's (e.g. option.expiry vs
        # VolSurface.expiry) — aliasing makes collisions impossible
        rhs = (rhs.select(pl.col(self.src._key).alias("_rkey"),
                          pl.col(self.src._at).alias("_rat"),
                          *(pl.col(t).alias(f"_rv{i}")
                            for i, t in enumerate(takes)), *dims)
                  .sort("_rat"))
        kw = dict(left_on=self.at, right_on="_rat",
                  by_left=[self.by, *dims], by_right=["_rkey", *dims])
        lo = (lf.sort(self.at)
                .join_asof(rhs, strategy="backward", **kw)
                .rename({"_rat": "_t0",
                         **{f"_rv{i}": f"_lo{i}" for i in range(len(takes))}}))
        if self.interp == "prev":
            return (lo.rename({f"_lo{i}": a for i, a in enumerate(aliases)})
                      .drop("_t0"))
        hi = (lo.join_asof(rhs, strategy="forward", **kw)
                .rename({"_rat": "_t1",
                         **{f"_rv{i}": f"_hi{i}" for i in range(len(takes))}}))
        w = ((pl.col(self.at) - pl.col("_t0")).dt.total_days()
             / (pl.col("_t1") - pl.col("_t0")).dt.total_days().clip(lower_bound=1)
             ).fill_null(0.0)
        lerps = (
            ((1 - w) * pl.col(f"_lo{i}").fill_null(pl.col(f"_hi{i}"))
             + w * pl.col(f"_hi{i}").fill_null(pl.col(f"_lo{i}"))).alias(a)
            for i, a in enumerate(aliases))
        temps = ["_t0", "_t1",
                 *(f"_lo{i}" for i in range(len(takes))),
                 *(f"_hi{i}" for i in range(len(takes)))]
        return hi.with_columns(*lerps).drop(temps)


@dataclass(frozen=True, slots=True)
class Pricer:
    itype: str
    schema: type[pa.DataFrameModel]
    lookups: tuple[Lookup, ...]
    calc: Calculator
    params: tuple[str, ...]
    legs: type[pa.DataFrameModel] | None = None
    explode: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None
    reduce: Literal["sum"] | None = None
    derive: Mapping[str, pl.Expr] | None = None
    """Declarative frame-algebra columns (e.g. year fractions) computed
    by the engine after ref_date injection, before lookups. Dates ->
    tau is frame work, not scalar math, so it gets its own slot."""

    def base_expr(self) -> pl.Expr:
        return self.calc(*(pl.col(p) for p in self.params))

    # -- self-contained usage: no central dispatch required -----------
    def price(self, instruments, mkt=None, *, trace: bool = False,
              strict: bool = True) -> pl.LazyFrame:
        from plib.engine import price as _price
        return _price(instruments, mkt, pricers=(self,),
                      trace=trace, strict=strict)

    def unresolved(self, instruments, mkt=None) -> pl.LazyFrame:
        from plib.engine import unresolved as _u
        return _u(instruments, mkt, pricers=(self,))


REGISTRY: dict[str, Pricer] = {}


def as_pricer(obj) -> Pricer:
    """Accept a Pricer or a decorated calculator interchangeably."""
    p = getattr(obj, "pricer", obj)
    if not isinstance(p, Pricer):
        raise TypeError(f"{obj!r} is not a pricer")
    return p


def resolve(lf: pl.LazyFrame, lookups: tuple[Lookup, ...], frames: Frames) -> pl.LazyFrame:
    return fold(lambda acc, lk: lk(acc, frames), lookups, lf)


def _checked(schema: type[pa.DataFrameModel],
             lookups: tuple[Lookup, ...]) -> tuple[set[str], set[str]]:
    cols = set(schema.to_schema().columns) | set(INJECTED)
    alias_list = [a for lk in lookups for a in lk.aliases]
    aliases = set(alias_list)
    if missing := {c for lk in lookups for c in (lk.by, lk.at)} - cols:
        raise TypeError(f"{schema.__name__}: lookups reference missing columns {missing}")
    if shadow := aliases & cols:
        raise TypeError(f"{schema.__name__}: lookup aliases shadow columns {shadow}")
    if len(alias_list) != len(aliases):
        raise TypeError(f"{schema.__name__}: duplicate lookup aliases")
    return cols, aliases


def pricer(schema: type[pa.DataFrameModel], lookups: tuple[Lookup, ...] = (),
           *, derive: Mapping[str, pl.Expr] | None = None):
    """Valuation: contract + lookups + pure calculator. The decorated
    function stays pure math AND gains .price / .unresolved / .pricer.
    `derive` declares engine-computed columns (frame algebra like
    tau = busdays(ref_date, expiry)/252) usable as parameters."""
    cols, aliases = _checked(schema, lookups)
    cols |= set(derive or ())

    def deco(fn: Calculator) -> Calculator:
        params = tuple(inspect.signature(fn).parameters)
        if unknown := set(params) - cols - aliases:
            raise TypeError(
                f"{fn.__name__}: parameters {unknown} are neither "
                f"{schema.__name__} columns nor lookup aliases {aliases}")
        p = Pricer(_snake(schema.__name__), schema, tuple(lookups), fn, params,
                   derive=derive)
        REGISTRY[p.itype] = p
        fn.pricer = p
        fn.price = p.price
        fn.unresolved = p.unresolved
        return fn

    return deco


def _auto_explode(structure: type[pa.DataFrameModel],
                  leg: type[pa.DataFrameModel],
                  sides: Mapping[str, float]) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Derive the explode from the two contracts + a sides mapping.

    For each leg column c: take `{side}_{c}` from the structure if it
    exists, else the shared column c; `sign` is the side's literal.
    Resolvability is checked here, at registration."""
    s_cols = set(structure.to_schema().columns)
    l_cols = [c for c in leg.to_schema().columns]
    plan: dict[str, list[pl.Expr]] = {}
    for side, sign in sides.items():
        exprs = []
        for c in l_cols:
            if f"{side}_{c}" in s_cols:
                exprs.append(pl.col(f"{side}_{c}").alias(c))
            elif c in s_cols:
                exprs.append(pl.col(c))
            elif c == "sign":
                exprs.append(pl.lit(1.0).alias("sign"))   # neutral; weight is _w
            else:
                raise TypeError(
                    f"{structure.__name__}: cannot source leg column {c!r} "
                    f"for side {side!r} (need {side}_{c} or {c})")
        exprs.append(pl.lit(float(sign)).alias("_w"))     # combinator weight
        plan[side] = exprs

    def explode(lf: pl.LazyFrame) -> pl.LazyFrame:
        return pl.concat([lf.select(exprs) for exprs in plan.values()])
    return explode


def price_legs(schema: type[pa.DataFrameModel], leg, *,
               explode: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
               reduce: Literal["sum"] = "sum"):
    """Structure as a FUNCTION OF SIDES. The decorated combinator's
    parameter names name the leg prefixes; its body IS the fold:

        @price_legs(BulletSwap, bullet)
        def bullet_swap(active, passive):
            return active - passive

    From that single pure function everything is derived at import:
    the side coefficients (+1, -1) by probing with unit inputs — the
    same trick sensitivities uses on calculators — the explode by the
    prefix convention ({side}_{c} from the structure contract, else the
    shared column c), and linearity is VERIFIED (a structure that is
    not a linear combination of its sides is not a price_legs structure
    and raises at registration). The combinator stays a plain testable
    function of floats: bullet_swap(active=2.0, passive=0.5) == 1.5.

    Genuinely irregular reshapes bypass the combinator with
    explode= (LazyFrame -> leg rows) and get the raw hook.
    """
    lp = as_pricer(leg)

    def _register(explode_fn, attach_to=None):
        p = Pricer(_snake(schema.__name__), schema, lp.lookups, lp.calc,
                   lp.params, legs=lp.schema, explode=explode_fn,
                   reduce=reduce, derive=lp.derive)
        REGISTRY[p.itype] = p
        if attach_to is not None:
            attach_to.pricer = p
            attach_to.price = p.price
            attach_to.unresolved = p.unresolved
            return attach_to
        return p

    if explode is not None:
        return _register(explode)

    def deco(fn: Callable[..., float]):
        sides = tuple(inspect.signature(fn).parameters)
        if not sides:
            raise TypeError(f"{fn.__name__}: a structure needs at least one side")
        coefs = {s: float(fn(*(1.0 if t == s else 0.0 for t in sides)))
                 for s in sides}
        if abs(fn(*(1.0 for _ in sides)) - sum(coefs.values())) > 1e-9:
            raise TypeError(
                f"{fn.__name__}: structure must be linear in its sides")
        return _register(_auto_explode(schema, lp.schema, coefs), attach_to=fn)

    return deco


def _snake(name: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")
