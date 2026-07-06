"""The engine.

price(): partition by instrument_type -> validate -> explode (if legs)
-> inject ref_date -> cross scenario dimension -> fold lookups -> pure
calculator -> reduce -> concat. One collect(), owned by the caller.

Market IS the context, explicitly. It is a frozen value: ref_date +
canonical frames. Transformations (with_shocks, bump) return NEW
Markets — no context managers, no globals. gs-quant's
`with PricingContext(...)` mutates hidden state; here `price(trades,
mkt.with_shocks(...))` says the same thing referentially transparently,
and two Markets can coexist in one expression (that is exactly how
dv01 works).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace

import polars as pl

from shen.contracts.market import MARKET_TYPES, MarketObject, as_lazy
from shen.core.registry import MEASURE, REGISTRY, Pricer, as_pricer, resolve

ID_COLS = ("instrument_id", "instrument_type")


@dataclass(frozen=True, slots=True)
class Market:
    frames: Mapping[type[MarketObject], pl.LazyFrame] = field(default_factory=dict)
    ref_date: dt.date | None = None

    @classmethod
    def load(cls, *data, ref_date: dt.date | None = None, **named) -> "Market":
        """Market.load({Curve: raw, ...}, ref_date=...)   # by contract
           Market.load(curve=raw, fx=raw, ref_date=...)   # by registered name

        Contracts self-register their snake name at definition
        (MarketObject.__init_subclass__), so a user-defined
        `class Vol(MarketObject)` immediately makes `vol=` valid here —
        init-able like a dataclass, open like a registry. Mapping /
        pairs / kwargs all normalize to the same shape; each contract's
        validate() is the gate.

        Conventions (contracts with _canonical set) are routed through
        _to_canonical and merged into their canonical contract's frame
        before validation — so `di_curve=df` (rate+anchor) and
        `curve=df` (log_df) land in the same Curve frame.
        """
        if len(data) == 1 and not isinstance(data[0], tuple):
            entries = dict(data[0])        # Mapping, or iterable of pairs
        else:
            entries = dict(data)           # legacy varargs pairs
        for name, raw in named.items():
            contract = MARKET_TYPES.get(name)
            if contract is None:
                raise TypeError(f"unknown market kwarg {name!r}; "
                                f"registered: {sorted(MARKET_TYPES)}")
            if contract in entries:
                raise TypeError(f"{contract.__name__} given twice")
            entries[contract] = raw
        groups: dict[type[MarketObject], list[pl.LazyFrame]] = {}
        for contract, raw in entries.items():
            target = contract._canonical
            lf = contract._to_canonical(as_lazy(raw)) if target is not None else as_lazy(raw)
            groups.setdefault(target or contract, []).append(lf)
        return cls(
            {c: c.validate(cls._merge_rows(lfs)) for c, lfs in groups.items()},
            ref_date)

    @staticmethod
    def _merge_rows(lfs: list[pl.LazyFrame]) -> pl.LazyFrame:
        """Concat rows for the same canonical contract. diagonal_relaxed
        unions columns (filling nulls) so convention sources (log_df)
        and direct loads (discount_factor) merge before _parse derives
        the missing per-row values inside validate."""
        return lfs[0] if len(lfs) == 1 else pl.concat(lfs, how="diagonal_relaxed")

    def __getitem__(self, contract: type[MarketObject]) -> pl.LazyFrame:
        return self.frames[contract]

    def with_data(self, *data, **named) -> "Market":
        """Functional update: a new Market with added frames, validated
        through the same gates. Rows MERGE into existing contracts —
        new data overrides matching (key, at) rows and extends the rest
        — so adding an implied curve never clobbers vendor curves. This
        is how one instrument's OUTPUT becomes another instrument's
        market object: DI1 quotes -> implied curve -> swaps."""
        add = Market.load(*data, ref_date=self.ref_date, **named)
        frames = dict(self.frames)
        for c, lf in add.frames.items():
            if c in frames:
                lf = (pl.concat([frames[c], lf])
                        .unique(subset=[c._key, c._at], keep="last")
                        .sort(c._key, c._at))
            frames[c] = lf
        return Market(frames, self.ref_date)

    def with_shocks(self, contract: type[MarketObject],
                    scenarios: pl.LazyFrame, apply: pl.Expr) -> "Market":
        """Cross a scenario dimension into one market object.

        `scenarios` must carry scenario_id + contract._key (+ shock
        params); `apply` produces the shocked value column. Scenarios
        stay a dimension; price() is unchanged.
        """
        shocked = (self[contract]
                   .join(scenarios, on=contract._key, how="inner")
                   .with_columns(apply)
                   .sort(contract._key, contract._at))
        return Market({**self.frames, contract: shocked}, self.ref_date)


_AMBIENT: ContextVar["Market | None"] = ContextVar("shen_market", default=None)


def market() -> "Market":
    """The ambient Market. Explicit beats ambient: every function here
    takes mkt= and prefers it; this resolves the fallback binding."""
    m = _AMBIENT.get()
    if m is None:
        raise LookupError(
            "no ambient market — enter `with MarketContext(...)` or pass mkt=")
    return m


def _resolve(mkt: "Market | None") -> "Market":
    return mkt if mkt is not None else market()


class MarketContext:
    """Bind a Market as the ambient context (dynamic scoping).

    contextvars, not a global: entering binds an immutable value,
    exiting token-resets — nesting and async are safe, nothing mutates.

        MarketContext(ref, data={Curve: df, ...})  build a Market, bind it
        MarketContext(ref, curve=df, fx=df)        same, by registered name
        MarketContext(ref, data=mkt)       rebind an existing Market at ref
        MarketContext(ref)                 inherit ambient frames, new ref
        MarketContext.bump(C, scen, apply) inherit ambient, shocked frames
    """

    def __init__(self, ref_date: dt.date | None = None, *, data=None, **named):
        if named and isinstance(data, Market):
            raise TypeError("pass either data=Market or named frames, not both")
        match data:
            case _ if named:
                m = Market.load(data or {}, ref_date=ref_date, **named)
            case Market():
                m = data if ref_date is None else replace(data, ref_date=ref_date)
            case None:
                base = market()
                m = base if ref_date is None else replace(base, ref_date=ref_date)
            case _:
                m = Market.load(data, ref_date=ref_date)
        self.market = m
        self._token = None

    @classmethod
    def bump(cls, contract: type[MarketObject], scenarios: pl.LazyFrame,
             apply: pl.Expr) -> "MarketContext":
        return cls(data=market().with_shocks(contract, scenarios, apply))

    def __enter__(self) -> Market:
        self._token = _AMBIENT.set(self.market)
        return self.market

    def __exit__(self, *exc) -> None:
        _AMBIENT.reset(self._token)


def _scenarios(mkt: Market) -> pl.LazyFrame | None:
    ids = [f.select("scenario_id") for f in mkt.frames.values()
           if "scenario_id" in f.collect_schema()]
    return pl.concat(ids).unique().sort("scenario_id") if ids else None


def _pricers(pricers) -> tuple[Pricer, ...]:
    """None -> everything indexed (convenience). Explicit iterable ->
    exactly those, registry not consulted: no central dispatch."""
    return tuple(as_pricer(p) for p in (REGISTRY.values() if pricers is None else pricers))


def _resolved(instruments, mkt: Market,
              pricers=None) -> Iterator[tuple[Pricer, pl.LazyFrame, list[str]]]:
    """Shared spine of price(), unresolved() and risk.*: yields
    (pricer, fully-bound rows, dimension columns) per supplied pricer.
    Fully lazy — no metadata collect. Rows whose instrument_type matches
    no supplied pricer are not priced; audit with unpriced()."""
    lf = as_lazy(instruments)
    scen = _scenarios(mkt)
    have = set(lf.collect_schema().names())
    for p in _pricers(pricers):
        if not set(p.schema.to_schema().columns) <= have:
            continue   # this type cannot exist in this frame; unpriced() audits
        rows = p.schema.validate(lf.filter(pl.col("instrument_type") == p.itype))
        if p.explode is not None:
            exploded = p.explode(rows)
            if "_w" in exploded.collect_schema():
                # keep the combinator weight through the strict leg gate
                body = p.legs.validate(exploded.drop("_w"))
                rows = pl.concat([body, exploded.select("_w")], how="horizontal")
            else:
                rows = p.legs.validate(exploded)
        if mkt.ref_date is not None:
            rows = rows.with_columns(ref_date=pl.lit(mkt.ref_date))
        if p.derive:
            rows = rows.with_columns(**p.derive)
        dims: list[str] = []
        if scen is not None:
            rows = rows.join(scen, how="cross")
            dims = ["scenario_id"]
        yield p, resolve(rows, p.lookups, mkt.frames), dims


def price(instruments, mkt: Market | None = None, *, pricers=None,
          trace: bool = False, strict: bool = True) -> pl.LazyFrame:
    """Value vector: one row per (instrument, scenario). Lazy.

    trace=True is the debug view: skip the final projection AND the
    reduce, returning every column at the finest grain — contract
    fields, injected ref_date, every lookup alias (spot, log_df_*,
    fx...), sign, and the leg-level value before folding. Nothing extra
    is computed to produce this; it is the same plan minus projection.
    (Conversely, trace=False is not cosmetic: projection pushdown lets
    Polars prune unused columns out of the joins entirely.)
    Different instrument types have different columns, so trace concats
    diagonally — absent fields are null.
    """
    mkt = _resolve(mkt)

    def one(p: Pricer, bound: pl.LazyFrame, dims: list[str]) -> pl.LazyFrame:
        out = bound.with_columns(p.base_expr().alias(MEASURE))
        if trace:
            return out
        if p.reduce == "sum":
            v = (pl.col(MEASURE) * pl.col("_w")
                 if "_w" in out.collect_schema() else pl.col(MEASURE))
            out = out.group_by(*ID_COLS, *dims).agg(v.sum().alias(MEASURE))
        return out.select(*ID_COLS, *dims, MEASURE)

    parts = (one(*r) for r in _resolved(instruments, mkt, pricers))
    out = pl.concat(parts, how="diagonal") if trace else pl.concat(parts)
    if strict and not trace:
        # a missing fixing/pillar must never price as null silently
        out = out.with_columns(
            pl.when(pl.col(MEASURE).is_null())
            .then(pl.lit(float("nan")))
            .otherwise(pl.col(MEASURE)).alias(MEASURE))
    return out


def unresolved(instruments, mkt: Market | None = None, *, pricers=None) -> pl.LazyFrame:
    """Data quality as a query: the rows (at the finest grain) whose
    lookups failed to bind — any null alias. Empty frame == the whole
    book resolves. Run this BEFORE trusting price()."""
    mkt = _resolve(mkt)

    def one(p: Pricer, bound: pl.LazyFrame, dims: list[str]) -> pl.LazyFrame:
        aliases = [a for lk in p.lookups for a in lk.aliases]
        return (bound
                .filter(pl.any_horizontal(pl.col(a).is_null() for a in aliases))
                .select(*ID_COLS, *dims, *aliases))
    return pl.concat((one(*r) for r in _resolved(instruments, mkt, pricers)),
                     how="diagonal")


def unpriced(instruments, *, pricers=None) -> pl.LazyFrame:
    """Rows no supplied pricer covers — the audit for explicit pricer
    lists (lazy: type membership is known python-side, no collect)."""
    types = [p.itype for p in _pricers(pricers)]
    return as_lazy(instruments).filter(~pl.col("instrument_type").is_in(types))
