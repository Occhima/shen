"""End-to-end: pandas in, lazy value vector out; positions; swap; risk."""

import datetime as dt
import math

import pandas as pd
import polars as pl
import pytest

import plib.pricers  # noqa: F401 — registers commodity + swap
from plib import Market, Curve, Fx, Position, Spot, mtm, pnl, price, sensitivities
from plib.pricers.commodity import commodity_forward
from plib.pricers.swap import bullet

REF = dt.date(2026, 7, 3)


@pytest.fixture
def mkt():
    return Market.load({
        Curve: pd.DataFrame({
            "curve_id": ["USD", "USD", "CUPOM", "CUPOM"],
            "pillar_date": [dt.date(2026, 7, 1), dt.date(2027, 7, 1)] * 2,
            "discount_factor": [1.0, 0.96, 1.0, 0.98],
        }),
        Spot: pd.DataFrame({
            "index": ["BRENT", "IPCA", "CDI"],
            "fixing_date": [dt.date(2026, 6, 30), dt.date(2026, 7, 2), dt.date(2026, 7, 2)],
            "value": [102.0, 105.0, 102.0],
            "publication_lag_days": [1, 0, 0],
        }),
        Fx: pd.DataFrame({
            "pair": ["USDBRL", "BRLBRL"],
            "date": [dt.date(2026, 7, 1), dt.date(2026, 7, 1)],
            "rate": [5.40, 1.0],
        })},
        ref_date=REF,
    )


def _fwd_trades():
    return pd.DataFrame({
        "instrument_id": ["t1"],
        "instrument_type": ["commodity_forward"],
        "strike": [95.0],
        "disc_curve": ["USD"],
        "index": ["BRENT"],
        "ccy_pair": ["USDBRL"],
        "fixing_date": [dt.date(2026, 7, 1)],
        "pay_date": [dt.date(2026, 12, 15)],
    })


def _expected_df():
    w = (dt.date(2026, 12, 15) - dt.date(2026, 7, 1)).days / 365
    return math.exp(w * math.log(0.96))


def test_forward_value_and_mtm(mkt):
    out = price(_fwd_trades(), mkt)
    assert isinstance(out, pl.LazyFrame)
    value = out.collect()["value"][0]
    assert abs(value - (102.0 - 95.0) * _expected_df() * 5.40) < 1e-6

    positions = Position.validate(
        pd.DataFrame({"instrument_id": ["t1"], "qty": [100.0]}))
    v = mtm(positions, out).collect()
    assert abs(v["mtm"][0] - 100.0 * value) < 1e-9


def test_swap_explodes_prices_reduces(mkt):
    swaps = pd.DataFrame({
        "instrument_id": ["s1"],
        "instrument_type": ["bullet_swap"],
        "end_date": [dt.date(2027, 7, 1)],       # exactly on pillar: df = 0.96
        "active_curve": ["USD"], "active_index": ["IPCA"],
        "active_index_base": [100.0],
        "active_ccy_pair": ["BRLBRL"], "active_fx_base": [1.0],
        "passive_curve": ["USD"], "passive_index": ["CDI"],
        "passive_index_base": [100.0],
        "passive_ccy_pair": ["BRLBRL"], "passive_fx_base": [1.0],
    })
    out = price(swaps, mkt)
    v = out.collect()
    assert len(v) == 1                            # reduced back to one row
    unit = (105.0 / 100.0 - 102.0 / 100.0) * 0.96 # UNIT value: factor hug only
    assert abs(v["value"][0] - unit) < 1e-9

    # notional enters as quantity, in positions — not in the pricer
    positions = Position.validate(
        pd.DataFrame({"instrument_id": ["s1"], "qty": [1_000_000.0]}))
    book = mtm(positions, out).collect()
    assert abs(book["mtm"][0] - 1_000_000.0 * unit) < 1e-6


def test_sensitivities_per_factor(mkt):
    s = sensitivities(_fwd_trades(), mkt).collect()
    df, fx = _expected_df(), 5.40
    dv = {r["factor"]: r["dv"] for r in s.iter_rows(named=True)}
    assert set(dv) == {"spot", "log_df_pay", "fx"}
    assert abs(dv["spot"] - df * fx) < 1e-6                    # linear: exact
    assert abs(dv["fx"] - (102.0 - 95.0) * df) < 1e-6          # linear: exact
    assert abs(dv["log_df_pay"] - (102.0 - 95.0) * df * fx) < 1e-2  # O(h)


def test_calculators_are_pure_math():
    assert commodity_forward(strike=95.0, spot=100.0, log_df_pay=0.0, fx=5.0) \
        == 5.0 * 5.0
    assert bullet(sign=-1.0, index_fwd=102.0, index_base=100.0,
                  fx_fwd=5.5, fx_base=5.0, log_df_end=0.0) == -1.02 * 1.1


def test_wdo_covered_parity(mkt):
    wdo = pd.DataFrame({
        "instrument_id": ["WDO-N27"],
        "instrument_type": ["wdo_future"],
        "strike": [5.35],
        "maturity": [dt.date(2027, 7, 1)],      # on pillar: exact dfs
        "di_curve": ["USD"],
        "cupom_curve": ["CUPOM"],
        "ccy_pair": ["USDBRL"],
    })
    v = price(wdo, mkt).collect()
    fwd = 5.40 * 0.98 / 0.96
    assert abs(v["value"][0] - (fwd - 5.35)) < 1e-9


def test_pnl_typed(mkt):
    out = price(_fwd_trades(), mkt)
    pos = Position.validate(pd.DataFrame({"instrument_id": ["t1"], "qty": [10.0]}))
    today = mtm(pos, out)
    yesterday = today.with_columns(mtm=pl.col("mtm") - 1.5)
    assert abs(pnl(today, yesterday).collect()["pnl"][0] - 1.5) < 1e-9


def test_review_fixes(mkt):
    from plib import unresolved
    from plib.registry import Lookup, pricer
    from plib.contracts.instruments import Instrument

    # 1. missing fixing: reported by unresolved(), NaN (not null) in price
    bad = _fwd_trades().assign(index="WTI")
    u = unresolved(bad, mkt).collect()
    assert len(u) == 1 and u["spot"][0] is None
    v = price(bad, mkt).collect()["value"][0]
    assert v != v                                    # NaN, loud not silent

    # 2. duplicate lots net at the gate: one row, qty summed
    pos = Position.validate(pd.DataFrame(
        {"instrument_id": ["t1", "t1"], "qty": [100.0, 50.0]}))
    book = mtm(pos, price(_fwd_trades(), mkt)).collect()
    assert len(book) == 1 and book["qty"][0] == 150.0

    # 3. alias shadowing a contract column fails at import time
    import polars as _pl
    import pytest as _pytest
    from plib.contracts.market import Curve as _Curve
    with _pytest.raises(TypeError, match="shadow"):
        class X(Instrument):
            strike: float
            curve: str
            d: _pl.Date
        @pricer(X, lookups=(Lookup(_Curve, by="curve", at="d",
                                   take="log_df", As="strike"),))
        def x(strike):
            return strike


def test_di_futures_book(mkt):
    trades = pd.DataFrame({
        "instrument_id": ["DI1-A", "DI1-B"],
        "instrument_type": ["di_future"] * 2,
        "strike_pu": [95_500.0, 95_500.0],
        "maturity": [dt.date(2027, 7, 1)] * 2,   # on pillar: df = 0.96 exact
        "di_curve": ["USD", "USD"],
    })
    v = price(trades, mkt).collect()
    assert all(abs(x - (100_000.0 * 0.96 - 95_500.0)) < 1e-9 for x in v["value"])

    pos = Position.validate(pd.DataFrame(
        {"instrument_id": ["DI1-A", "DI1-B"], "qty": [10.0, -10.0]}))
    book = mtm(pos, v.lazy()).collect()
    assert abs(book["mtm"].sum()) < 1e-9         # flat book nets to zero


def test_decentralized_pricing(mkt):
    """No central dispatch: the function IS the pricer."""
    from plib import price as book_price, unpriced
    from plib.pricers.commodity import commodity_forward
    from plib.pricers.swap import bullet_swap

    trades = _fwd_trades()
    a = commodity_forward.price(trades, mkt).collect()          # fn-level
    b = book_price(trades, mkt, pricers=(commodity_forward,)).collect()
    assert a["value"][0] == b["value"][0]

    # explicit pricer list that does NOT cover the book -> audit finds it
    left = unpriced(trades, pricers=(bullet_swap,)).collect()
    assert left["instrument_id"].to_list() == ["t1"]

    # registry as an index (recovery, not routing)
    from plib.registry import REGISTRY
    assert REGISTRY["commodity_forward"] is commodity_forward.pricer


def test_curve_from_rate_convention(mkt):
    """Convention is a contract characteristic: rate+anchor in, DFs out."""
    import numpy as np

    anchor, pillar, rate = dt.date(2026, 7, 1), dt.date(2027, 7, 1), 0.11
    c = Curve.validate(pd.DataFrame({
        "curve_id": ["DI"], "pillar_date": [pillar],
        "rate": [rate], "anchor_date": [anchor],
    })).collect()
    du = int(np.busday_count(anchor, pillar))
    expected = -(du / 252) * math.log1p(rate)
    assert abs(c["log_df"][0] - expected) < 1e-12
    assert abs(c["discount_factor"][0] - math.exp(expected)) < 1e-12


def test_fungibilidade_preserves_mtm(mkt):
    """sum(q_i*(V-K_i)) == (sum q)*(V-Kbar) — the invariant that makes
    notional-weighted averaging the RIGHT merge."""
    from plib import Book
    from plib.pricers.di import DiFuture

    trades = pd.DataFrame({
        "instrument_id":   ["lotA", "lotB", "lotC", "t1"],
        "instrument_type": ["di_future"] * 3 + ["commodity_forward"],
        "strike_pu":       [95_000.0, 96_000.0, 95_500.0, None],
        "maturity":        [dt.date(2027, 7, 1)] * 3 + [None],
        "di_curve":        ["USD", "USD", "USD", None],
        # commodity columns for t1
        "strike":       [None] * 3 + [95.0],
        "disc_curve":   [None] * 3 + ["USD"],
        "index":        [None] * 3 + ["BRENT"],
        "ccy_pair":     [None] * 3 + ["USDBRL"],
        "fixing_date":  [None] * 3 + [dt.date(2026, 7, 1)],
        "pay_date":     [None] * 3 + [dt.date(2026, 12, 15)],
    })
    positions = pd.DataFrame({
        "instrument_id": ["lotA", "lotB", "lotC", "t1"],
        "qty":           [300.0, 200.0, -100.0, 10.0],
    })
    book = Book.load(trades, positions)
    before = book.mtm(mkt).collect()
    merged = book.consolidate(DiFuture)
    after = merged.mtm(mkt).collect()

    # one DI row now; qty netted; strike_pu notional-weighted
    di = merged.trades.filter(
        pl.col("instrument_type") == "di_future").collect()
    assert len(di) == 1
    k_bar = (95_000 * 300 + 96_000 * 200 - 95_500 * 100) / 400
    assert abs(di["strike_pu"][0] - k_bar) < 1e-9
    assert di["instrument_id"][0] == "di_future:2027-07-01:USD"  # pin id shape

    # the invariant: total MTM identical; commodity untouched
    assert abs(before["mtm"].sum() - after["mtm"].sum()) < 1e-6
    assert after.filter(pl.col("instrument_id") == "t1")["qty"][0] == 10.0


def test_fungibilidade_closed_position_vanishes(mkt):
    from plib.book import consolidate as consolidate_fn
    from plib.pricers.di import DiFuture
    from plib import Position

    trades = pd.DataFrame({
        "instrument_id": ["a", "b"], "instrument_type": ["di_future"] * 2,
        "strike_pu": [95_000.0, 96_000.0],
        "maturity": [dt.date(2027, 7, 1)] * 2, "di_curve": ["USD"] * 2,
    })
    pos = Position.validate(pd.DataFrame(
        {"instrument_id": ["a", "b"], "qty": [100.0, -100.0]}))
    t2, p2 = consolidate_fn(DiFuture, pl.from_pandas(trades).lazy(), pos)
    assert t2.collect().is_empty() and p2.collect().is_empty()


def test_tree_plugin_and_context(mkt):
    import polars as _pl
    from plib.plugins import MarketContext, Tree, market

    g = Tree("t")

    @g.node
    def trades():
        return _fwd_trades()

    @g.instruments("commodity_forward")     # registry recovery by name
    def values(trades):
        return trades

    direct = price(_fwd_trades(), mkt).collect()["value"][0]
    assert g.run("values", mkt=mkt).collect()["value"][0] == direct

    with MarketContext(data=mkt):
        assert market().ref_date == REF
        assert g.run("values").collect()["value"][0] == direct
        # inherit frames, rebind ref_date; token-reset on exit
        with MarketContext(dt.date(2026, 12, 31)):
            assert market().ref_date == dt.date(2026, 12, 31)
        assert market().ref_date == REF
        # nested bump: shocked inside, base restored outside
        scen = _pl.LazyFrame({"curve_id": ["USD"], "shift": [0.01]})
        with MarketContext.bump(
                Curve, scen,
                apply=(_pl.col("log_df") - _pl.col("shift")).alias("log_df")):
            assert g.run("values").collect()["value"][0] != direct
        assert g.run("values").collect()["value"][0] == direct

    import pytest as _pytest
    with _pytest.raises(LookupError):
        market()                              # no ambient outside the with


def test_ambient_market_is_used_by_core(mkt):
    """price/dv01/Pricer.price/Book.mtm all resolve the ambient context."""
    from plib import Book, MarketContext, dv01
    from plib.pricers.commodity import commodity_forward
    import pytest as _pytest

    trades = _fwd_trades()
    explicit = price(trades, mkt).collect()["value"][0]
    pos = pd.DataFrame({"instrument_id": ["t1"], "qty": [10.0]})

    with MarketContext(data=mkt):
        assert price(trades).collect()["value"][0] == explicit
        assert commodity_forward.price(trades).collect()["value"][0] == explicit
        assert Book.load(trades, pos).mtm().collect()["mtm"][0] == 10.0 * explicit
        assert not dv01(trades).collect().is_empty()

    with _pytest.raises(LookupError):
        price(trades)          # no ambient, no explicit -> loud


def test_market_load_shapes():
    """dict is canonical; pairs (iterable or varargs) normalize into it."""
    df = pd.DataFrame({
        "curve_id": ["X"], "pillar_date": [dt.date(2027, 1, 4)],
        "discount_factor": [0.95]})
    a = Market.load({Curve: df}, ref_date=REF)
    b = Market.load([(Curve, df)], ref_date=REF)
    c = Market.load((Curve, df), ref_date=REF)
    for m in (a, b, c):
        assert Curve in m.frames and m.ref_date == REF


def test_market_named_kwargs_and_registration():
    """Contracts register their name at definition: init like a
    dataclass, open like a registry."""
    import pytest as _pytest
    from plib.contracts.market import MARKET_TYPES, MarketObject
    from typing import ClassVar

    df = pd.DataFrame({"curve_id": ["X"], "pillar_date": [dt.date(2027, 1, 4)],
                       "discount_factor": [0.95]})
    a = Market.load({Curve: df}, ref_date=REF)
    b = Market.load(curve=df, ref_date=REF)
    assert a[Curve].collect().equals(b[Curve].collect())

    with _pytest.raises(TypeError, match="unknown market kwarg"):
        Market.load(unregistered_thing=df)

    class Basis(MarketObject):                     # user contract...
        _key: ClassVar[str] = "name"
        _at: ClassVar[str] = "date"
        name: str
        date: pl.Date
        value: float

    assert MARKET_TYPES["basis"] is Basis          # ...registers itself
    m = Market.load(basis=pd.DataFrame(
        {"name": ["fra-cupom"], "date": [REF], "value": [0.001]}))
    assert Basis in m.frames

    from plib import MarketContext
    with MarketContext(REF, curve=df) as mkt:      # kwargs on the context too
        assert Curve in mkt.frames and mkt.ref_date == REF


def test_dependent_instruments_via_with_data(mkt):
    """Instrument A's output as instrument B's market object."""
    implied = pd.DataFrame({
        "curve_id": ["IMPLIED"], "pillar_date": [dt.date(2027, 7, 1)],
        "rate": [0.11], "anchor_date": [REF]})
    derived = mkt.with_data(curve=implied)
    assert Curve in derived.frames and mkt.frames is not derived.frames
    # original curve rows AND implied rows both present in the derived frame
    ids = derived[Curve].select("curve_id").unique().collect()["curve_id"].to_list()
    assert set(ids) >= {"USD", "IMPLIED"}

    di = pd.DataFrame({
        "instrument_id": ["d"], "instrument_type": ["di_future"],
        "strike_pu": [90_000.0], "maturity": [dt.date(2027, 7, 1)],
        "di_curve": ["IMPLIED"]})
    assert not price(di, derived).collect()["value"].is_null().any()


def test_structure_combinator():
    import pytest as _pytest
    from plib.registry import REGISTRY, price_legs
    from plib.contracts.instruments import BulletSwap
    from plib.pricers.swap import bullet, bullet_swap

    assert bullet_swap(active=2.0, passive=0.5) == 1.5   # plain function

    row = pl.LazyFrame({
        "instrument_id": ["s"], "instrument_type": ["bullet_swap"],
        "end_date": [dt.date(2027, 7, 1)],
        "active_curve": ["DI"], "active_index": ["IPCA"],
        "active_index_base": [100.0], "active_ccy_pair": ["BRLBRL"],
        "active_fx_base": [1.0],
        "passive_curve": ["DI"], "passive_index": ["CDI"],
        "passive_index_base": [100.0], "passive_ccy_pair": ["BRLBRL"],
        "passive_fx_base": [1.0],
    })
    legs = REGISTRY["bullet_swap"].explode(row).collect()
    assert legs["sign"].to_list() == [1.0, 1.0]           # sign is neutral now
    assert sorted(legs["_w"].to_list()) == [-1.0, 1.0]    # weights from the body

    with _pytest.raises(TypeError, match="linear"):
        @price_legs(BulletSwap, bullet)
        def bad(active, passive):
            return active * passive


def _opt_mkt():
    return Market.load(
        curve=pd.DataFrame({
            "curve_id": ["USD"] * 2,
            "pillar_date": [dt.date(2026, 7, 1), dt.date(2027, 7, 1)],
            "discount_factor": [1.0, 0.96]}),
        spot=pd.DataFrame({
            "index": ["BRENT"], "fixing_date": [dt.date(2026, 7, 2)],
            "value": [100.0], "publication_lag_days": [0]}),
        vol_surface=pd.DataFrame({
            "surface_id": ["BRENT-VOL"] * 2,
            "expiry": [dt.date(2026, 7, 1), dt.date(2027, 7, 1)],
            "atm": [0.25, 0.25], "skew": [0.0, 0.0], "curv": [0.0, 0.0]}),
        ref_date=REF)


def test_vanilla_option_matches_hand_black():
    import numpy as np
    from plib.math import norm_cdf
    from plib.pricers.vanilla import vanilla_option

    mkt = _opt_mkt()
    trades = pd.DataFrame({
        "instrument_id": ["OPT-1"], "instrument_type": ["vanilla_option"],
        "cp": [1.0], "strike": [100.0], "expiry": [dt.date(2027, 7, 1)],
        "index": ["BRENT"], "disc_curve": ["USD"], "surface": ["BRENT-VOL"]})
    got = price(trades, mkt).collect()["value"][0]

    tau = int(np.busday_count(REF, dt.date(2027, 7, 1))) / 252
    f = 100.0 / 0.96
    v = 0.25 * math.sqrt(tau)
    d1 = -math.log(100.0 / f) / v + 0.5 * v
    expected = 0.96 * (f * norm_cdf(d1) - 100.0 * norm_cdf(d1 - v))
    assert abs(got - expected) < 1e-9          # same formula, same DSL

    # the calculator is still a plain function of floats
    assert vanilla_option(cp=1.0, strike=1e-9, tau=1.0, spot=100.0,
                          log_df_exp=0.0, atm=0.2, skew=0.0,
                          curv=0.0) > 99.0     # deep ITM call ~ forward


def test_butterfly_is_the_declared_combination():
    mkt = _opt_mkt()
    fly = pd.DataFrame({
        "instrument_id": ["FLY-1"], "instrument_type": ["butterfly"],
        "cp": [1.0], "expiry": [dt.date(2027, 7, 1)],
        "index": ["BRENT"], "disc_curve": ["USD"], "surface": ["BRENT-VOL"],
        "low_strike": [90.0], "mid_strike": [100.0], "high_strike": [110.0]})
    legs = pd.DataFrame({
        "instrument_id": ["a", "b", "c"],
        "instrument_type": ["vanilla_option"] * 3,
        "cp": [1.0] * 3, "strike": [90.0, 100.0, 110.0],
        "expiry": [dt.date(2027, 7, 1)] * 3,
        "index": ["BRENT"] * 3, "disc_curve": ["USD"] * 3,
        "surface": ["BRENT-VOL"] * 3})
    v_fly = price(fly, mkt).collect()["value"][0]
    v = price(legs, mkt).collect().sort("instrument_id")["value"].to_list()
    assert abs(v_fly - (v[0] - 2 * v[1] + v[2])) < 1e-9
    assert v_fly > 0                            # long fly has positive value
