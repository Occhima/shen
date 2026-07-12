from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shen.core.pricing.engine import compare_values, parallel_curve_shift, price
from shen.core.pricing.registry import Registry
from shen.domain.contracts.market import Curve, MarketContract
from shen.domain.market import Market

type Scenario = Callable[[Market], Market]
type Reprice = Callable[[Any, Market], Any]


def central_difference(up, down, step):
    return (up - down) / (2 * step)


def apply_scenario(market: Market, scenario: Scenario) -> Market:
    shocked = scenario(market)
    if shocked is market:
        raise ValueError("a scenario must return a new Market")
    return shocked


def parallel_curve_scenario(
    *,
    curve: type[MarketContract] = Curve,
    shift: float,
    scenario_id: str,
    basis: float = 1e-4,
) -> Scenario:
    def scenario(market: Market) -> Market:
        return parallel_curve_shift(
            market,
            curve=curve,
            shift=shift,
            scenario_id=scenario_id,
            basis=basis,
        )

    return scenario


def finite_difference(
    terms: Any,
    market: Market,
    *,
    up: Scenario,
    down: Scenario,
    step: float,
    using: Registry,
    name: str,
):
    up_values = price(terms, apply_scenario(market, up), using=using)
    down_values = price(terms, apply_scenario(market, down), using=using)
    return compare_values(up_values, down_values, step=step, name=name)


def dv01(
    terms: Any,
    market: Market,
    *,
    using: Registry,
    curve: type[MarketContract] = Curve,
    bps: float = 1.0,
):
    return finite_difference(
        terms,
        market,
        up=parallel_curve_scenario(
            curve=curve, shift=bps, scenario_id="dv01_up"
        ),
        down=parallel_curve_scenario(
            curve=curve, shift=-bps, scenario_id="dv01_down"
        ),
        step=bps,
        using=using,
        name="dv01",
    )


__all__ = [
    "Scenario",
    "apply_scenario",
    "central_difference",
    "dv01",
    "finite_difference",
    "parallel_curve_scenario",
]
