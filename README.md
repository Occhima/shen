<div align="center">

# Schenberg Risk Toolkit

**Lazy pricing DSL built on symbolic formulas, Polars expressions, and typed boundary schemas.**

[Concepts](docs/concepts.md) · [Extending](docs/extending.md) · [Examples](docs/examples/)

</div>

Schenberg is a compact pricing DSL. Market data is resolved into ordinary input
columns before a formula graph runs; the graph itself is a pure symbolic program.
Pricing functions return lazy frames and do not execute trade-side queries until
the caller collects.

## Minimal forward example

```python
from datetime import date
import polars as pl

from schenberg import Formula, MarketSnapshot, With, bind, exp, market_role
from schenberg.domain.base import SchenbergDataFrameModel

ForwardRate = (
    market_role("forward_rate")
    .read("curves", "forward_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)
RiskFreeRate = (
    market_role("risk_free_rate")
    .read("curves", "risk_free_rate")
    .by(indexer="id_indexador", payment_days="tenor_days")
)

class ForwardInput(With[ForwardRate], With[RiskFreeRate], SchenbergDataFrameModel):
    instrument_id: str
    indexer: str
    currency: str
    strike: float
    payment_days: int

class ForwardOutput(SchenbergDataFrameModel):
    instrument_id: str
    future_value: float
    present_value: float
    value: float
    delta: float
    currency: str

g = Formula[ForwardInput, ForwardOutput]("forward")

@g.formula(symbol="T")
def year_fraction(payment_days):
    return payment_days / 252.0

@g.formula(symbol="DF")
def discount_factor(risk_free_rate, year_fraction):
    return exp(-risk_free_rate * year_fraction)

@g.formula(symbol="FV")
def future_value(forward_rate, strike):
    return forward_rate - strike

@g.formula(symbol="PV")
def present_value(future_value, discount_factor):
    return future_value * discount_factor

@g.formula(symbol="Delta")
def delta(discount_factor):
    return discount_factor

g.returns(
    "output",
    ForwardOutput,
    instrument_id="instrument_id",
    future_value="future_value",
    present_value="present_value",
    value="present_value",
    delta="delta",
    currency="currency",
)

trades = pl.DataFrame({
    "instrument_id": ["FWD-1"],
    "indexer": ["DI"],
    "currency": ["BRL"],
    "strike": [100.0],
    "payment_days": [252],
}).lazy()
market = (
    MarketSnapshot.at(date(2026, 6, 6))
    .source(
        "curves",
        pl.DataFrame({
            "id_indexador": ["DI"],
            "tenor_days": [252],
            "forward_rate": [112.0],
            "risk_free_rate": [0.10],
        }),
        unique_by=("id_indexador", "tenor_days"),
    )
    .build()
)

result = g.plan(bind(trades, market, ForwardInput), view="output")  # LazyFrame
print(result.collect())
```

## Formula decorator

`@g.formula(...)` is the ergonomic API for registering terms. The default term
name is the Python function name; `name=`, `symbol=`, `description=`, `tags=` and
`dtype=` are supported. Dependencies are declared as **headless parameters**:
each argument name is resolved to a symbolic `var` — from an earlier term first,
otherwise from the graph's input schema (contract *and* pre-resolved market
columns alike). So a formula reads like the math it represents — `def
present_value(future_value, discount_factor): ...` — with no `c.`/`contract.`
indirection. With an input schema declared, an unknown parameter fails fast at
definition time. The legacy namespace names `c`, `contract`, `input` and
`inputs` still receive the whole input namespace for backward compatibility.

The decorated function returns Schenberg `Expr` nodes, not opaque Python UDFs, so
introspection and compilation still work:

- `formula.formulas()` and `formula.formula_of(...)` derive LaTeX from the symbolic IR.
- `formula.explain(...)`, `formula.to_mermaid(...)`, `formula.info(...)`, and
  `formula.stage(...)` remain available.
- `formula.plan(...)` compiles to lazy Polars expressions and does not call
  `.collect()`.
- `g.let(...)` remains the lower-level primitive for manually registering a term.

## Position layer

Pricing returns pure per-instrument values and risks: no `side`, no book, no
comprado/vendido. The position layer lifts those pure values onto a `Position`:

```python
from schenberg.position import position_value, position_risk, book_value_rollup

pv = position_value(positions, value=instrument_values, book=book, fx=fx)
pr = position_risk(positions, risk=instrument_risk)
rollup = book_value_rollup.compute(pv)
```

## Design rules

- Market joins are declared as `With[role]` mixins and resolved by `bind` before
  the formula graph runs.
- Pure pricing graphs compute own-currency instrument values and sensitivities
  only; position direction belongs to the position layer.
- Missing inputs fail loudly at plan time.
- Examples define their pricers locally with the public API; Schenberg does not
  centralise those example pricers in a pricing API module.
- Example notebooks are Quarto `.qmd` files rendered with `quarto render`.

## Example notebooks (Quarto)

The examples are [Quarto](https://quarto.org/) notebooks that render to
standalone HTML, leaning on Quarto's native LaTeX (MathJax) and Mermaid so the
graph's own `to_latex()` / `to_mermaid()` output renders as real math and
diagrams:

| Notebook | Shows |
|---|---|
| `01_forward_pricer.qmd` | A formula graph from scratch — headless params, semantic roles, lazy plan |
| `02_vanilla_option.qmd` | Black-Scholes price & Greeks, then the same graph swept over a spot ladder |
| `03_autodiff_greeks.qmd` | One `Expr` → Polars + JAX + LaTeX; autodiff **vanna & volga** reconciled vs finite differences |
| `04_scenario_var.qmd` | Shocks, named stresses, and a historical **VaR/ES** via `reprice_under` |
| `05_quantlib_benchmark.qmd` | Price/delta reconciled against **QuantLib**, plus vectorized throughput |
| `06_equity_book.qmd` | A B3 equity+option book: a `Router` over two pricers, a custom `PositionView` of measures, a `Fold` roll-up, and a market stress — full stack, end to end |

Render them all to HTML (needs the [Quarto CLI](https://quarto.org/docs/get-started/)):

```bash
uv run poe examples-html      # render docs/examples/*.qmd -> *.html
uv run poe examples-preview   # live preview while editing
```

## Install and check

```bash
uv sync --all-groups
uv run pytest
uv run poe check
```

## Semantic market roles and option example

`Formula[Input, Output]` sees only resolved parameters. Semantic helpers such as `CURVES`,
`FIXINGS`, and `VOLS` build `MarketRole` declarations outside the graph; `bind(...)`
uses those roles to enrich a trade frame, and `formula.plan(...)` prices lazily.
Pandera validation belongs at public pricer/schema boundaries (for example via
`@price_function`), not inside formula math.

```python
from schenberg import CURVES, FIXINGS, VOLS, Formula, With, bind

Spot = FIXINGS.value("USD/BRL", as_="spot").source("fixings")
Vol = (
    VOLS.implied("USD/BRL", as_="vol")
    .source("vol_surface")
    .for_expiry("expiry")
    .for_strike("strike")
)
RiskFree = CURVES.zero_rate("BRL_DI", as_="risk_free_rate").source("curves").for_tenor(
    "payment_days"
)

class VanillaOptionInput(With[Spot], With[Vol], With[RiskFree], SchenbergDataFrameModel):
    ...

enriched = bind(trades, market, VanillaOptionInput)
priced = option_formula.plan(enriched, view="output")  # LazyFrame
```

See `docs/examples/02_vanilla_option.qmd` for Black-Scholes price/Greeks and
`docs/examples/03_autodiff_greeks.qmd` for autodiff vanna/volga rendered from the
same symbolic formula.
