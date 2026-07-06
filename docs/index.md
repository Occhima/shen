# Schenberg Risk Toolkit

**Composable, lazy pricing for financial instruments — as symbolic formula graphs.**

Schenberg is a lazy pricing DSL built on Polars LazyFrames, Pandera boundary
schemas, market roles, and a symbolic `FormulaGraph`. Market data is resolved by
`bind` before the graph runs; formulas remain pure and inspectable.

## Five-minute example

```python
from schenberg import FormulaGraph
from schenberg.core.expr import exp

g = FormulaGraph("forward", input=ForwardInput)

@g.formula(symbol="T")
def year_fraction(payment_days):
    return payment_days / 252.0

@g.formula(symbol="DF")
def discount_factor(risk_free_rate, year_fraction):
    return exp(-risk_free_rate * year_fraction)
```

The same declaration can be interpreted as lazy Polars, LaTeX, Mermaid,
`explain()` text, or a staged debug frame.

## Where to start

| | |
|---|---|
| **[Concepts](concepts.md)** | Formula graphs, market binding, staging, and positions. |
| **[Extending](extending.md)** | Add a pricer, fixing convention, or position measure. |
| **[Examples](examples/01_forward_pricer.html)** | Marimo examples exported as mobile-friendly HTML. |
| **[API Reference](api/index.md)** | Auto-generated from source docstrings. |

## Architecture

```text
domain/        Pandera boundary schemas and contract rules
core/          Expr, FormulaGraph, Formula, Router, Fold
market_data/   MarketSnapshot, sources, roles, Fixing, Shock
position/      PositionView, reusable measures, book rollups
```

Pure pricing graphs never contain position direction. Position views and folds own
`side`, `quantity`, reporting FX, and aggregation.
