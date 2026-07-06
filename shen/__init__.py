"""shen — contract-oriented, lazy, vectorized pricing.

Laws:
  1. Instruments and market objects are ROWS under pandera contracts.
     Prep/fixing logic lives in the contract's validate() gate.
  2. A calculator is a pure function of Exprs returning ONE Expr — the
     unit value vector. Math via shen.core.math (backend-portable DSL).
  3. Binding the environment is relational: fold(join) over Lookups.
  4. Scenarios are a dimension, not a loop. Risk reuses price().
  5. Valuation (price) and holdings (positions.mtm/pnl) are separate.
"""

from shen.contracts.conventions import DICurve
from shen.contracts.instruments import BulletSwap, CommodityForward, Instrument
from shen.contracts.market import Curve, Fx, MarketObject, Spot
from shen.core.engine import Market, MarketContext, market, price, unpriced, unresolved
from shen.core.registry import Lookup, price_legs, pricer
from shen.positions import Position, PriceVector, Valuation, basis, mtm, pnl
from shen.positions.book import Book, consolidate
from shen.risk_management import dv01, sensitivities

__all__ = [
    "Book", "BulletSwap", "CommodityForward", "Curve", "DICurve", "Fx", "Instrument",
    "Lookup", "Market", "MarketContext", "MarketObject", "Position", "PriceVector", "Spot",
    "Valuation", "basis", "consolidate", "dv01", "mtm", "pnl", "price", "price_legs",
    "market", "pricer", "sensitivities", "unpriced", "unresolved",
]
