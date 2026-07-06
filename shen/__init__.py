"""plib — contract-oriented, lazy, vectorized pricing.

Laws:
  1. Instruments and market objects are ROWS under pandera contracts.
     Prep/fixing logic lives in the contract's validate() gate.
  2. A calculator is a pure function of Exprs returning ONE Expr — the
     unit value vector. Math via plib.math (backend-portable DSL).
  3. Binding the environment is relational: fold(join) over Lookups.
  4. Scenarios are a dimension, not a loop. Risk reuses price().
  5. Valuation (price) and holdings (positions.mtm/pnl) are separate.
"""

from plib.book import Book, consolidate
from plib.contracts.instruments import BulletSwap, CommodityForward, Instrument
from plib.contracts.market import Curve, Fx, MarketObject, Spot
from plib.engine import Market, MarketContext, market, price, unpriced, unresolved
from plib.positions import Position, PriceVector, Valuation, basis, mtm, pnl
from plib.registry import Lookup, price_legs, pricer
from plib.risk import dv01, sensitivities

__all__ = [
    "Book", "BulletSwap", "CommodityForward", "Curve", "Fx", "Instrument",
    "Lookup", "Market", "MarketContext", "MarketObject", "Position", "PriceVector", "Spot",
    "Valuation", "basis", "consolidate", "dv01", "mtm", "pnl", "price", "price_legs",
    "market", "pricer", "sensitivities", "unpriced", "unresolved",
]
