"""plib.plugins — optional layers over the explicit core.

Nothing here changes core semantics: Market stays a frozen value,
price(trades, mkt) stays explicit. Plugins add ergonomics for
pipelines: ambient market context and networkx pricing trees.
Requires the `tree` extra: pip install plib[tree]
"""

from plib.plugins.context import MarketContext, market
from plib.plugins.tree import Tree

__all__ = ["MarketContext", "Tree", "market"]
