from shen.core.registry import PricingUniverse
from shen.products.commodities.forward import commodity_forward
from shen.products.fx.ndf import NdfTicket, ParsedTrade, ndf
from shen.products.fx.wdo import wdo_future
from shen.products.options.vanilla import butterfly, vanilla_option
from shen.products.rates.bullet import bullet, bullet_swap
from shen.products.rates.di_future import di_future

STANDARD = PricingUniverse.from_pricers(
    commodity_forward,
    di_future,
    wdo_future,
    bullet,
    bullet_swap,
    vanilla_option,
    butterfly,
    ndf,
)
