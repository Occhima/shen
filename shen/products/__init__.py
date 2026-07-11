from shen.pricing.pricer import PricingUniverse
from shen.products.commodities.forward import commodity_forward
from shen.products.fx.ndf import NdfTicket, ParsedTrade, ndf
from shen.products.fx.wdo import wdo_future
from shen.products.options.vanilla import vanilla_option
from shen.products.rates.bullet import bullet
from shen.products.rates.di_future import di_future

STANDARD = PricingUniverse.from_pricers(
    commodity_forward, di_future, wdo_future, bullet, vanilla_option, ndf
)
