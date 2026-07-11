class ShenError(Exception):
    pass


class ContractError(ShenError):
    pass


class UnknownProductTypeError(ShenError):
    pass


class MissingMarketObjectError(ShenError):
    pass


class MissingMarketDataError(ShenError):
    pass


class LookupResolutionError(ShenError):
    pass


class ExtrapolationError(LookupResolutionError):
    pass


class StaleMarketDataError(ShenError):
    pass


class DuplicateQuoteError(ShenError):
    pass


class UnitMismatchError(ShenError):
    pass


class CurrencyMismatchError(ShenError):
    pass


class PricingDomainError(ShenError):
    pass


class ScenarioError(ShenError):
    pass
