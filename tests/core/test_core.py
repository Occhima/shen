import shen
from shen.products.fx.ndf import ndf


def test_pricer_is_object_not_mutated_function():
    assert isinstance(ndf, shen.Pricer)
    assert ndf.requirements().product_type == "ndf"


def test_standard_universe_explicit():
    assert "ndf" in shen.STANDARD.pricers
