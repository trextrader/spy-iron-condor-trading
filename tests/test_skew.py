from analytics.skew import SkewCalculator


def test_skew_penalty_triggers():
    skew_calc = SkewCalculator()
    skew = skew_calc.skew_metric(iv_put=0.30, iv_call=0.20, iv_atm=0.25)
    assert skew > 0.0
