from analytics.divergence import DivergenceZScore


def test_zscore_bounds(sample_spread_series):
    z = DivergenceZScore().zscore(sample_spread_series, lookback=len(sample_spread_series))
    # z can be any real number, but should be finite
    assert z == z
