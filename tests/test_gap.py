from analytics.gaps import GapAnalyzer


def test_small_gap_threshold():
    ga = GapAnalyzer()
    prev_close = 100.0
    open_price = 100.10  # 0.10% gap
    assert ga.is_small_gap(open_price, prev_close, threshold=0.0019) is True
