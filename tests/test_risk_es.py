"""
Placeholder tests for Expected Shortfall (will be implemented in Phase 5).
"""

def test_es_tail_mean():
    # stub: ES should be <= 0 for loss distribution
    # TODO: implement when risk/expected_shortfall.py exists
    pass


def test_es_rejects():
    # stub: illustrates policy check pattern
    proposed_es = -2000.0
    limit = -1000.0
    assert (proposed_es < limit) is True
