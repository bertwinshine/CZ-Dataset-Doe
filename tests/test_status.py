import numpy as np

from cz_doe.status import plateau_ratio


def test_falling_residual_has_small_ratio():
    continuity = np.logspace(-2, -6, 20000)
    assert plateau_ratio(continuity) < 0.2


def test_oscillating_plateau_has_ratio_near_one():
    i = np.arange(20000)
    continuity = 1e-2 * (1 + 0.3 * np.sin(i / 50))
    assert abs(plateau_ratio(continuity) - 1) < 0.05


def test_short_history_gives_nan():
    assert np.isnan(plateau_ratio(np.ones(100)))
