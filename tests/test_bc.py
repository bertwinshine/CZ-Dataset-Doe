import pytest

from cz_doe.bc import CaseParams, crucible_bottom_temperature, free_surface_swirl

ANCHOR = CaseParams(crystal_rpm=8.0, crucible_rpm=-3.0, t_hot=1750.0)


def test_anchor_edge_speeds():
    assert ANCHOR.v_crystal_edge == pytest.approx(0.150796, abs=1e-6)
    assert ANCHOR.v_crucible_wall == pytest.approx(-0.094248, abs=1e-6)


def test_free_surface_ramp_meets_both_walls():
    assert free_surface_swirl(0.18, ANCHOR) == pytest.approx(ANCHOR.v_crystal_edge)
    assert free_surface_swirl(0.30, ANCHOR) == pytest.approx(ANCHOR.v_crucible_wall)


def test_bottom_ramp_ends():
    assert crucible_bottom_temperature(0.0, ANCHOR) == pytest.approx(1700.0)
    assert crucible_bottom_temperature(0.30, ANCHOR) == pytest.approx(1750.0)
