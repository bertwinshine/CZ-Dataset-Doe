import numpy as np

from cz_doe.manifest import LOWER, PARAMS, SOBOL_LOG2_N, UPPER, build_manifest


def test_case_counts_and_ids():
    m = build_manifest()
    assert m.groupby("group").size().to_dict() == {"sobol": 256, "corner": 8, "centre": 1}
    assert m.case_id.is_unique


def test_all_cases_inside_box():
    values = build_manifest()[PARAMS].to_numpy()
    assert (values >= LOWER).all() and (values <= UPPER).all()


def test_sobol_is_stratified_in_every_parameter():
    # A balanced 2**n Sobol set puts exactly one point in each of the 2**n slices of
    # every single parameter.
    m = build_manifest()
    unit = (m[m.group == "sobol"][PARAMS].to_numpy() - LOWER) / (UPPER - LOWER)
    slices = np.floor(unit * 2**SOBOL_LOG2_N).astype(int)
    for column in slices.T:
        assert sorted(column) == list(range(2**SOBOL_LOG2_N))


def test_corners_are_the_box_corners():
    corners = build_manifest().query("group == 'corner'")[PARAMS].to_numpy()
    assert len({tuple(c) for c in corners}) == 8
    assert np.isin(corners, np.vstack([LOWER, UPPER])).all()
