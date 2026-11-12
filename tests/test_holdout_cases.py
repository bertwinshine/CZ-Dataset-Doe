import numpy as np

from cz_doe.case_matrix import LOWER, PARAMS, UPPER
from cz_doe.holdout_cases import build_holdout_cases


def test_case_ids_are_unique():
    cases = build_holdout_cases()
    assert cases.case_id.is_unique


def test_sweeps_stay_inside_the_training_box():
    sweeps = build_holdout_cases().query("group == 'sweep'")[PARAMS].to_numpy()
    assert len(sweeps) == 31
    assert ((sweeps >= LOWER) & (sweeps <= UPPER)).all()


def test_every_ood_case_leaves_the_training_box():
    ood = build_holdout_cases().query("group == 'ood'")[PARAMS].to_numpy()
    assert len(ood) == 12
    assert ((ood < LOWER) | (ood > UPPER)).any(axis=1).all()


def test_sweeps_share_the_v1_anchor():
    cases = build_holdout_cases().set_index("case_id")
    assert np.allclose(cases.loc["sweep_anchor", PARAMS], [8.0, -3.0, 1745.0])
