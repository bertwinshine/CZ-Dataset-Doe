import numpy as np
import pandas as pd

from cz_doe.mesh_study import gci, probe


def test_gci_recovers_a_known_second_order_error():
    # f = 1 + 0.01 * h**2 on cells of size 1, 2, 4: exact order 2, exact answer 1.
    fine, base, coarse = 1.01, 1.04, 1.16
    result = gci(fine, base, coarse, r_fine_base=2.0, r_base_coarse=2.0)
    assert np.isclose(result["order"], 2.0)
    assert np.isclose(result["extrapolated"], 1.0)
    assert np.isclose(result["gci_base"], 1.25 * (0.12 / 1.04) / 3)
    # Each GCI is relative to its own mesh value, so exact data gives fine / base, not 1.
    assert np.isclose(result["asymptotic"], fine / base)


def test_gci_is_undefined_when_the_change_reverses():
    result = gci(1.02, 1.04, 1.01, r_fine_base=2.0, r_base_coarse=2.0)
    assert not result["monotonic"]
    assert np.isnan(result["gci_base"])


def test_probe_interpolates_between_nodes():
    r, z = np.meshgrid(np.linspace(0, 0.3, 31), np.linspace(0, 0.15, 16))
    field = pd.DataFrame({"r": r.ravel(), "z": z.ravel()})
    field["T"] = 1700 + 100 * field.r + 50 * field.z
    assert np.isclose(probe(field, "T", 0.155, 0.0725), 1700 + 15.5 + 3.625)
