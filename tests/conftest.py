import numpy as np
import pandas as pd
import pytest

from cz_doe import boundary_conditions as bc
from cz_doe.boundary_conditions import CaseParams


@pytest.fixture
def anchor() -> CaseParams:
    return CaseParams(crystal_rpm=8.0, crucible_rpm=-3.0, t_hot=1750.0)


@pytest.fixture
def field(anchor: CaseParams) -> pd.DataFrame:
    """A structured 101 x 81 = 8181-node field that satisfies every boundary condition."""
    r, z = np.meshgrid(np.linspace(0, bc.R_CRUCIBLE, 101), np.linspace(0, bc.MELT_HEIGHT, 81))
    r, z = r.ravel(), z.ravel()
    s, h = r / bc.R_CRUCIBLE, z / bc.MELT_HEIGHT
    top, wall, bottom = np.isclose(z, bc.MELT_HEIGHT), np.isclose(r, bc.R_CRUCIBLE), z == 0

    swirl = anchor.omega_crucible * r * (1 - h) + anchor.omega_crystal * r * h
    swirl[top] = np.where(
        r[top] <= bc.R_CRYSTAL,
        bc.crystal_face_swirl(r[top], anchor),
        bc.free_surface_swirl(r[top], anchor),
    )
    swirl[wall] = anchor.v_crucible_wall

    t = bc.T_BOTTOM_AXIS + s * (anchor.t_hot - bc.T_BOTTOM_AXIS) * (1 - h) + 10 * h
    t[top] = np.where(
        r[top] <= bc.R_CRYSTAL, bc.T_CRYSTAL, bc.free_surface_temperature(r[top], anchor)
    )
    t[wall & ~top] = anchor.t_hot
    t[bottom & ~wall] = bc.crucible_bottom_temperature(r[bottom & ~wall], anchor)

    u_r = 1e-3 * s * (1 - s) * np.sin(np.pi * h)
    u_z = 1e-3 * np.cos(np.pi * s) * h * (1 - h)
    p = 10 * (1 - h)
    return pd.DataFrame({"r": r, "z": z, "u_r": u_r, "u_z": u_z, "u_swirl": swirl, "p": p, "T": t})
