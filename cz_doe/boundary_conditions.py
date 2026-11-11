"""Boundary conditions of the CZ melt model, derived from a case's three parameters."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

R_CRUCIBLE = 0.30
R_CRYSTAL = 0.18
MELT_HEIGHT = 0.15
T_CRYSTAL = 1685.0
T_BOTTOM_AXIS = 1700.0
RPM_TO_RAD_S = 2.0 * np.pi / 60.0


@dataclass(frozen=True)
class CaseParams:
    crystal_rpm: float
    crucible_rpm: float
    t_hot: float

    @property
    def omega_crystal(self) -> float:
        return self.crystal_rpm * RPM_TO_RAD_S

    @property
    def omega_crucible(self) -> float:
        return self.crucible_rpm * RPM_TO_RAD_S

    @property
    def v_crystal_edge(self) -> float:
        return self.omega_crystal * R_CRYSTAL

    @property
    def v_crucible_wall(self) -> float:
        return self.omega_crucible * R_CRUCIBLE


def _free_surface_fraction(r: ArrayLike) -> np.ndarray:
    return (np.asarray(r, float) - R_CRYSTAL) / (R_CRUCIBLE - R_CRYSTAL)


def free_surface_swirl(r: ArrayLike, p: CaseParams) -> np.ndarray:
    """Linear swirl ramp from the crystal edge to the crucible wall."""
    return p.v_crystal_edge + _free_surface_fraction(r) * (p.v_crucible_wall - p.v_crystal_edge)


def free_surface_temperature(r: ArrayLike, p: CaseParams) -> np.ndarray:
    return T_CRYSTAL + _free_surface_fraction(r) * (p.t_hot - T_CRYSTAL)


def crystal_face_swirl(r: ArrayLike, p: CaseParams) -> np.ndarray:
    return p.omega_crystal * np.asarray(r, float)


def crucible_bottom_swirl(r: ArrayLike, p: CaseParams) -> np.ndarray:
    return p.omega_crucible * np.asarray(r, float)


def crucible_bottom_temperature(r: ArrayLike, p: CaseParams) -> np.ndarray:
    return T_BOTTOM_AXIS + np.asarray(r, float) / R_CRUCIBLE * (p.t_hot - T_BOTTOM_AXIS)
