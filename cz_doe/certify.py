"""Certify dataset CSVs against the boundary conditions their manifest row prescribes."""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from cz_doe import bc
from cz_doe.bc import CaseParams
from cz_doe.fluent_io import FIELDS

N_NODES = 8181
REL_TOL_WALL_SWIRL = 1e-3
ABS_TOL_RAMP_SWIRL = 1e-3  # m/s
# Node values on the crystal face are interpolated from the wall, so this is looser.
REL_TOL_CRYSTAL_SWIRL = 5e-3
TOL_T = 0.05  # K
TOL_T_BOTTOM = 0.2  # K
MAX_AXIS_RATIO = 0.01
GEOM_EPS = 1e-6


class Check(NamedTuple):
    name: str
    passed: bool
    detail: str


def boundary_masks(d: pd.DataFrame) -> dict[str, pd.Series]:
    top = np.isclose(d.z, bc.MELT_HEIGHT, atol=1e-9)
    # Corner nodes shared by two boundaries are blended by the solver, so they are excluded.
    return {
        "free": top & (d.r > bc.R_CRYSTAL + GEOM_EPS),
        "crystal": top & (d.r < bc.R_CRYSTAL - GEOM_EPS),
        "wall": np.isclose(d.r, bc.R_CRUCIBLE, atol=1e-9)
        & (d.z > GEOM_EPS)
        & (d.z < bc.MELT_HEIGHT - GEOM_EPS),
        "bottom": (d.z < 1e-12) & (d.r > GEOM_EPS) & (d.r < bc.R_CRUCIBLE - GEOM_EPS),
        "axis": d.r < 1e-9,
    }


def swirl_checks(d: pd.DataFrame, m: dict[str, pd.Series], p: CaseParams) -> list[Check]:
    wall_speed = abs(p.v_crucible_wall)
    edge_speed = abs(p.v_crystal_edge)
    free, crystal, bottom = d[m["free"]], d[m["crystal"]], d[m["bottom"]]
    wall_err = np.abs(d[m["wall"]].u_swirl - p.v_crucible_wall).max() / wall_speed
    bottom_err = np.abs(bottom.u_swirl - bc.crucible_bottom_swirl(bottom.r, p)).max() / wall_speed
    ramp_err = np.abs(free.u_swirl - bc.free_surface_swirl(free.r, p)).max()
    crystal_err = np.abs(crystal.u_swirl - bc.crystal_face_swirl(crystal.r, p)).max() / edge_speed
    return [
        Check("crucible wall swirl", wall_err < REL_TOL_WALL_SWIRL, f"{wall_err:.2e} rel"),
        Check("crucible bottom swirl", bottom_err < REL_TOL_WALL_SWIRL, f"{bottom_err:.2e} rel"),
        Check("free surface swirl", ramp_err < ABS_TOL_RAMP_SWIRL, f"{ramp_err:.2e} m/s"),
        Check("crystal face swirl", crystal_err < REL_TOL_CRYSTAL_SWIRL, f"{crystal_err:.2e} rel"),
    ]


def thermal_checks(d: pd.DataFrame, m: dict[str, pd.Series], p: CaseParams) -> list[Check]:
    free, bottom = d[m["free"]], d[m["bottom"]]
    free_err = np.abs(free["T"] - bc.free_surface_temperature(free.r, p)).max()
    bottom_err = np.abs(bottom["T"] - bc.crucible_bottom_temperature(bottom.r, p)).max()
    t_min, t_max = d["T"].min(), d["T"].max()
    in_bounds = abs(t_min - bc.T_CRYSTAL) < TOL_T and abs(t_max - p.t_hot) < TOL_T
    return [
        Check("free surface T", free_err < TOL_T, f"{free_err:.3f} K"),
        Check("crucible bottom T", bottom_err < TOL_T_BOTTOM, f"{bottom_err:.3f} K"),
        Check("T bounds", in_bounds, f"{t_min:.2f}-{t_max:.2f} K"),
    ]


def sanity_checks(d: pd.DataFrame, m: dict[str, pd.Series]) -> list[Check]:
    axis = d[m["axis"]]
    # u_r and u_swirl must vanish on the axis; a transposed u_r/u_z export also fails here.
    ur_ratio = np.abs(axis.u_r).max() / np.abs(d.u_r).max()
    swirl_ratio = np.abs(axis.u_swirl).max() / np.abs(d.u_swirl).max()
    # Swapped r/z columns would put the crucible wall at 0.15 m instead of 0.30 m.
    extent_ok = np.isclose(d.r.max(), bc.R_CRUCIBLE) and np.isclose(d.z.max(), bc.MELT_HEIGHT)
    return [
        Check("schema", list(d.columns) == FIELDS, ", ".join(d.columns)),
        Check("node count", len(d) == N_NODES, f"{len(d)} rows"),
        Check(
            "domain extent", bool(extent_ok), f"r max {d.r.max():.3f} m, z max {d.z.max():.3f} m"
        ),
        Check("finite values", bool(np.isfinite(d.to_numpy()).all()), ""),
        Check("axis u_r -> 0", ur_ratio < MAX_AXIS_RATIO, f"{ur_ratio:.3%} of max"),
        Check("axis swirl -> 0", swirl_ratio < MAX_AXIS_RATIO, f"{swirl_ratio:.3%} of max"),
    ]


def certify(field: pd.DataFrame, p: CaseParams) -> list[Check]:
    masks = boundary_masks(field)
    return (
        sanity_checks(field, masks)
        + swirl_checks(field, masks, p)
        + thermal_checks(field, masks, p)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cz_manifest.csv"))
    parser.add_argument("data", type=Path, help="folder with the converted case CSVs")
    args = parser.parse_args()

    csv_files = sorted(args.data.glob("*.csv"))
    if not csv_files:
        sys.exit(f"no .csv files in {args.data}")
    manifest = pd.read_csv(args.manifest).set_index("case_id")
    failed = 0
    for path in csv_files:
        if path.stem not in manifest.index:
            failed += 1
            print(f"REJECTED {path.stem}: no manifest row\n")
            continue
        row = manifest.loc[path.stem]
        checks = certify(
            pd.read_csv(path), CaseParams(row.crystal_rpm, row.crucible_rpm, row.t_hot)
        )
        ok = all(c.passed for c in checks)
        failed += not ok
        print(f"{'CERTIFIED' if ok else 'REJECTED'} {path.stem}")
        for c in checks:
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name:<22} {c.detail}")
        print()
    print(f"{len(csv_files) - failed} of {len(csv_files)} certified")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
