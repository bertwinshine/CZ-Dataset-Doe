"""Compare the coarse, base and fine meshes case by case and compute the grid convergence index."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from cz_doe.fluent_io import FIELDS

MESHES = ["coarse", "base", "fine"]
# Probe points sit mid-depth, away from boundaries where node values are blended.
PROBES = {
    "uz_axis_mid": ("u_z", 0.0, 0.075),
    "swirl_probe": ("u_swirl", 0.24, 0.075),
    "T_probe": ("T", 0.15, 0.075),
}
# Celik et al. (2008), J. Fluids Eng. 130, 078001.
SAFETY_FACTOR = 1.25


def probe(field: pd.DataFrame, name: str, r: float, z: float) -> float:
    points = field[["r", "z"]].to_numpy()
    return float(griddata(points, field[name].to_numpy(), [(r, z)], method="linear")[0])


def quantities(field: pd.DataFrame) -> dict[str, float]:
    values = {
        "max_uz": field.u_z.abs().max(),
        "max_ur": field.u_r.abs().max(),
        "p_range": field.p.max() - field.p.min(),
    }
    values.update({key: probe(field, *spec) for key, spec in PROBES.items()})
    return values


def gci(fine: float, base: float, coarse: float, r_fine_base: float, r_base_coarse: float) -> dict:
    """Celik's three-mesh GCI; undefined when the change does not keep its direction."""
    e21, e32 = base - fine, coarse - base
    if e21 == 0 or e32 / e21 < 0:
        nan = float("nan")
        return dict(monotonic=False, order=nan, extrapolated=nan, gci_base=nan, asymptotic=nan)
    ratio = np.log(e32 / e21)
    order = abs(ratio) / np.log(r_fine_base)
    for _ in range(100):
        shift = np.log((r_fine_base**order - 1) / (r_base_coarse**order - 1))
        order = abs(ratio + shift) / np.log(r_fine_base)
    gain = r_fine_base**order
    gci_fine = SAFETY_FACTOR * abs(e21 / fine) / (gain - 1)
    gci_base = SAFETY_FACTOR * abs(e32 / base) / (r_base_coarse**order - 1)
    return dict(
        monotonic=True,
        order=order,
        extrapolated=(gain * fine - base) / (gain - 1),
        gci_base=gci_base,
        asymptotic=gci_base / (gain * gci_fine),
    )


def field_difference(field: pd.DataFrame, reference: pd.DataFrame, name: str) -> float:
    """RMS difference on the reference nodes, relative to the field's own size."""
    values = griddata(
        field[["r", "z"]].to_numpy(), field[name].to_numpy(), reference[["r", "z"]].to_numpy()
    )
    ref = reference[name].to_numpy()
    # Temperature is ~1700 K everywhere, so scale by its variation, not its magnitude.
    scale = np.std(ref) if name == "T" else np.sqrt(np.mean(ref**2))
    return float(np.sqrt(np.nanmean((values - ref) ** 2)) / scale)


def compare_case(fields: dict[str, pd.DataFrame], ratios: tuple[float, float]) -> list[dict]:
    q = {mesh: quantities(fields[mesh]) for mesh in MESHES}
    rows = []
    for name in q["base"]:
        fine, base, coarse = q["fine"][name], q["base"][name], q["coarse"][name]
        rows.append(
            dict(quantity=name, coarse=coarse, base=base, fine=fine)
            | gci(fine, base, coarse, *ratios)
            | dict(change_base_to_fine=abs(fine - base) / abs(fine))
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for mesh in MESHES:
        parser.add_argument(f"--{mesh}", type=Path, required=True, help=f"{mesh} dataset folder")
    parser.add_argument("--cells", type=int, nargs=3, default=[2000, 8000, 31840])
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    cells = dict(zip(MESHES, args.cells, strict=True))
    ratios = (np.sqrt(cells["fine"] / cells["base"]), np.sqrt(cells["base"] / cells["coarse"]))
    quantity_rows, field_rows = [], []
    for path in sorted(args.fine.glob("*.csv")):
        fields = {mesh: pd.read_csv(getattr(args, mesh) / path.name) for mesh in MESHES}
        quantity_rows += [dict(case_id=path.stem) | row for row in compare_case(fields, ratios)]
        for mesh in ["coarse", "fine"]:
            diffs = {f: field_difference(fields[mesh], fields["base"], f) for f in FIELDS[2:]}
            field_rows.append(dict(case_id=path.stem, mesh=mesh) | diffs)

    quantities_df, fields_df = pd.DataFrame(quantity_rows), pd.DataFrame(field_rows)
    quantities_df.to_csv(args.out / "mesh_study_quantities.csv", index=False, float_format="%.6g")
    fields_df.to_csv(args.out / "mesh_study_fields.csv", index=False, float_format="%.6g")
    summary = quantities_df.groupby("quantity")[["gci_base", "change_base_to_fine"]]
    print(summary.agg(["median", "max"]).to_string(float_format=lambda v: f"{v:.2%}"))
    print(
        fields_df.groupby("mesh")[FIELDS[2:]].median().to_string(float_format=lambda v: f"{v:.2%}")
    )


if __name__ == "__main__":
    main()
