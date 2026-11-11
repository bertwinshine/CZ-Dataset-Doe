"""Sort the finished DOE runs into kept and excluded, with the evidence for each."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cz_doe.check_runs import check_case
from cz_doe.fluent_io import energy_imbalance, read_transcript, residual_history

RESIDUALS = ["res_continuity", "res_u_axial", "res_u_radial", "res_swirl", "res_energy"]
# A steady case drops by orders of magnitude over this many iterations.
WINDOW = 5000
# Converged, but slow enough to sit near the unsteady zone.
SLOW_ITERATIONS = 10000


def plateau_ratio(continuity: np.ndarray) -> float:
    """Median of the last window over the window before; about 1 means it stopped falling."""
    if len(continuity) < 2 * WINDOW:
        return float("nan")
    return float(np.median(continuity[-WINDOW:]) / np.median(continuity[-2 * WINDOW : -WINDOW]))


def case_status(case_id: str, results: Path) -> dict:
    transcript = results / f"{case_id}.trn"
    clean, _ = check_case(transcript)
    log = read_transcript(transcript)
    continuity = residual_history(transcript)[:, 0]
    return {
        "case_id": case_id,
        "status": "kept" if clean else "excluded",
        "iterations": log.iterations,
        **dict(zip(RESIDUALS, log.final_residuals, strict=True)),
        "energy_imbalance": energy_imbalance(results / f"{case_id}_heatflux.txt"),
        "plateau_ratio": plateau_ratio(continuity),
        "slow": clean and log.iterations > SLOW_ITERATIONS,
    }


def build_status(case_matrix: pd.DataFrame, results: Path) -> pd.DataFrame:
    rows = [case_status(case_id, results) for case_id in case_matrix.case_id]
    return case_matrix.merge(pd.DataFrame(rows), on="case_id")


def plot_parameter_space(status: pd.DataFrame, path: Path) -> None:
    labels = {
        "crystal_rpm": "crystal rotation (rpm)",
        "crucible_rpm": "crucible rotation (rpm)",
        "t_hot": "hot-wall temperature (K)",
    }
    pairs = [("crystal_rpm", "crucible_rpm"), ("crystal_rpm", "t_hot"), ("crucible_rpm", "t_hot")]
    kept = status[status.status == "kept"]
    groups = [
        (kept[~kept.slow], "kept", dict(s=14, c="#9aa5b1")),
        (kept[kept.slow], "kept, slow to converge", dict(s=40, c="#e8a33d", marker="s")),
        (
            status[status.status == "excluded"],
            "excluded, unsteady",
            dict(s=60, c="#c8333a", marker="x", linewidths=2),
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (x, y) in zip(axes, pairs, strict=True):
        for df, label, style in groups:
            ax.scatter(df[x], df[y], label=f"{label} ({len(df)})", **style)
        ax.set_xlabel(labels[x])
        ax.set_ylabel(labels[y])
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)
    n_out = (status.status == "excluded").sum()
    fig.suptitle(f"CZ DOE: {len(status)} cases, {len(status) - n_out} kept, {n_out} excluded")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="folder with the .trn and heat-flux files")
    parser.add_argument("--case-matrix", type=Path, default=Path("case_matrix.csv"))
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    status = build_status(pd.read_csv(args.case_matrix), args.results)
    status.to_csv(args.out / "run_status.csv", index=False, float_format="%.6g")
    status[status.status == "excluded"].to_csv(
        args.out / "excluded_cases.csv", index=False, float_format="%.6g"
    )
    plot_parameter_space(status, args.out / "figures" / "doe_parameter_space.png")
    print(status.status.value_counts().to_string())


if __name__ == "__main__":
    main()
