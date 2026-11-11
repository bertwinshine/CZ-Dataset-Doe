"""Build the case manifest: a scrambled Sobol design plus the corners and centre of the box."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

PARAMS = ["crystal_rpm", "crucible_rpm", "t_hot"]
LOWER = np.array([4.0, -10.0, 1730.0])
UPPER = np.array([20.0, -1.0, 1785.0])

# 2**8 = 256 points; Sobol balance only holds at powers of two.
SOBOL_LOG2_N = 8
SOBOL_SEED = 42


def sobol_cases() -> pd.DataFrame:
    sampler = qmc.Sobol(d=len(PARAMS), scramble=True, seed=SOBOL_SEED)
    points = qmc.scale(sampler.random_base2(SOBOL_LOG2_N), LOWER, UPPER)
    return _frame("sobol", points)


def corner_cases() -> pd.DataFrame:
    return _frame("corner", np.array(list(itertools.product(*zip(LOWER, UPPER, strict=True)))))


def centre_case() -> pd.DataFrame:
    return _frame("centre", ((LOWER + UPPER) / 2)[None, :])


def _frame(group: str, points: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(points, columns=PARAMS)
    df.insert(0, "case_id", [f"{group}_{i:03d}" for i in range(len(df))])
    df.insert(1, "group", group)
    return df


def build_manifest() -> pd.DataFrame:
    """All cases of the DOE; case_id doubles as the file stem of every output."""
    return pd.concat([sobol_cases(), corner_cases(), centre_case()], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("cz_manifest.csv"))
    args = parser.parse_args()

    manifest = build_manifest()
    manifest.to_csv(args.out, index=False, float_format="%.10g")
    print(f"wrote {len(manifest)} cases to {args.out}")
    print(manifest.groupby("group").size().to_string())


if __name__ == "__main__":
    main()
