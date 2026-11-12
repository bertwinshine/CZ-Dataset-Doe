"""Build the hold-out cases: the v1 one-parameter sweeps and cases outside the training box."""

import argparse
from pathlib import Path

import pandas as pd

from cz_doe.case_matrix import LOWER, PARAMS, UPPER

# Rerun at the exact v1 values, so the v1 data (stopped at 1e-3) can be checked against 1e-6.
V1_ANCHOR = (8.0, -3.0, 1745.0)
V1_CRYSTAL = [4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]
V1_CRUCIBLE = [-1, -2, -3, -3.5, -4, -4.5, -5, -5.5, -6, -7, -8, -10]
V1_T_HOT = [1730, 1740, 1745, 1755, 1760, 1765, 1770, 1775, 1780, 1785]

CENTRE = tuple((LOWER + UPPER) / 2)
# Two values just past each limit, one far but inside industrial practice where a source exists:
# crystal 5-30 rpm and crucible 0.5-10 rpm (US 11408090). The 1810 K wall has no such source.
OOD_CRYSTAL = [22, 24, 30]
OOD_CRUCIBLE = [-11, -12, -0.5]
OOD_T_HOT = [1790, 1795, 1810]
OOD_MIXED = [(22, -11, 1790), (24, -12, 1795), (30, -12, 1810)]


def _label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def sweep_cases() -> list[tuple]:
    crystal, crucible, t_hot = V1_ANCHOR
    rows = [("sweep_anchor", *V1_ANCHOR)]
    rows += [
        (f"sweep_crystal_{_label(v)}", v, crucible, t_hot) for v in V1_CRYSTAL if v != crystal
    ]
    rows += [
        (f"sweep_crucible_{_label(v)}", crystal, v, t_hot) for v in V1_CRUCIBLE if v != crucible
    ]
    rows += [(f"sweep_t_hot_{_label(v)}", crystal, crucible, v) for v in V1_T_HOT if v != t_hot]
    return [("sweep", *row) for row in rows]


def ood_cases() -> list[tuple]:
    crystal, crucible, t_hot = CENTRE
    rows = [(f"ood_crystal_{_label(v)}", v, crucible, t_hot) for v in OOD_CRYSTAL]
    rows += [(f"ood_crucible_{_label(v)}", crystal, v, t_hot) for v in OOD_CRUCIBLE]
    rows += [(f"ood_t_hot_{_label(v)}", crystal, crucible, v) for v in OOD_T_HOT]
    rows += [(f"ood_mixed_{i}", *values) for i, values in enumerate(OOD_MIXED)]
    return [("ood", *row) for row in rows]


def build_holdout_cases() -> pd.DataFrame:
    rows = [(case_id, group, *values) for group, case_id, *values in sweep_cases() + ood_cases()]
    return pd.DataFrame(rows, columns=["case_id", "group", *PARAMS])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("holdout_cases.csv"))
    args = parser.parse_args()

    cases = build_holdout_cases()
    cases.to_csv(args.out, index=False, float_format="%.10g")
    print(f"wrote {len(cases)} cases to {args.out}")
    print(cases.groupby("group").size().to_string())


if __name__ == "__main__":
    main()
