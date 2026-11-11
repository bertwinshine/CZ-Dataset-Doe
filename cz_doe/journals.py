"""Write one Fluent journal per case in the case matrix, plus run_all.jou to run them all."""

import argparse
from pathlib import Path

import pandas as pd

from cz_doe.boundary_conditions import CaseParams

MAX_ITERATIONS = 30000
RESIDUAL_TARGET = "1e-06"
N_EQUATIONS = 5  # continuity, x-velocity, y-velocity, swirl, energy

# Every command was checked interactively in the Fluent 2026 R1 console. The trailing
# "q" closes the named-expression editor; "() ... () no" means whole domain, end of the
# variable list, node values (not cell-centred).
CASE_TEMPLATE = """\
/file/start-transcript "{results}/{case_id}.trn"
/file/read-case "{base_case}"
/define/named-expressions/edit "omega_cry" definition "{omega_cry:.9f} [s^-1]" q
/define/named-expressions/edit "omega_cru" definition "{omega_cru:.9f} [s^-1]" q
/define/named-expressions/edit "T_hot" definition "{t_hot:.10g} [K]" q
/solve/monitors/residual/convergence-criteria {criteria}
/solve/initialize/hyb-initialization
/solve/iterate {max_iterations}
/file/write-case-data "{results}/{case_id}.cas.h5"
/report/fluxes/heat-transfer yes yes "{results}/{case_id}_heatflux.txt"
/file/export/ascii "{results}/{case_id}_raw.txt" () yes pressure temperature \
axial-velocity radial-velocity swirl-velocity () no
/file/stop-transcript
"""

# An unanswered "overwrite?" prompt would stall an unattended batch.
HEADER = "/file/confirm-overwrite? no\n"


def fluent_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def case_journal(row: pd.Series, base_case: str | Path, results: str | Path) -> str:
    p = CaseParams(row.crystal_rpm, row.crucible_rpm, row.t_hot)
    return CASE_TEMPLATE.format(
        case_id=row.case_id,
        base_case=fluent_path(base_case),
        results=fluent_path(results),
        omega_cry=p.omega_crystal,
        omega_cru=p.omega_crucible,
        t_hot=p.t_hot,
        criteria=" ".join([RESIDUAL_TARGET] * N_EQUATIONS),
        max_iterations=MAX_ITERATIONS,
    )


def write_journals(case_matrix: pd.DataFrame, base_case: str, results: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bodies = []
    for _, row in case_matrix.iterrows():
        body = case_journal(row, base_case, results)
        (out_dir / f"{row.case_id}.jou").write_text(HEADER + body)
        bodies.append(body)
    (out_dir / "run_all.jou").write_text(HEADER + "\n".join(bodies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-matrix", type=Path, default=Path("case_matrix.csv"))
    parser.add_argument("--base-case", required=True, help="path to the base .cas.h5 for Fluent")
    parser.add_argument("--results", required=True, help="folder Fluent writes results into")
    parser.add_argument("--out", type=Path, default=Path("journals"))
    parser.add_argument("--cases", nargs="+", help="only these case_ids, e.g. for a pilot run")
    args = parser.parse_args()

    case_matrix = pd.read_csv(args.case_matrix)
    if args.cases:
        unknown = set(args.cases) - set(case_matrix.case_id)
        if unknown:
            parser.error(f"not in the case matrix: {', '.join(sorted(unknown))}")
        case_matrix = case_matrix[case_matrix.case_id.isin(args.cases)]

    write_journals(case_matrix, args.base_case, args.results, args.out)
    print(f"wrote {len(case_matrix)} journals and run_all.jou to {args.out}")


if __name__ == "__main__":
    main()
