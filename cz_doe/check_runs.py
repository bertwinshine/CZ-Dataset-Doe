"""Report whether each Fluent run converged and closed its energy balance."""

import argparse
import sys
from pathlib import Path

from cz_doe.fluent_io import energy_imbalance, read_transcript

# A case converged to 1e-6 closes to ~1e-4 (baseline: 0.011 %).
MAX_IMBALANCE = 1e-3


def check_case(transcript: Path) -> tuple[bool, str]:
    log = read_transcript(transcript)
    flux_file = transcript.with_name(f"{transcript.stem}_heatflux.txt")
    imbalance = energy_imbalance(flux_file) if flux_file.exists() else None

    problems = []
    if not log.converged:
        problems.append("not converged")
    if log.solver_error:
        problems.append("solver error in log")
    if imbalance is None:
        problems.append("no heat-flux report")
    elif imbalance > MAX_IMBALANCE:
        problems.append(f"energy imbalance {imbalance:.3%}")

    imbalance_text = "-" if imbalance is None else f"{imbalance:.4%}"
    detail = f"iter {log.iterations}  imbalance {imbalance_text}"
    if problems:
        detail += "  (" + ", ".join(problems) + ")"
    return not problems, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="folder with the .trn transcripts")
    args = parser.parse_args()

    transcripts = sorted(args.results.glob("*.trn"))
    if not transcripts:
        sys.exit(f"no .trn files in {args.results}")

    failed = 0
    for trn in transcripts:
        ok, detail = check_case(trn)
        failed += not ok
        print(f"{'OK   ' if ok else 'CHECK'} {trn.stem:<20} {detail}")
    print(f"{len(transcripts) - failed} of {len(transcripts)} clean")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
