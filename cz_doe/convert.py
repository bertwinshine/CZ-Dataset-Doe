"""Convert raw Fluent exports into dataset CSVs on the reference node order."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cz_doe.fluent_io import align_to_reference, read_export

RAW_SUFFIX = "_raw.txt"


def convert(raw: Path, reference: pd.DataFrame, out_dir: Path) -> Path:
    field = align_to_reference(read_export(raw), reference)
    if not np.isfinite(field.to_numpy()).all():
        raise ValueError("NaN or inf in field")
    target = out_dir / f"{raw.name.removesuffix(RAW_SUFFIX)}.csv"
    field.to_csv(target, index=False, float_format="%.10g")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="CSV defining node order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("results", type=Path, help=f"folder with the *{RAW_SUFFIX} exports")
    args = parser.parse_args()

    raw_files = sorted(args.results.glob(f"*{RAW_SUFFIX}"))
    if not raw_files:
        sys.exit(f"no *{RAW_SUFFIX} files in {args.results}")
    reference = pd.read_csv(args.reference)
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for raw in raw_files:
        try:
            print(f"OK    {raw.name} -> {convert(raw, reference, args.out).name}")
        except ValueError as err:
            failed += 1
            print(f"FAIL  {raw.name}: {err}")
    print(f"{len(raw_files) - failed} of {len(raw_files)} converted")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
