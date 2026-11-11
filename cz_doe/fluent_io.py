"""Readers for the files Fluent writes per case: ASCII export, transcript and heat-flux report."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["r", "z", "u_r", "u_z", "u_swirl", "p", "T"]

# 2D axisymmetric with the axis along x.
FLUENT_TO_DATASET = {
    "y-coordinate": "r",
    "x-coordinate": "z",
    "radial-velocity": "u_r",
    "axial-velocity": "u_z",
    "swirl-velocity": "u_swirl",
    "pressure": "p",
    "temperature": "T",
}

# 1e-10 m is far below the smallest cell, so equal keys mean the same node.
COORD_DECIMALS = 10

_RESIDUAL_LINE = re.compile(r"^\s*(\d+)((?:\s+\d\.\d+e[-+]\d+){5})\s", re.MULTILINE)


@dataclass(frozen=True)
class RunLog:
    converged: bool
    iterations: int | None
    final_residuals: tuple[float, ...] | None
    solver_error: bool


def read_export(path: Path) -> pd.DataFrame:
    """Read a Fluent ASCII node export into the dataset columns, by name not position."""
    raw = pd.read_csv(path, skipinitialspace=True)
    raw.columns = raw.columns.str.strip()
    raw = raw.loc[:, ~raw.columns.duplicated()]
    missing = set(FLUENT_TO_DATASET) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return raw[list(FLUENT_TO_DATASET)].rename(columns=FLUENT_TO_DATASET)[FIELDS]


def align_to_reference(field: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Reorder a field onto the reference node order, matched by coordinates.

    Fluent numbers nodes differently between GUI and console exports, so row order
    cannot be trusted.
    """
    if len(field) != len(reference):
        raise ValueError(f"{len(field)} nodes, reference has {len(reference)}")
    keyed = field.set_index(_node_keys(field))
    if not keyed.index.is_unique:
        raise ValueError("duplicate node coordinates in export")
    ref_keys = _node_keys(reference)
    missing = ref_keys.difference(keyed.index)
    if len(missing):
        raise ValueError(f"{len(missing)} reference nodes not found in export")
    return keyed.loc[ref_keys].reset_index(drop=True)


def _node_keys(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [np.round(df.r.to_numpy(), COORD_DECIMALS), np.round(df.z.to_numpy(), COORD_DECIMALS)]
    )


def read_transcript(path: Path) -> RunLog:
    text = path.read_text(errors="replace")
    rows = _RESIDUAL_LINE.findall(text)
    lowered = text.lower()
    return RunLog(
        converged="solution is converged" in lowered,
        iterations=int(rows[-1][0]) if rows else None,
        final_residuals=tuple(float(v) for v in rows[-1][1].split()) if rows else None,
        solver_error=any(s in lowered for s in ("divergence", "floating point", "error:")),
    )


def residual_history(path: Path) -> np.ndarray:
    """All residual rows of a transcript, one column per equation, in Fluent's order."""
    rows = _RESIDUAL_LINE.findall(path.read_text(errors="replace"))
    return np.array([[float(v) for v in row[1].split()] for row in rows])


def read_heat_flux(path: Path) -> tuple[dict[str, float], float]:
    """Return per-boundary heat flow (W) and the net value from a flux report."""
    flows, net = {}, None
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if parts[0] == "Net":
            net = value
        else:
            flows[parts[0]] = value
    if net is None or not flows:
        raise ValueError(f"{path.name}: not a heat-flux report")
    return flows, net


def energy_imbalance(path: Path) -> float:
    """Net heat flow as a fraction of the total heat entering the melt."""
    flows, net = read_heat_flux(path)
    return abs(net) / sum(v for v in flows.values() if v > 0)
