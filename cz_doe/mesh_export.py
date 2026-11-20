"""Convert a Fluent 2D mesh (.msh) into an HDF5 mesh file on the dataset node order.

The file holds what graph and FEM-style models need and the CSVs lack: cells, edges and
boundary node sets, with nodes in exactly the row order of the case CSVs on that mesh.
"""

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Far below the smallest cell edge (~0.1 mm on the fine mesh) and far above CSV round-off.
MATCH_TOL = 1e-8

_HEADER = re.compile(rb"\((\d+) \(([0-9a-fA-F ]+)\)")
_ZONE_NAME = re.compile(rb"\(45 \((\d+) (\S+) ([^)\s]+)\)")
_LINEAR_FACE = 2
_FLOAT_BYTES = {"10": 0, "2010": 4, "3010": 8}  # 0 means ASCII


@dataclass(frozen=True)
class FluentMesh:
    """Mesh as Fluent stores it: 0-based node ids, faces with their two cells (-1 at walls)."""

    points: np.ndarray  # (N, 2) Fluent x, y
    faces: np.ndarray  # (F, 2) node ids
    face_cells: np.ndarray  # (F, 2) right and left cell ids
    face_zone: np.ndarray  # (F,) zone id
    zone_names: dict[int, tuple[str, str]]  # id -> (type, name)


@dataclass(frozen=True)
class Mesh:
    """Mesh on the dataset node order, coordinates as (r, z)."""

    nodes: np.ndarray
    cells: np.ndarray
    edges: np.ndarray
    boundaries: dict[str, np.ndarray]


def read_fluent_msh(path: Path) -> FluentMesh:
    """Read a 2D Fluent mesh with linear faces, binary or ASCII."""
    data = path.read_bytes()
    points, faces, zones = {}, [], []
    for match in _HEADER.finditer(data):
        section, fields = match.group(1).decode(), [int(v, 16) for v in match.group(2).split()]
        zone, first, last = fields[:3]
        if zone == 0 or section not in {*_FLOAT_BYTES, "13", "2013"}:
            continue
        count = last - first + 1
        body = data.index(b"(", match.end()) + 1
        if section in _FLOAT_BYTES:
            if fields[4] != 2:
                raise ValueError(f"{path.name}: only 2D meshes are supported")
            points[first] = _read_block(data, body, count * 2, _FLOAT_BYTES[section], float)
        else:
            if fields[4] != _LINEAR_FACE:
                raise ValueError(f"{path.name}: zone {zone} is not made of linear faces")
            ascii_ints = 0 if section == "13" else 4
            faces.append(_read_block(data, body, count * 4, ascii_ints, int).reshape(-1, 4))
            zones.append(np.full(count, zone))
    if not points or not faces:
        raise ValueError(f"{path.name}: no node or face sections found")

    face_data = np.concatenate(faces)
    # Fluent writes zone ids in hex in section headers but in decimal in the (45 ...) names.
    zone_names = {
        int(m.group(1)): (m.group(2).decode(), m.group(3).decode())
        for m in _ZONE_NAME.finditer(data)
    }
    return FluentMesh(
        points=np.concatenate([points[k].reshape(-1, 2) for k in sorted(points)]),
        faces=face_data[:, :2] - 1,
        face_cells=face_data[:, 2:] - 1,
        face_zone=np.concatenate(zones),
        zone_names=zone_names,
    )


def _read_block(data: bytes, start: int, n: int, item_bytes: int, kind: type) -> np.ndarray:
    if item_bytes:
        dtype = {(float, 4): "<f4", (float, 8): "<f8", (int, 4): "<i4"}[(kind, item_bytes)]
        return np.frombuffer(data, dtype=dtype, count=n, offset=start).astype(kind)
    tokens = data[start : data.index(b")", start)].split()
    if len(tokens) != n:
        raise ValueError(f"expected {n} values, found {len(tokens)}")
    return np.array([float(t) if kind is float else int(t, 16) for t in tokens])


def build_cells(fluent: FluentMesh) -> np.ndarray:
    """Recover each cell's nodes, counter-clockwise in the (r, z) plane, from its faces."""
    n_cells = fluent.face_cells.max() + 1
    cell_edges: list[list[tuple[int, int]]] = [[] for _ in range(n_cells)]
    for (a, b), cells in zip(fluent.faces, fluent.face_cells, strict=True):
        for c in cells:
            if c >= 0:
                cell_edges[c].append((a, b))

    loops = [_edge_loop(edges, c) for c, edges in enumerate(cell_edges)]
    if len({len(loop) for loop in loops}) != 1:
        raise ValueError("mixed cell types are not supported")
    cells = np.array(loops)
    rz = fluent.points[:, ::-1]
    flip = signed_areas(rz, cells) < 0
    cells[flip] = cells[flip, ::-1]
    return cells


def _edge_loop(edges: list[tuple[int, int]], cell: int) -> list[int]:
    neighbours: dict[int, list[int]] = {}
    for a, b in edges:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    if len(edges) < 3 or any(len(v) != 2 for v in neighbours.values()):
        raise ValueError(f"cell {cell}: faces do not form a closed loop")
    loop = [edges[0][0], edges[0][1]]
    while len(loop) < len(edges):
        prev, here = loop[-2], loop[-1]
        a, b = neighbours[here]
        loop.append(b if a == prev else a)
    if loop[0] not in neighbours[loop[-1]]:
        raise ValueError(f"cell {cell}: faces do not form a closed loop")
    return loop


def signed_areas(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    x, y = points[cells, 0], points[cells, 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


def to_dataset_order(fluent: FluentMesh, cells: np.ndarray, reference: pd.DataFrame) -> Mesh:
    """Renumber nodes to the row order of a case CSV on this mesh, matched by (r, z)."""
    rz = fluent.points[:, ::-1]  # Fluent axisymmetric: x is the axis (z), y the radius (r)
    if len(reference) != len(rz):
        raise ValueError(f"mesh has {len(rz)} nodes, reference has {len(reference)}")
    dist, fluent_id = cKDTree(rz).query(reference[["r", "z"]].to_numpy())
    if dist.max() > MATCH_TOL or len(np.unique(fluent_id)) != len(rz):
        raise ValueError(f"reference nodes do not match the mesh (worst gap {dist.max():.2e} m)")
    new_id = np.empty(len(rz), dtype=np.int64)
    new_id[fluent_id] = np.arange(len(rz))

    boundaries = {}
    for zone, (zone_type, name) in fluent.zone_names.items():
        in_zone = fluent.face_zone == zone
        if in_zone.any() and zone_type != "interior":
            boundaries[name] = np.unique(new_id[fluent.faces[in_zone]])
    return Mesh(
        nodes=rz[fluent_id],
        cells=new_id[cells],
        edges=np.sort(new_id[fluent.faces], axis=1),
        boundaries=boundaries,
    )


def check(mesh: Mesh) -> None:
    """Fail on connectivity that would silently corrupt a model trained on it."""
    used = np.zeros(len(mesh.nodes), dtype=bool)
    used[mesh.cells] = True
    if not used.all():
        raise ValueError(f"{(~used).sum()} nodes belong to no cell")
    areas = signed_areas(mesh.nodes, mesh.cells)
    if areas.min() <= 0:
        raise ValueError("cell with zero or negative area")
    # The domain area from the boundary alone must equal the summed cell areas.
    boundary_area = _area_from_cell_boundary(mesh.cells, mesh.nodes)
    if not np.isclose(boundary_area, areas.sum(), rtol=1e-9):
        raise ValueError(
            f"cells cover {areas.sum():.6e} m², boundary encloses {boundary_area:.6e} m²"
        )


def _area_from_cell_boundary(cells: np.ndarray, nodes: np.ndarray) -> float:
    directed = np.stack([cells, np.roll(cells, -1, axis=1)], axis=-1).reshape(-1, 2)
    keys = {tuple(e) for e in directed}
    outer = np.array([e for e in directed if (e[1], e[0]) not in keys])
    a, b = nodes[outer[:, 0]], nodes[outer[:, 1]]
    return 0.5 * float(np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]))


def mesh_hash(mesh: Mesh) -> str:
    """Identifies the mesh, so a loader can check a case file belongs to it."""
    digest = hashlib.sha256(mesh.nodes.astype("<f8").tobytes())
    digest.update(mesh.cells.astype("<i8").tobytes())
    return digest.hexdigest()


def write_hdf5(mesh: Mesh, path: Path, name: str) -> None:
    with h5py.File(path, "w") as f:
        f.attrs.update(
            mesh_name=name,
            n_nodes=len(mesh.nodes),
            n_cells=len(mesh.cells),
            mesh_sha256=mesh_hash(mesh),
            description=(
                "2D axisymmetric melt mesh. nodes are (r, z) in m, in the row order of the "
                "case CSVs on this mesh. cells list node indices counter-clockwise in (r, z)."
            ),
        )
        f.create_dataset("nodes", data=mesh.nodes)
        f.create_dataset("cells", data=mesh.cells)
        f.create_dataset("edges", data=mesh.edges)
        group = f.create_group("boundary")
        for zone, ids in sorted(mesh.boundaries.items()):
            group.create_dataset(zone, data=ids)


def export(msh: Path, reference_csv: Path, out: Path, name: str) -> Mesh:
    fluent = read_fluent_msh(msh)
    mesh = to_dataset_order(fluent, build_cells(fluent), pd.read_csv(reference_csv))
    check(mesh)
    write_hdf5(mesh, out, name)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("msh", type=Path, help="Fluent mesh file (.msh)")
    parser.add_argument("--reference", type=Path, required=True, help="any case CSV on this mesh")
    parser.add_argument("--name", required=True, help="mesh name, e.g. base, coarse, fine")
    parser.add_argument("--out", type=Path, required=True, help="output .h5 file")
    args = parser.parse_args()
    mesh = export(args.msh, args.reference, args.out, args.name)
    zones = ", ".join(f"{k} ({len(v)})" for k, v in sorted(mesh.boundaries.items()))
    print(f"{args.out.name}: {len(mesh.nodes)} nodes, {len(mesh.cells)} cells; {zones}")


if __name__ == "__main__":
    main()
