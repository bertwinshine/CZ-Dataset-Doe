import h5py
import numpy as np
import pandas as pd
import pytest

from cz_doe.mesh_export import export, read_fluent_msh, signed_areas

NR, NZ = 4, 3  # nodes along r and z
R_MAX, Z_MAX = 0.3, 0.15
ZONES = {
    3: ("interior", "interior-fluid"),
    5: ("wall", "crucible_bottom"),
    7: ("wall", "crucible_wall"),
    8: ("axis", "axis"),
    9: ("wall", "free_surface"),
}


def grid_mesh():
    """Structured quad mesh with deliberately scrambled Fluent node numbering."""
    r, z = np.meshgrid(np.linspace(0, R_MAX, NR), np.linspace(0, Z_MAX, NZ))
    rz = np.column_stack([r.ravel(), z.ravel()])
    perm = np.random.default_rng(0).permutation(len(rz))  # perm[grid id] = fluent id
    points = np.empty_like(rz)
    points[perm] = rz[:, ::-1]  # Fluent x is the axis (z)

    def node(i, j):
        return perm[j * NR + i]

    def cell(i, j):
        return j * (NR - 1) + i if 0 <= i < NR - 1 and 0 <= j < NZ - 1 else -1

    faces = {zone: [] for zone in ZONES}
    for j in range(NZ):
        for i in range(NR - 1):  # faces along r, between cells (i, j-1) and (i, j)
            zone = {0: 5, NZ - 1: 9}.get(j, 3)
            below, above = cell(i, j - 1), cell(i, j)
            right, left = (above, below) if above >= 0 else (below, -1)
            faces[zone].append((node(i, j), node(i + 1, j), right, left))
    for i in range(NR):
        for j in range(NZ - 1):  # faces along z, between cells (i-1, j) and (i, j)
            zone = {0: 8, NR - 1: 7}.get(i, 3)
            before, after = cell(i - 1, j), cell(i, j)
            right, left = (after, before) if after >= 0 else (before, -1)
            faces[zone].append((node(i, j), node(i, j + 1), right, left))
    reference = pd.DataFrame(rz, columns=["r", "z"]).sample(frac=1, random_state=1)
    return points, faces, reference.reset_index(drop=True)


def write_msh(path, points, faces, binary=True):
    """Write the sections Fluent writes, in its binary or ASCII form."""
    out = bytearray(b'(0 "test mesh")\n(2 2)\n')
    first = 1
    out += f"({3010 if binary else 10} (1 1 {len(points):x} 1 2)\n(".encode()
    out += points.astype("<f8").tobytes() if binary else _ascii(points.ravel(), "{:.12e}")
    out += b"))\n"
    for zone, rows in faces.items():
        if not rows:
            continue
        ids = np.array(rows) + 1  # Fluent is 1-based, 0 means no cell
        last = first + len(ids) - 1
        out += f"({2013 if binary else 13} ({zone:x} {first:x} {last:x} 3 2)\n(".encode()
        out += ids.astype("<i4").tobytes() if binary else _ascii(ids.ravel(), "{:x}")
        out += b"))\n"
        first = last + 1
    for zone, (kind, name) in ZONES.items():
        out += f"(45 ({zone} {kind} {name})())\n".encode()
    path.write_bytes(bytes(out))


def _ascii(values, fmt):
    return ("\n" + " ".join(fmt.format(v) for v in values) + "\n").encode()


@pytest.fixture
def grid(tmp_path):
    points, faces, reference = grid_mesh()
    reference.to_csv(tmp_path / "case.csv", index=False)
    return tmp_path, points, faces, reference


@pytest.mark.parametrize("binary", [True, False])
def test_export_matches_csv_order(grid, binary):
    tmp, points, faces, reference = grid
    write_msh(tmp / "m.msh", points, faces, binary)
    mesh = export(tmp / "m.msh", tmp / "case.csv", tmp / "m.h5", "test")

    np.testing.assert_allclose(mesh.nodes, reference[["r", "z"]].to_numpy())
    areas = signed_areas(mesh.nodes, mesh.cells)
    assert mesh.cells.shape == ((NR - 1) * (NZ - 1), 4)
    assert np.allclose(areas, R_MAX * Z_MAX / len(areas))
    assert np.allclose(mesh.nodes[mesh.boundaries["axis"], 0], 0)
    assert np.allclose(mesh.nodes[mesh.boundaries["crucible_bottom"], 1], 0)
    assert "interior-fluid" not in mesh.boundaries
    with h5py.File(tmp / "m.h5") as f:
        assert f.attrs["n_nodes"] == NR * NZ
        np.testing.assert_array_equal(f["cells"][:], mesh.cells)
        np.testing.assert_array_equal(f["boundary/axis"][:], mesh.boundaries["axis"])


def test_reads_fluent_axis_convention(grid):
    tmp, points, faces, _ = grid
    write_msh(tmp / "m.msh", points, faces)
    np.testing.assert_allclose(read_fluent_msh(tmp / "m.msh").points, points)


def test_rejects_reference_from_another_mesh(grid):
    tmp, points, faces, reference = grid
    write_msh(tmp / "m.msh", points, faces)
    reference.assign(r=reference.r * 1.01).to_csv(tmp / "other.csv", index=False)
    with pytest.raises(ValueError, match="do not match"):
        export(tmp / "m.msh", tmp / "other.csv", tmp / "m.h5", "test")


def test_rejects_cell_with_missing_face(grid):
    tmp, points, faces, _ = grid
    faces[8] = faces[8][1:]
    write_msh(tmp / "m.msh", points, faces)
    with pytest.raises(ValueError, match="closed loop"):
        export(tmp / "m.msh", tmp / "case.csv", tmp / "m.h5", "test")
