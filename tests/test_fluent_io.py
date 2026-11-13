import pytest

from cz_doe.fluent_io import (
    align_to_reference,
    energy_imbalance,
    read_export,
    read_transcript,
    residual_history,
)

HEAT_FLUX = """\
                          "Flux Report"

        Total Heat Transfer Rate                  [W]
-------------------------------- --------------------
                    crsytal_face           -5427.7028
                 crucible_bottom            7782.8263
                   crucible_wall            7607.0323
                    free_surface            -9960.409
                ---------------- --------------------
                             Net            1.7467661
"""

TRANSCRIPT = """\
  iter  continuity  x-velocity  y-velocity       swirl      energy     time/iter
  3805  1.0016e-06  6.9455e-10  1.5163e-10  9.3455e-12  1.9817e-10  0:13:24 26195
  3806  9.9909e-07  6.9280e-10  1.5132e-10  9.3278e-12  1.9766e-10  0:13:52 26194
! 3806 solution is converged
"""


def write_fluent_export(df, path):
    """Mimic the console export: reversed variable order, repeated coordinate columns."""
    out = df.sample(frac=1, random_state=0).reset_index(drop=True)
    cols = {
        "nodenumber": range(1, len(out) + 1),
        "x-coordinate": out.z,
        "y-coordinate": out.r,
        "x-coordinate.1": out.z,
        "y-coordinate.1": out.r,
        "swirl-velocity": out.u_swirl,
        "radial-velocity": out.u_r,
        "axial-velocity": out.u_z,
        "temperature": out["T"],
        "pressure": out.p,
    }
    lines = [", ".join(name.removesuffix(".1") for name in cols)]
    rows = zip(*cols.values(), strict=True)
    lines += [
        ", ".join(f"{v:.9E}" if isinstance(v, float) else str(v) for v in row) for row in rows
    ]
    path.write_text("\n".join(lines) + "\n")


def test_export_is_realigned_to_reference(field, tmp_path):
    path = tmp_path / "case_raw.txt"
    write_fluent_export(field, path)
    aligned = align_to_reference(read_export(path), field)
    assert (aligned.columns == field.columns).all()
    assert ((aligned - field).abs().to_numpy().max()) < 1e-9 * field.abs().to_numpy().max()


def test_missing_node_is_rejected(field):
    shifted = field.copy()
    shifted.loc[0, "r"] += 1e-6
    with pytest.raises(ValueError, match="not found"):
        align_to_reference(shifted, field)


def test_transcript(tmp_path):
    path = tmp_path / "case.trn"
    path.write_text(TRANSCRIPT)
    log = read_transcript(path)
    assert log.converged and not log.solver_error
    assert log.iterations == 3806
    assert log.final_residuals[0] == pytest.approx(9.9909e-07)


def test_energy_imbalance(tmp_path):
    path = tmp_path / "case_heatflux.txt"
    path.write_text(HEAT_FLUX)
    assert energy_imbalance(path) == pytest.approx(1.7467661 / 15389.8586, rel=1e-6)


def test_residual_history_reads_every_row(tmp_path):
    path = tmp_path / "case.trn"
    path.write_text(TRANSCRIPT)
    history = residual_history(path)
    assert history.shape == (2, 5)
    assert history[-1, 0] == 9.9909e-07


def test_converged_line_from_an_earlier_run_does_not_count(tmp_path):
    # A run continued from a converged case repeats that case's "converged" line.
    path = tmp_path / "continued.trn"
    path.write_text(
        TRANSCRIPT
        + "  iter  continuity  x-velocity  y-velocity       swirl      energy     time/iter\n"
        + "  3807  5.1000e-03  4.8000e-06  2.9000e-06  2.1000e-07  5.6000e-07  0:13:52 26194\n"
    )
    assert not read_transcript(path).converged
