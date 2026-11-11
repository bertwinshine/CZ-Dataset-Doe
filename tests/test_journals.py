import pandas as pd

from cz_doe.journals import HEADER, case_journal, write_journals

BASE = "D:/data/CZ_base_1750K.cas.h5"
RESULTS = "D:/data/results"

# The command sequence that reproduced the manual 1750 K baseline on 6 Nov 2026.
VALIDATED_ANCHOR = """\
/file/start-transcript "D:/data/results/anchor.trn"
/file/read-case "D:/data/CZ_base_1750K.cas.h5"
/define/named-expressions/edit "omega_cry" definition "0.837758041 [s^-1]" q
/define/named-expressions/edit "omega_cru" definition "-0.314159265 [s^-1]" q
/define/named-expressions/edit "T_hot" definition "1750 [K]" q
/solve/monitors/residual/convergence-criteria 1e-06 1e-06 1e-06 1e-06 1e-06
/solve/initialize/hyb-initialization
/solve/iterate 30000
/file/write-case-data "D:/data/results/anchor.cas.h5"
/report/fluxes/heat-transfer yes yes "D:/data/results/anchor_heatflux.txt"
/file/export/ascii "D:/data/results/anchor_raw.txt" () yes pressure temperature \
axial-velocity radial-velocity swirl-velocity () no
/file/stop-transcript
"""

ANCHOR = pd.Series(
    {"case_id": "anchor", "crystal_rpm": 8.0, "crucible_rpm": -3.0, "t_hot": 1750.0}
)


def test_anchor_journal_matches_validated_commands():
    assert case_journal(ANCHOR, BASE, RESULTS) == VALIDATED_ANCHOR


def test_windows_paths_are_converted():
    text = case_journal(ANCHOR, "D:\\data\\CZ_base_1750K.cas.h5", "D:\\data\\results")
    assert text == VALIDATED_ANCHOR


def test_run_all_contains_every_case(tmp_path):
    case_matrix = pd.DataFrame([ANCHOR, ANCHOR.replace("anchor", "second")])
    write_journals(case_matrix, BASE, RESULTS, tmp_path)
    run_all = (tmp_path / "run_all.jou").read_text()
    assert run_all.startswith(HEADER)
    assert run_all.count("/file/read-case") == 2
    assert (tmp_path / "second.jou").exists()
