# CZ-Dataset-Doe

CFD reference data for the silicon melt in a Czochralski (CZ) crystal puller, run in ANSYS Fluent
over 265 combinations of crystal rotation, crucible rotation and hot-wall temperature.

The data is made for people building fast prediction models (surrogates, physics-informed neural
networks, graph networks) who need trustworthy CFD results to train and test against. Every case
uses the same mesh, the same model and the same convergence target, and every case that did not
behave is listed with the reason.

This is the second dataset of the project. The first one (three one-parameter sweeps, 33 cases,
converged to Fluent's default 1e-3) is published separately and is not changed by this repository.

## The physical problem

In the Czochralski process a single silicon crystal is pulled slowly out of a crucible of molten
silicon. The crystal and the crucible rotate in opposite directions, and the crucible wall is hotter
than the crystal. Rotation and buoyancy together drive the flow in the melt, and that flow controls
how heat reaches the growing crystal.

I model a vertical half-slice of the melt (from the centre axis to the crucible wall) and let
Fluent treat it as rotationally symmetric. That keeps every case small enough to run on a student
licence while still capturing swirl and buoyancy.

| Item | Value |
|---|---|
| Model | 2D axisymmetric with swirl, steady, laminar |
| Geometry | crucible radius 0.30 m, crystal radius 0.18 m, melt height 0.15 m |
| Mesh | 8181 nodes (8000 cells), identical for every case |
| Crystal face | rotates with the crystal, 1685 K |
| Crucible wall | rotates with the crucible, at the hot-wall temperature T_hot |
| Crucible bottom | rotates with the crucible, temperature linear from 1700 K at the axis to T_hot at the wall |
| Melt surface outside the crystal | no-slip lid with a prescribed swirl that ramps linearly from the crystal edge to the wall, no up/down or in/out flow; temperature linear from 1685 K to T_hot |
| Buoyancy | Boussinesq, β = 1.5e-4 1/K, reference temperature 1717 K, g = 9.81 m/s² downwards |
| Pressure | reported without the hydrostatic part |
| Convergence | all five residuals (continuity, two velocities, swirl, energy) below 1e-6 |

The setup follows Huang et al., *AIP Advances* 15(10), 105202 (2025),
doi:[10.1063/5.0271778](https://doi.org/10.1063/5.0271778).

## What was run

Instead of changing one parameter at a time, all three are changed together, spread evenly over
the whole operating box with a Sobol design (a way of placing points so no region is left empty).
The corners and the centre of the box are added on top.

| Parameter | Range |
|---|---|
| crystal rotation | 4 to 20 rpm |
| crucible rotation | −10 to −1 rpm (negative = opposite to the crystal) |
| hot-wall temperature | 1730 to 1785 K |

| Group | Cases | Why |
|---|---|---|
| `sobol` | 256 | evenly spread over the box (seed 42) |
| `corner` | 8 | the extremes of the box |
| `centre` | 1 | the middle of the box |

`cz_manifest.csv` holds the exact values of every case and is the single reference for them.

## Results so far

Of the 265 runs, **254 reached a steady solution and are kept**. **11 are excluded.**

![DOE parameter space](figures/doe_parameter_space.png)

The 11 excluded cases ran the full 30,000 iterations without converging. Their continuity residual
stopped falling at about 1e-2 and kept swinging up and down, and their heat balance (heat in vs heat
out of the melt) was off by 0.6 to 7 %. For comparison, the kept cases converged in about 3,500
iterations on average and their heat balance closes within 0.02 %. In other words, these cases are
not slow, the flow there does not settle to a steady state at all. A steady model cannot describe
them, so they are left out of the dataset rather than included with a misleading field.

They are not scattered randomly. All 11 sit in one region: crystal rotation of roughly 5 to 8 rpm,
weak crucible rotation (−1 to −5 rpm) and a hot wall above about 1753 K. This is the same crystal
rotation range where the first dataset showed the main flow loop reversing direction, which is a
likely explanation, but I have not proven it with transient runs.

Two kept cases, `sobol_068` and `sobol_084`, needed 13,000 and 24,000 iterations. They pass every
check but lie at the edge of that region, so they are marked `slow` in `run_status.csv`.

Files with the details:

- `run_status.csv`: every case with its parameters, kept or excluded, iterations, final residuals,
  heat-balance error and the plateau ratio (see below).
- `excluded_cases.csv`: only the 11 excluded cases.

The **plateau ratio** compares the typical continuity residual of the last 5,000 iterations with the
5,000 before. A case that is still converging gives a small number (the two slow cases give about
0.05). A case that has stopped improving gives about 1. All 11 excluded cases are between 0.98 and
1.02.

## What is in this repository

```
cz_manifest.csv        the 265 cases and their parameter values
run_status.csv         outcome of every run
excluded_cases.csv     the 11 runs that did not reach a steady solution
figures/               plots
cz_doe/                the Python code (described below)
tests/                 automatic checks for the code
```

What is deliberately **not** in this repository:

- The Fluent case and data files (`.cas.h5`, `.dat.h5`). They are large and need Fluent to open.
- The Fluent transcripts (`.trn`). They contain details of my computer and folder paths, and every
  number that matters from them is already in `run_status.csv`.
- The generated journal files. They contain the absolute folder paths of my computer and are
  rebuilt with one command.

The converted CFD fields (one table per case with r, z, u_r, u_z, u_swirl, p and T at every node)
will be published on Zenodo with their own DOI once all cases are certified.

## How the work flows

Each step is one command. The code only automates what I would otherwise click through in Fluent
265 times, and it checks the results instead of trusting them.

```
python -m cz_doe.manifest                       # 1. list the cases (already done: cz_manifest.csv)
python -m cz_doe.journals --base-case <case.cas.h5> --results <results folder>
                                                # 2. write one Fluent journal per case
# in Fluent (2D, double precision): File > Read > Journal > journals/run_all.jou
                                                # 3. Fluent runs every case and exports the results
python -m cz_doe.check_runs <results folder>    # 4. quick check: converged? heat balance closed?
python -m cz_doe.status <results folder>        # 5. sort cases into kept / excluded, make the plot
python -m cz_doe.convert --reference <reference.csv> --out data <results folder>
                                                # 6. turn Fluent exports into clean tables
python -m cz_doe.certify data                   # 7. check every boundary condition in every table
```

What each part of `cz_doe/` does:

| File | Job |
|---|---|
| `manifest.py` | builds the case list (Sobol points, corners, centre) |
| `journals.py` | writes the Fluent journal for each case; only the three parameters change |
| `bc.py` | the boundary conditions written as formulas, so they can be checked |
| `fluent_io.py` | reads Fluent exports, transcripts and heat-flux reports by column name, never by position |
| `check_runs.py` | flags runs that did not converge or do not close the heat balance |
| `status.py` | writes `run_status.csv`, `excluded_cases.csv` and the parameter-space plot |
| `convert.py` | matches nodes by their coordinates, because Fluent numbers nodes differently between exports |
| `certify.py` | recomputes every boundary condition from the case parameters and compares with the exported field |

The base case must define three named expressions, `omega_cry` and `omega_cru` (in 1/s) and
`T_hot`, with every boundary condition written in terms of them. The journals only change these
three values.

## Limits

These are controlled reference solutions for developing and testing prediction methods, not a model
of a real industrial puller. The flow is assumed laminar, axisymmetric and steady, with a flat melt
surface, no Marangoni (surface tension) flow, no radiation and no magnetic field. At the higher
rotation rates the rotational Reynolds number is at or above the range where real CZ melts become
turbulent, so the results there describe the laminar model, not the real furnace.

## Still to do

- Convert and certify the 254 kept cases, then publish them on Zenodo.
- Coarser and finer mesh runs on a subset of cases, for a mesh-convergence check and for
  multi-fidelity training.
- One HDF5 file per case including the mesh connectivity, for graph-based models.
- Test sets: the one-parameter sweeps re-run to 1e-6, and cases outside the training box.

## Running the code yourself

You need Python 3.10 or newer.

```
pip install -e .[dev]
pytest                       # runs the automatic checks
ruff format . && ruff check .  # keeps the code tidy
```

Steps 2 and 3 above need ANSYS Fluent. Everything else runs without it.

## How to cite

See `CITATION.cff`, or use the "Cite this repository" button on GitHub. A DOI for the data will be
added here once it is on Zenodo.

## Licence

- Code: MIT, see [LICENSE](LICENSE).
- Data (case list, run summaries, figures and the converted fields): CC BY 4.0, see
  [LICENSE-DATA](LICENSE-DATA). Use it freely, including commercially, as long as you credit it.

## Use of AI

The Fluent model setup, the validation and the design decisions are mine. The Python code was
written with AI assistance (Claude). I reviewed it and tested it against Fluent results before use.
