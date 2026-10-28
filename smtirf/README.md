# Parts 2 and 3: single-molecule TIRF

`mangotrace/` turns each raw movie into a file of idealised traces (part 2). `kinetics/` turns the 36 Chapter 4 files into the tables behind sections 4.3.3 to 4.3.8 (part 3). Both write tables only; neither draws figures.

## Part 2: trace analysis (`mangotrace/`)

Localisation by net-gradient detection and Gaussian least-squares fitting, drift correction by adaptive intersection maximisation, clustering of localisations into binding sites, trace extraction, and change-point idealisation (PELT, minimum segment one frame, penalty 10 sigma squared). Every setting is in the `SETTINGS` block of `run_mangotrace.py`. Each movie gives one `<movie>_CPD.h5` (format version 4) and CSV tables of the localisations, drift and clusters.

```bash
python smtirf/mangotrace/run_mangotrace.py path/to/movie.ome.tif --outdir path/to/results
```

Installation, settings, outputs and the layout of the HDF5 file: `mangotrace/README.md`.

Part 3 reads the condition of each acquisition from its file name: the fluorogen (`SALAD` or `TO`), the concentration (`1nM`, `500pM`), the laser power when it is not 1.5 mW (`1mW`, `2mW`) and the replicate number before `.h5`, as in `CPD_SALAD_8nM_1mW_2.h5` or `TO_4nM_3.h5`. A file from part 2 is renamed to this pattern before part 3.

## Part 3: kinetic analysis (`kinetics/`)

Put the 36 files of `kinetics/FILES_CHAPTER4.txt` in one folder, directly or in subfolders, with no other `.h5` file in it. Then, from the repository folder:

```bash
python smtirf/kinetics/run_kinetics.py --data path/to/that/folder
python smtirf/kinetics/run_kinetics.py --data path/to/that/folder --skip-validation
python smtirf/kinetics/run_kinetics.py --only filters kd
```

The runner checks the folder against `FILES_CHAPTER4.txt` and stops if a file is missing or if another `.h5` file is present, because the analysis reads every `.h5` file it finds. `--skip-validation` leaves out the two simulation steps. The first run reads every HDF5 file and keeps a per-file extract in `kinetics/cache/`; later runs reuse it, and later steps read every extract there, so the runner also stops if `cache/` holds an extract of another file. Every table goes to `kinetics/tables/`.

| step | script | role |
|---|---|---|
| `analysis` | `analyse_smtirf.py` | reads the HDF5 files; quality checks, intensity and contrast, the detection test, fraction bound, signal persistence and dwell times; the settings every later step imports |
| `curves` | `occupancy_curves.py` | fraction bound against time, one mean curve per condition, from which the window is set and the fraction bound at the first frame is extrapolated |
| `validation` | `simulate_validate.py` | simulated traces with known rates, for the recovery tests; it re-implements the change-point idealisation and needs `ruptures` |
| `truncation` | `run_truncation_study.py` | detection efficiency at each minimum duration; needs `ruptures` |
| `window` | `window_select.py` | the analysis window; without `ruptures` the adopted rule is applied without scoring the candidates |
| `estimators` | `cross_estimators.py` | dark-interval estimators compared, window diagnostics, and the dissociation constant from mean dwell times |
| `cascade` | `plot_spec.py` | the dwell filter cascade, and the display limits of the dwell-time panels |
| `filters` | `dwell_filters.py` | the repeated-binding filter, the upper limits, the two-component dwell-time fits and the windowed dwell table |
| `interior` | `interior_kinetics.py` | dark intervals closed by two events, and per-condition dissociation constants |
| `compare` | `fit_compare.py` | one against two exponential components |
| `sensitivity` | `sensitivity_adopted.py` | the 64-combination grid of analysis choices |
| `kd` | `kd_per_condition.py` | fraction bound at the first frame, and the dissociation constant by the two-point route |

`mixture_truncated.py` holds the intensity-mixture fits that `analyse_smtirf.py` imports.

One input is stored rather than computed: `kinetics/stored/prebleach_initial_occupancy_langmuir.csv`, an isotherm fitted interactively on 14 September 2026 to the fraction bound at the first frame, before the analysis was restricted to the 36 files. `kd_per_condition.py` takes from it the ceiling of one route, R3, which the thesis does not report; no script here writes it.

### Where the thesis numbers are

All in `kinetics/tables/`. Rows at 1.5 mW unless the number concerns laser power.

| page | number | file, column |
|---|---|---|
| 101 | the analysis window opens at one quarter of the initial fraction bound; its length, and the fall of the fraction bound inside it | `window_definitions.csv`, `win_frames`, `residual_fold` (see the input set, below) |
| 101, 149 | detection 100 and 96 per cent at 3 frames, 97 and 73 per cent at 2 | `sim_e6_efficiency_pooled.csv` |
| 145, 147 | brightness ratio 1.08 to 1.24 across 1 to 8 nM, median 1.16; at 8 nM 1.15 (1.5 mW), 1.10 (2.0 mW), 0.89 (1.0 mW) | `brightness_ratio_by_condition.csv`, `ratio_median` |
| 145 | spots failing the acceptance test, 7 to 14 per cent (SALAD1) and 29 to 46 per cent (TO1-B); 89 to 97 per cent of them carry signal | `detection_floor_ladder.csv`, `frac_below_threshold`, and `dim_binders` over `below_threshold` |
| 145 | signal persistence about 150 s (SALAD1) and 80 s (TO1-B), the mean of the four Kaplan-Meier medians at 1 to 8 nM; TO1-B 39 to 186 s; SALAD1 one component at 1, 4 and 8 nM, two at 2 nM | `photostability_survival.csv`, `km_median_active_s`, `model_selected` |
| 148 | pooled median intensity 307 counts (53,138 SALAD1 spots) and 265 counts (34,718 TO1-B spots) | `brightness_per_molecule.csv` |
| 149, 151-152 | mean dwell times, for example 1.08 to 0.71 s (SALAD1) and 1.67 to 1.31 s (TO1-B) at 8 nM from 1.0 to 2.0 mW; SALAD1 1.54 s at 1 nM to 0.79 s at 8 nM | `dwell_fits_filtered.csv`, variant `repeat + upper censored`, `tau_amp_weighted_s` |
| 149, 151 | ranges under the 64 analysis choices; the effect of gap bridging, minimum duration and edge correction | `sensitivity_adopted.csv` |
| 153 | dark interval 18.9 to 9.1 s (SALAD1) and 17.9 to 15.5 s (TO1-B); dissociation constant from dwell times and dark intervals rising 7- to 11-fold | `kinetics_interior_only.csv`, `tau_off_interior_s`, `K_D_per_condition_nM` |
| 153 | mean dwell times 0.004 to 0.015 of the window; slow dark-interval component 0.11 to 0.14 (SALAD1) and 0.12 to 0.21 (TO1-B) of it | `kinetics_interior_only.csv`, `tau_on_s` and `off_tau_slow_s`, over the window length |
| 153 | seven estimators of the dark interval disagree threefold and 45-fold at 1 nM | `cross_estimator_tau.csv` |
| 153 | goodness of fit falls only from about 0.99 to 0.94 across a fifteen-fold disagreement between estimators | `cross_estimator_spread.csv`, `R2_median`, `spread_fold` |
| 153 | dissociation constant from mean dwell times rising 6- to 9-fold | `kd_mean_dwell_lee2024.csv`, `K_D_mean_dwell_nM` |
| 153 | affinity by the two-point route, 1.55 nM (SALAD1) and 0.39 nM (TO1-B) | `kd_two_point_summary.csv`, `K_D_median_nM` |

### The input set

The repository reads the 36 acquisitions of `FILES_CHAPTER4.txt`, the ones the thesis reports. The submitted thesis took its numbers from a run that also read the 0.5 and 16 nM acquisitions, and four quantities carried them into reported numbers. First, the condition slowest to relax sets the length of the analysis window; in that run it was SALAD1 at 0.5 nM. Second, the upper limit on bound dwells is set over every condition (see the settings below). Third, the detection efficiencies pool the simulations over concentrations. Fourth, the range of the fall inside the window was taken over every condition, and SALAD1 at 0.5 and 16 nM set both of its ends. With the 36 files the window runs 1,838 frames (183.8 s) instead of 1,809 (180.9 s), and these printed numbers move:

| page | submitted thesis | this repository |
|---|---|---|
| 101 | window 1,809 frames (180.9 s); fraction bound falls 3.3- to 8.9-fold inside it | 1,838 frames (183.8 s); 3.6- to 8.5-fold |
| 101 | detection at 3 frames at least about 96 per cent for both fluorogens | about 97 per cent |
| 149 | TO1-B mean dwell time at 8 nM, 1.67 s at 1.0 mW to 1.31 s at 2.0 mW | 1.67 to 1.30 s |
| 149 | lowest TO1-B mean dwell time at 1 nM over the 64 analysis choices, 1.70 s | 1.69 s |
| 149 | detection 100 (SALAD1) and 96 (TO1-B) per cent at 3 frames, 97 and 73 per cent at 2 | 100 and 97 per cent at 3 frames, 97 and 75 per cent at 2 |
| 151 | SALAD1 mean dwell time at 8 nM, 0.43 to 0.59 of its value at 1 nM over the 64 analysis choices | 0.43 to 0.58 |
| 153 | dark interval 18.9 to 9.1 s (SALAD1) and 17.9 to 15.5 s (TO1-B) | 19.1 to 9.1 s and 18.3 to 15.7 s |
| 153 | dark-interval estimators disagree 45-fold at 1 nM (TO1-B) | 46-fold |
| 153 | goodness of fit from about 0.99 to 0.94 across a fifteen-fold disagreement between estimators | sixteen-fold |

Every other number listed under "Where the thesis numbers are" keeps its printed value.

### Settings the thesis does not state, kept in the code

- Upper limit on bound dwells, entered as right-censoring: the smallest whole number of frames that leaves at most 0.2 per cent of the dwells of every condition beyond it, one limit per fluorogen (`dwell_filters.py`, `TAIL_FRACTION`). It was 30.4 s for SALAD1 and 48.1 s for TO1-B in the thesis run, and is 30.6 s and 47.8 s with the 36 files.
- End rule for signal persistence: a site's span is censored when less observation remains after its last event than the expected dark interval (`analyse_smtirf.py`, `SPAN_CENSORING` and `CENSOR_MARGIN`).
- Repeated binding: the dwell-time fits use molecules with at least two bound dwells (`dwell_filters.py`, `MIN_EVENTS_PER_MOLECULE`).

## Requirements

The environment of the main README, or for part 3 alone `pip install -r smtirf/kinetics/requirements.txt` (numpy, scipy, pandas, h5py, statsmodels and ruptures).
