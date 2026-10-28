# Part 1: TCSPC and bulk fluorescence

Code and data behind the lifetimes and the bulk-emission ratios of Chapter 4 (sections 4.3.1 and 4.3.2), using the method of Chapter 2, section 2.8. The part reads the decay files, fits the decays, refits them over one common window and derives the bulk, lifetime and emitting-fraction ratios. Every result is a CSV table.

## Run

From the repository folder, with the environment of the main README active:

```bash
python tcspc/run_tcspc.py                       # every step
python tcspc/run_tcspc.py --only refit ratios   # some steps
```

| step | script | writes |
|---|---|---|
| `fits` | `code/run_fits.py` | `code/out/fit_runs.json`: every global fit of the per-file-window analysis. About 25 minutes on a recent laptop; finished fits are kept, and a rerun skips them. |
| `shared` | `code/run_shared.py` | `code/out/shared_short_component.csv`, `shared_short_summary.csv`: the test of a short component shared by the two fluorogens |
| `tables` | `code/make_tables.py` | `code/out/*.csv`: window comparison, window sensitivity, model order, scatter term, colour shift, the per-file-window lifetimes, and the fitted curves of the 8 and 12 bp decays |
| `refit` | `code/refit_common_window.py` | `code/out/common_window/*.csv`: the refit over one common window, channels 232 to 1,411, which gives the reported lifetimes. About 1 minute. |
| `ratios` | `code/ratio_numbers.py` | `code/out/ratio_numbers.csv`: bulk, lifetime and emitting-fraction numbers, one row each, with the thesis page |
| `absolute` | `absolute_route/mango_tcspc_analysis.py` | `absolute_route/results/*.csv`: the absolute radiative-rate route |

Each script also runs on its own, from any folder, for example `python tcspc/code/refit_common_window.py`.

The `ratios` step reads one table of part 3, `smtirf/kinetics/tables/brightness_ratio_by_condition.csv`, for its single-molecule rows. Before the kinetic analysis has run, those rows stay empty; afterwards, `python tcspc/run_tcspc.py --only ratios` fills them.

## Where the thesis numbers are

| page | number | file, column |
|---|---|---|
| 92 | common fit window, channels 232 to 1,411 (12.8 to 77.9 ns) | `code/out/common_window/fit_windows.csv`, `new_lo`, `new_hi`, `new_lo_ns`, `new_hi_ns` |
| 92 | lifetimes 0.4 to 0.6 per cent lower than with per-file window ends | `code/out/common_window/common_window_changes.csv`, `tau_amp_change_pct` |
| 93 | a third component's lifetime changes about twofold with the end of the fit range | `code/out/model_order.csv`, `tau3_ns` under the two windows |
| 138 | bulk ratio 2.04 at the band maximum and 2.16 integrated; A22U 1.22 (TO1-B) and 0.55 (SALAD1); best mutant combination 60 per cent; emission maxima 532.9 and 534.8 nm | `code/out/ratio_numbers.csv` |
| 140-141 | mean lifetimes 3.01 and 3.57 ns; components 2.08 and 4.69 ns (TO1-B), 2.04 and 5.40 ns (SALAD1); long-state share 36 and 46 per cent | `code/out/common_window/primary_report.csv`, 12 bp rows, `tau_amp_ns`, `tau1_ns`, `tau2_ns`, `a2_pct` |
| 141 | lifetime ratio 1.19 (1.186); split of the 0.56 ns difference, about 0.30 ns population and 0.26 ns lifetime; spread 0.52 per cent (TO1-B) and 1.8 per cent (SALAD1); 1.72; 0.98 | `code/out/ratio_numbers.csv` |
| 143 | extinction ratio 0.907; permitted radiative-rate ratio 0.72 to 1.17; 1.90; radiative term 0.952; requirement 1.99, interval 1.49 to 2.68; upper bounds about 50 and 58 per cent | `code/out/ratio_numbers.csv` |
| 143 | absolute route: 0.31 to 0.52 of the ceiling; about 2,400 cm-1; quantum yield near 0.51 against a published 0.72 | `absolute_route/results/strickland_berg_absolute.csv` |
| 145 | single-molecule brightness ratio 1.08 to 1.24, median 1.16 | `code/out/ratio_numbers.csv` (from part 3) |

## Data (`data/`)

| file | content |
|---|---|
| `*_batch.das` (9 files) | decays and instrument response, 28 acquisitions: TO1-B on seven constructs from 6 to 18 bp (four acquisitions at 6 bp, three at each other length) and SALAD1 on the 8 and 12 bp constructs (three each) |
| `*_bi_fit.txt`, `*_tri_fit.txt` | the manufacturer's two- and three-component fit reports; `run_fits.py` takes the manufacturer's fit window from them for the window comparison, and no reported lifetime comes from them |
| `tcspc_global_fit_results.csv` | an earlier global fit, read by `make_tables.py` for comparison only |
| `screen_spectra_triplicates.xlsx` | emission spectra of the four aptamer-fluorogen combinations, three replicates each, water blank subtracted once |
| `stem_Brightness.xlsx` | bulk readings across the stem series, three replicates |

## Code (`code/`)

| file | role |
|---|---|
| `das_io.py` | reads the decay files and the fit reports |
| `fit_tcspc.py` | the estimator: iterative reconvolution with the measured instrument response, two exponential components, a scattered-light term and a constant baseline, Poisson maximum likelihood, lifetimes shared by every acquisition of one fluorogen, and the fit-range rule |
| `run_fits.py` | runs every global fit |
| `run_shared.py`, `extras.py` | the shared short-component test and the slow-tail checks |
| `make_tables.py` | the tables of the per-file-window analysis |
| `refit_common_window.py` | refits every decay over one common window |
| `das_reader.py`, `das_fit.py`, `das_global.py` | the decay reader and loader that `refit_common_window.py` uses |
| `TCSPC_reanalysis_tables.xlsx` | the tables of the per-file-window run, collected in one workbook; `refit_common_window.py` labels the decays with its acquisition inventory, compares the refit with its per-file results, and copies its other sheets into `out/common_window/` unchanged |
| `ratio_numbers.py` | the bulk, lifetime and emitting-fraction numbers |

## Notes

- The decay files carry no time calibration. The channel width, 0.055176 ns (226 ns over 4,096 channels), comes from the acquisition record and is set in `fit_tcspc.py`.
- `TCSPC_reanalysis_tables.xlsx` is the one stored intermediate. No script here writes its acquisition inventory or six of its sheets: `fixed_tail_test`, the two `baseline_hypothesis_test` sheets, `mean_lifetime_decomposition`, `validation_A_vs_das6` and `vendor_perfile_fits`. The refit copies them into `out/common_window/` unchanged.
- The fits use no random numbers, but each global fit stops where its search meets its tolerance, so a result can move slightly with the platform and the library versions. A rerun of the refit on Python 3.10, NumPy 2.2 and SciPy 1.15 (Linux, ARM) gave TO1-B lifetimes of 2.09 and 4.71 ns against 2.08 and 4.69 ns in the thesis; the SALAD1 values agreed within 0.04 per cent.
- `absolute_route/mango_tcspc_analysis.py` is an earlier analysis script, with its six inputs in `absolute_route/data/`. Chapter 4 uses one of its tables, `strickland_berg_absolute.csv`; the others are not reported in the thesis.
