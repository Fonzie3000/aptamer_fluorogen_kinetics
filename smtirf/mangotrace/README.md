# MangoTrace headless pipeline: raw movie to CPD-fitted HDF5

One script turns a raw single-molecule TIRF movie into a file of idealized binding
traces and events, with every analysis setting written in the script itself. It runs
the same numerical functions as the MangoTrace GUI, without the GUI, on Linux, macOS and
Windows.

## What is in this folder

| Path | Purpose |
|---|---|
| `run_mangotrace.py` | the pipeline; settings in the `SETTINGS` block at the top |
| `mangotrace_core/` | copy of the MangoTrace numerical core (localization, drift, clustering, trace extraction, change-point idealization, event extraction). No GUI code. |
| `requirements.txt` | pip dependencies |
| `environment.yml` | conda environment with the same dependencies |
| `README.md` | this file |

## What the pipeline does

1. **Localization, pass 1** on the raw movie: 7 x 7 net-gradient detection (threshold 256
   counts), photon conversion (baseline 400, sensitivity 1, gain 1, QE 0.82), elliptical
   Gaussian least-squares fit, Mortensen precision. Restricted to the ROI x 15-560, y 15-560.
2. **Drift estimate** by adaptive intersection maximization (AIM): 100-frame segments,
   0.154 px grid, 0.462 px search radius, two passes, cubic-spline interpolation.
3. **Undrifted movie**: every frame translated by the negative drift with bilinear
   interpolation and written as `<stem>_undrifted.tif`.
4. **Localization, pass 2** on the undrifted movie, same settings.
5. **Localization filter** (inclusive ranges): sigma x, sigma y 0.55-2.7 px; precision x, y
   0-0.36 px; photons 64-10000.
6. **DBSCAN** on the filtered localizations: radius 0.4 px, minimum 10 localizations;
   clusters smaller than 10 dropped; one site per cluster at the centroid.
7. **Trace extraction** from the undrifted movie: mean of the 13 pixels within 2 px minus
   the median of the 52-pixel annulus from 3 to 5 px, times 13 (integrated counts above
   local background).
8. **Change-point idealization** per trace: robust noise estimate, PELT with minimum
   segment 1 frame and penalty 10 sigma squared, Otsu two-level split, 3 sigma gate; then
   binding events with start, end (exclusive), duration, mean intensity, SNR and
   censoring flags.
9. **Export** `<stem>_CPD.h5` (MangoTrace `hmm_results` format version 4).

## Installation

Python 3.10 to 3.13; on Windows, Python 3.10 to 3.12. The environment of the main README of this repository covers this part. For this part alone, on Linux or macOS, either

```bash
conda env create -f environment.yml
conda activate mangotrace
```

or, on any system,

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `dbscan` package (exact grid DBSCAN, wangyiqiu/dbscan-python) has prebuilt versions for
Linux and macOS. On Windows its newest prebuilt version is 0.0.12, built for NumPy 1.x, so
`requirements.txt` installs that version there with NumPy below 2; this needs Python 3.10 to
3.12. On about 250,000 simulated localisations, versions 0.0.12 and 1.0.0 gave identical
cluster labels.

No GPU libraries are needed. The first run compiles the numba kernels (about a minute);
later runs reuse the cache, which is placed in a per-user cache directory on every OS.

## Running

```bash
python run_mangotrace.py /data/SALAD0.5nM_1.5mW_1_MMStack_Pos0.ome.tif
python run_mangotrace.py movie1.tif movie2.tif --outdir /data/results
python run_mangotrace.py movie.tif --skip-undrift      # reuse an existing <stem>_undrifted.tif
python run_mangotrace.py --print-settings > my_settings.json
python run_mangotrace.py movie.tif --settings my_settings.json
```

Paths may be absolute or relative, with forward or back slashes. Quote paths that
contain spaces. Several movies are processed in sequence; a failure in one is reported
and the others continue.

A 2500-frame, 575 x 575 pixel movie (1.6 GB) takes about 4.5 minutes on a laptop with
16 cores; peak memory is roughly twice the movie size.

## Outputs (next to the movie, or in `--outdir`)

| File | Content |
|---|---|
| `<stem>_locs_pass1.csv` | localizations of the raw movie: frame, x, y (px), photons, sx, sy, bg, lpx, lpy, net_gradient |
| `<stem>_drift.csv` | per-frame drift in pixels (x, y), zero at frame 0 |
| `<stem>_undrifted.tif` | drift-corrected movie, same dtype and size as the input |
| `<stem>_locs_filtered.csv` | pass-2 localizations that passed the filter |
| `<stem>_clusters.csv` | filtered localizations with their DBSCAN `cluster` id (-1 = noise) |
| `<stem>_CPD.h5` | traces, states, idealized levels, events, site coordinates |
| `<stem>_pipeline_log.json` | settings used, counts per step, timings, platform |

### Layout of `<stem>_CPD.h5`

Root attributes `mangotrace_format = "hmm_results"`, `format_version = 4`, `idealizer = "cpd"`,
`filtered = False`, `source_file`. Group `channel_results/0` with datasets `traces`,
`states` (0 unbound, 1 bound), `idealized`, `time` (frame index), all `(frames, molecules)`;
`molecule_ids` (DBSCAN cluster ids), `locations` (x, y in px), `trace_lengths`; and
`events/<i>/records`, a JSON list per molecule of dicts with `Start`, `End` (exclusive),
`Duration` (frames), `MeanIntensity`, `SNR`, `StartIdx`, `EndIdx`, `StartTrunc`, `EndTrunc`.

Read it with any HDF5 tool:

```python
import json, h5py
with h5py.File("movie_CPD.h5") as f:
    g = f["channel_results/0"]
    traces, states = g["traces"][:], g["states"][:]
    ids = [s.decode() for s in g["molecule_ids"][:]]
    events = [json.loads(g[f"events/{i}/records"][()]) for i in range(len(ids))]
```

The file also loads in the MangoTrace GUI (Kinetics, Load HMM Results). Part 3 of this
repository, `../kinetics/`, reads it for the dwell-time analysis, after renaming to the
pattern described in `../README.md`.

## Changing settings

Edit the `SETTINGS` block or pass a JSON file with only the keys you want to change,
for example:

```json
{"dbscan": {"radius": 0.5}, "cpd": {"penalty": 15}, "roi": null}
```

| Key | Meaning |
|---|---|
| `detect_channel` | 0 for single-channel movies; 1-based channel for multi-channel stacks |
| `localization` | detection threshold, box size, camera conversion, worker threads (clamped to the CPU count), `use_gpu` (keep false) |
| `roi` | `{x_min, y_min, x_max, y_max}` in pixels, or `null` for the full frame |
| `frames` | `[start, stop]` frame range or `null` |
| `aim` | `segmentation` (frames), `intersect_d` and `roi_r` (px) |
| `filter` | inclusive `[min, max]` per localization column |
| `dbscan` | `radius` (px), `min_locs` |
| `dbscan_input` | `"filtered"` (default) or `"raw"` pass-2 localizations |
| `trace` | `inner_radius`, `outer_radius`, `buffer` (px), `signal_method` and `bg_method` (`mean` or `median`) |
| `cpd` | `min_size` (frames), `penalty` (sigma squared units), `denoise` (keep false: the Bayesian denoiser delays edges by 1-2 frames) |
| `output_suffix` | name of the fitted file |

## Notes

- The first run creates a `_workspace/cache` folder next to the script for the compiled
  numba kernels (or uses the per-user cache directory when that folder is not writable).
  It can be deleted at any time.
- Drift estimation is deterministic, so re-running on the same raw movie reproduces the
  same undrifted movie and the same result.
- The GUI's second localization pass, filter, clustering, extraction and CPD are the
  same functions; given the same undrifted movie and settings, the output is identical.
- Platform-specific code paths inside `mangotrace_core` (a Windows memory query) are
  guarded and fall back to `psutil` elsewhere.
- Movies on cloud-synced folders (OneDrive, Dropbox) must be fully downloaded; the
  reader handles online-only stubs on Windows by copying them to a temporary file.
