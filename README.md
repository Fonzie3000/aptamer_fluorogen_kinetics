# aptamer_fluorogen_kinetics

Fluorescence lifetime and single-molecule binding kinetics of the Mango II RNA aptamer with two fluorogens, TO1-B and SALAD1. The code compares the two complexes by bulk emission, fluorescence lifetime (time-correlated single photon counting, TCSPC) and single-molecule total internal reflection fluorescence (TIRF) imaging, and writes every result as a CSV table.

This is the analysis code for Chapter 4 of the PhD thesis "Fluorogenic RNA Mango Aptamers: Single-Molecule Characterisation and Applications in CRISPR-Mediated Genomic Imaging" (Fangqi Liu, Imperial College London, 2026).

## Three parts

| part | folder | input | output | thesis |
|---|---|---|---|---|
| 1. TCSPC and bulk fluorescence | `tcspc/` | decay files (`.das`) and bulk spectra, in `tcspc/data/` | lifetimes and amplitudes, the refit over one common window, bulk and emitting-fraction ratios | Chapter 2, section 2.8; Chapter 4, sections 4.3.1 and 4.3.2 |
| 2. Single-molecule traces | `smtirf/mangotrace/` | one raw TIRF movie per acquisition | localisations, drift, binding sites, intensity traces and their change-point idealisation, as `<movie>_CPD.h5` | Chapter 2, sections 2.9.5 and 2.9.6 |
| 3. Single-molecule kinetics | `smtirf/kinetics/` | the 36 CPD-fitted HDF5 files of Chapter 4 | dwell times, dark intervals, fraction bound, signal persistence, intensity, detection and affinity | Chapter 2, sections 2.9.7 to 2.9.10; Chapter 4, sections 4.3.3 to 4.3.8 |

`tcspc/README.md` and `smtirf/README.md` list the steps, the tables each step writes and the table that holds each number quoted in the thesis.

## Installation

The code runs on Linux, macOS and Windows with Python 3.10 to 3.13; on Windows, Python 3.10 to 3.12 on a 64-bit Intel or AMD processor. One environment serves all three parts. From the repository folder, create it:

```bash
python -m venv .venv
```

Activate it:

| system | command |
|---|---|
| macOS, Linux | `source .venv/bin/activate` |
| Windows, PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows, Command Prompt | `.venv\Scripts\activate.bat` |

Install the packages:

```bash
python -m pip install -r requirements.txt
```

Three notes on the interpreter:

- Before the environment is active, macOS and Linux may know the interpreter only as `python3`.
- On Windows with several Python versions installed, `py -3.12 -m venv .venv` creates the environment with Python 3.12.
- If PowerShell refuses to run the activation script, use Command Prompt, or call the environment's interpreter directly, for example `.venv\Scripts\python.exe tcspc/run_tcspc.py`.

Each part also has its own `requirements.txt`. The trace analysis needs the `dbscan` package: on Windows its only prebuilt version, 0.0.12, needs NumPy below 2, and `requirements.txt` selects both there.

## Running

With the environment active, the same commands work on every system, from the repository folder:

```bash
python tcspc/run_tcspc.py
python smtirf/mangotrace/run_mangotrace.py path/to/movie.ome.tif
python smtirf/kinetics/run_kinetics.py --data path/to/folder_with_the_36_files
```

Paths may be relative or absolute, and on Windows either `\` or `/` works. Quote a path that contains spaces. Each part writes its results beside its own code.

## Data

- TCSPC and bulk: in `tcspc/data/`. Nine decay files hold the 28 acquisitions; beside them are the manufacturer's fit reports, the bulk screen spectra and the stem-series readings.
- Single molecule: Chapter 4 uses 36 acquisitions, listed in `smtirf/kinetics/FILES_CHAPTER4.txt`. The 36 HDF5 files (about 15 GB) and the raw movies are too large for this repository and are available from the author.

## Citation

If you use this code, please cite the thesis:

Fangqi Liu (2026). Fluorogenic RNA Mango Aptamers: Single-Molecule Characterisation and Applications in CRISPR-Mediated Genomic Imaging. PhD thesis, Imperial College London.

GitHub's "Cite this repository" button gives the same reference from `CITATION.cff`.

## Licence

The code is released under the MIT licence; see `LICENSE`. The parts of `smtirf/mangotrace/mangotrace_core/` adapted from Picasso keep Picasso's own MIT notice; see `THIRD_PARTY_NOTICES.md`.

## Contact

Fangqi Liu, MRC Laboratory of Medical Sciences and Imperial College London
