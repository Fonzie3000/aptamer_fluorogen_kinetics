#!/usr/bin/env python
"""Run the single-molecule kinetic analysis of Chapter 4, start to finish, on Linux, macOS or Windows.

    python run_kinetics.py --data FOLDER                     every step
    python run_kinetics.py --data FOLDER --skip-validation   without the two simulation steps
    python run_kinetics.py --only filters kd                 some steps (names below)

FOLDER holds the 36 CPD-fitted HDF5 files listed in FILES_CHAPTER4.txt, directly or in
subfolders. The analysis reads every .h5 file it finds there, so the runner stops if the
folder holds any other .h5 file (override with --allow-other-files). Without --data the
folder named by the environment variable SMTIRF_DATA is used, or smtirf/TIRF.

Each step is one script in this folder, run with the same Python interpreter as this file:

    analysis      analyse_smtirf.py          quality checks, intensity, detection, fraction
                                             bound, signal persistence, dwell times; per-file
                                             extracts in cache/
    curves        occupancy_curves.py        fraction bound against time, per condition
    validation    simulate_validate.py       simulated traces with known rates (needs ruptures)
    truncation    run_truncation_study.py    detection efficiency per minimum duration (needs ruptures)
    window        window_select.py           the analysis window (without ruptures: the adopted
                                             rule is applied without scoring the candidates)
    estimators    cross_estimators.py        dark-interval estimators and window diagnostics
    cascade       plot_spec.py               dwell filter cascade and display limits
    filters       dwell_filters.py           repeated-binding filter, upper limits, dwell fits
    interior      interior_kinetics.py       dark intervals closed by two events
    compare       fit_compare.py             one against two exponential components
    sensitivity   sensitivity_adopted.py     the 64-combination grid of analysis choices
    kd            kd_per_condition.py        fraction bound at the first frame, per condition, and
                                             the dissociation constant by the two-point route

Every table goes to tables/ beside the scripts.
"""
import argparse
import glob
import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "FILES_CHAPTER4.txt")
STEPS = [
    ("analysis", "analyse_smtirf.py"),
    ("curves", "occupancy_curves.py"),
    ("validation", "simulate_validate.py"),
    ("truncation", "run_truncation_study.py"),
    ("window", "window_select.py"),
    ("estimators", "cross_estimators.py"),
    ("cascade", "plot_spec.py"),
    ("filters", "dwell_filters.py"),
    ("interior", "interior_kinetics.py"),
    ("compare", "fit_compare.py"),
    ("sensitivity", "sensitivity_adopted.py"),
    ("kd", "kd_per_condition.py"),
]
NEED_RUPTURES = {"validation", "truncation"}


def check_data(folder, allow_other):
    """The data folder must hold the 36 listed files and, by default, no other .h5 file."""
    if not os.path.isdir(folder):
        sys.exit(f"Data folder not found: {folder}\nGive it with --data FOLDER.")
    with open(FILES, encoding="utf-8") as fh:
        listed = [s.strip() for s in fh if s.strip() and not s.startswith("#")]
    found = glob.glob(os.path.join(folder, "**", "*.h5"), recursive=True)
    names = {os.path.basename(p).lower() for p in found}
    missing = [n for n in listed if n.lower() not in names]
    other = sorted(os.path.basename(p) for p in found
                   if os.path.basename(p).lower() not in {n.lower() for n in listed})
    if missing:
        sys.exit(f"{len(missing)} of the {len(listed)} files in FILES_CHAPTER4.txt are not under {folder}:\n  "
                 + "\n  ".join(missing))
    if other and not allow_other:
        sys.exit(f"{folder} also holds {len(other)} .h5 file(s) that Chapter 4 does not use, and the "
                 "analysis would read them:\n  " + "\n  ".join(other)
                 + "\nMove them out of the folder, or pass --allow-other-files to include them.")
    print(f"data: {folder} ({len(listed)} files)", flush=True)


def check_cache(allow_other):
    """Later steps read every extract in cache/, so it must hold only the 36 files."""
    with open(FILES, encoding="utf-8") as fh:
        listed = {s.strip()[:-3].lower() + ".pkl" for s in fh if s.strip().endswith(".h5")}
    found = glob.glob(os.path.join(HERE, "cache", "*.pkl"))
    other = sorted(os.path.basename(p) for p in found if os.path.basename(p).lower() not in listed)
    if other and not allow_other:
        sys.exit(f"cache/ holds {len(other)} extract(s) of files that Chapter 4 does not use, and the "
                 "later steps would read them:\n  " + "\n  ".join(other)
                 + f"\nDelete them (or the whole folder {os.path.join(HERE, 'cache')}), "
                 "or pass --allow-other-files.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("SMTIRF_DATA",
                                                     os.path.join(HERE, os.pardir, "TIRF")),
                    help="folder holding the 36 HDF5 files")
    ap.add_argument("--skip-validation", action="store_true",
                    help="skip the two simulation steps (validation, truncation)")
    ap.add_argument("--only", nargs="+", choices=[s[0] for s in STEPS], metavar="STEP",
                    help="run only these steps: " + ", ".join(s[0] for s in STEPS))
    ap.add_argument("--allow-other-files", action="store_true",
                    help="let the analysis read .h5 files, or cached extracts, of files that "
                         "FILES_CHAPTER4.txt does not list")
    args = ap.parse_args()

    data = os.path.abspath(args.data)
    todo = [s for s in STEPS if not args.only or s[0] in args.only]
    if any(name == "analysis" for name, _ in todo):
        check_data(data, args.allow_other_files)
    check_cache(args.allow_other_files)
    have_ruptures = importlib.util.find_spec("ruptures") is not None

    # UTF-8 for every text file and for the console, whatever the system default
    env = dict(os.environ, SMTIRF_DATA=data, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t_all = time.time()
    for i, (name, script) in enumerate(todo, 1):
        cmd = [sys.executable, script]
        if name in NEED_RUPTURES and args.skip_validation:
            print(f"\n=== {i}/{len(todo)}  {name}: skipped (--skip-validation) ===", flush=True)
            continue
        if name in NEED_RUPTURES and not have_ruptures:
            print(f"\n=== {i}/{len(todo)}  {name}: skipped, ruptures is not installed "
                  f"(install it with:  {os.path.basename(sys.executable)} -m pip install ruptures) ===",
                  flush=True)
            continue
        if name == "window" and not have_ruptures:
            print("ruptures is not installed: the adopted window rule is applied without scoring the candidates.")
            cmd += ["--stage", "apply"]
        print(f"\n=== {i}/{len(todo)}  {name}: {' '.join(cmd[1:])} ===", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=HERE, env=env)
        if r.returncode != 0:
            sys.exit(f"\nStopped: step '{name}' ended with exit code {r.returncode}.")
        print(f"--- {name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nAll step(s) finished in {(time.time() - t_all) / 60:.1f} min. Tables are in "
          f"{os.path.join(HERE, 'tables')}.")


if __name__ == "__main__":
    main()
