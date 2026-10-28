#!/usr/bin/env python
"""Run the TCSPC and bulk analysis of Chapter 4, start to finish, on Linux, macOS or Windows.

    python run_tcspc.py                      every step
    python run_tcspc.py --only refit ratios  some steps (names below)

Each step is one script, run with the same Python interpreter as this file:

    fits      code/run_fits.py                 every global fit -> code/out/fit_runs.json
                                               (about 25 minutes; finished fits are kept and
                                               skipped when the step is run again)
    shared    code/run_shared.py               shared short-component test -> code/out/
    tables    code/make_tables.py              per-file-window tables -> code/out/
    refit     code/refit_common_window.py      common-window refit, the reported lifetimes
                                               -> code/out/common_window/
    ratios    code/ratio_numbers.py            bulk, lifetime and emitting-fraction numbers
                                               -> code/out/ratio_numbers.csv
    absolute  absolute_route/mango_tcspc_analysis.py
                                               absolute radiative-rate route
                                               -> absolute_route/results/

The ratios step reads one single-molecule table from smtirf/kinetics if it exists; run it
again after the kinetic analysis to fill those rows.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("fits", "code", "run_fits.py"),
    ("shared", "code", "run_shared.py"),
    ("tables", "code", "make_tables.py"),
    ("refit", "code", "refit_common_window.py"),
    ("ratios", "code", "ratio_numbers.py"),
    ("absolute", "absolute_route", "mango_tcspc_analysis.py"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=[s[0] for s in STEPS], metavar="STEP",
                    help="run only these steps: " + ", ".join(s[0] for s in STEPS))
    args = ap.parse_args()
    # UTF-8 for every text file and for the console, whatever the system default
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    todo = [s for s in STEPS if not args.only or s[0] in args.only]
    t_all = time.time()
    for i, (name, folder, script) in enumerate(todo, 1):
        print(f"\n=== {i}/{len(todo)}  {name}: {folder}/{script} ===", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, script], cwd=os.path.join(HERE, folder), env=env)
        if r.returncode != 0:
            sys.exit(f"\nStopped: step '{name}' ended with exit code {r.returncode}.")
        print(f"--- {name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nAll {len(todo)} step(s) finished in {(time.time() - t_all) / 60:.1f} min.")


if __name__ == "__main__":
    main()
