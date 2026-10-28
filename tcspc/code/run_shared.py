"""Section 5, Q1: is the short component genuinely shared between the two fluorogens?

Fits both series jointly with tau1 forced to one common value and tau2 free per dye,
and compares against the sum of the two independent series fits.
"""

import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import extras
import run_fits

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

acqs, vend = run_fits.build()
run_fits.set_window(acqs, vend, "rule")
by_dye = {d: [a for a in acqs if a.dye == d] for d in ("TO1-B", "SALAD1")}
runs = json.load(open(os.path.join(OUT, "fit_runs.json"), encoding="utf-8"))

tbl, summary = extras.shared_short_test(runs, by_dye)
tbl.to_csv(os.path.join(OUT, "shared_short_component.csv"), index=False)
pd.DataFrame([summary]).to_csv(os.path.join(OUT, "shared_short_summary.csv"), index=False)
print(tbl.to_string(index=False))
print(summary)
