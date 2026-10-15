"""Cycle period of the ten chemostat runs, from the peaks of the algae density.

Reads DATA_PATH/C*.csv (default data/) and writes periods.csv and periods.png
to RESULTS_PATH (default results/); both variables may be set in .env.
"""
import csv
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA = Path(os.environ.get("DATA_PATH", "data"))
RESULTS = Path(os.environ.get("RESULTS_PATH", "results"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is set)

MIN_GAP = 3.0  # minimum spacing between two peaks, in days


def load_run(path):
    """Time (days) and algae density (10^6 cells/ml) of one run, missing samples dropped."""
    d = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
    t, algae = d[:, 0], d[:, 1]
    keep = ~np.isnan(algae)
    return t[keep], algae[keep]


def peaks(t, algae):
    """Indices of the local maxima above the mean density, at least MIN_GAP days apart."""
    level = algae.mean()
    found = []
    for i in range(1, len(algae) - 1):
        if not (algae[i - 1] < algae[i] >= algae[i + 1] and algae[i] > level):
            continue
        if found and t[i] - t[found[-1]] < MIN_GAP:
            if algae[i] > algae[found[-1]]:
                found[-1] = i
            continue
        found.append(i)
    return np.array(found, dtype=int)


def cycle_period(t, algae):
    """Mean peak-to-peak spacing in days, and the peak times."""
    p = peaks(t, algae)
    if len(p) < 3:
        return np.nan, t[p]
    return (t[p[-1]] - t[p[0]]) / (len(p) - 1), t[p]


def main():
    files = sorted(DATA.glob("C*.csv"), key=lambda f: int(f.stem[1:]))
    if not files:
        raise SystemExit(f"analyse.py: no C*.csv found in {DATA} (set DATA_PATH)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, peak_times = [], []
    for f in files:
        t, algae = load_run(f)
        period, tp = cycle_period(t, algae)
        rows.append((f.stem, len(t), len(tp), period))
        peak_times.append(tp)
    # peak times padded to a common length, one row per run
    width = max(len(tp) for tp in peak_times)
    table = np.full((len(files), width), np.nan)
    for i, tp in enumerate(peak_times):
        table[i, : len(tp)] = tp
    print(f"{'run':<5}{'samples':>9}{'peaks':>7}{'period (days)':>15}")
    for name, n, k, period in rows:
        print(f"{name:<5}{n:>9}{k:>7}{period:>15.3f}")
    with open(RESULTS / "periods.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("run", "samples", "peaks", "period_days"), *rows])
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, _, _, period) in enumerate(rows):
        ax.plot(table[i], np.full(width, i), "|", ms=10, color="C0")
    ax.set_yticks(range(len(rows)), [f"{r[0]}  ({r[3]:.2f} d)" for r in rows])
    ax.set_xlabel("time (days)")
    ax.set_title("Algae peaks of each run")
    fig.savefig(RESULTS / "periods.png", dpi=150, bbox_inches="tight")
    print(f"written: {RESULTS / 'periods.csv'}, {RESULTS / 'periods.png'}")


if __name__ == "__main__":
    main()
