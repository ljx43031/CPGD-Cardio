import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

SOURCES = [("MIT-BIH (2-lead)", "eval_mitdb_ecg_s42.csv"),
           ("PTB-XL (12-lead)", "eval_ptbxl_ecg_s42.csv"),
           ("SensSmartTech ECG", "eval_sens_ecg_s42.csv"),
           ("SensSmartTech ACC", "eval_sens_acc_s42.csv"),
           ("SensSmartTech PCG", "eval_sens_pcg_s42.csv")]
PROTOS = ["blackout20", "seg5x20", "point10", "sensordrop"]
METHODS = ["CPGD-Cardio", "GRU", "linear", "CSDI"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--out", default="results/tables/paper_numbers.json")
    args = ap.parse_args()
    R = lambda f: os.path.join(args.dir, f)

    print("=== Table main: normalized MSE (mean over test windows) ===")
    wins = {}
    csdi = {"w": 0, "n": 0, "ex": []}
    for label, f in SOURCES:
        if not os.path.exists(R(f)):
            print(f"missing {R(f)}")
            continue
        d = pd.read_csv(R(f))
        for p in PROTOS:
            cells = {}
            for m in METHODS:
                v = d[(d.protocol == p) & (d.method == m)]["mse_norm"].mean()
                if v == v:
                    cells[m] = float(v)
            if len(cells) < 2:
                continue
            best = min(cells, key=cells.get)
            wins[best] = wins.get(best, 0) + 1
            if "CSDI" in cells:
                csdi["n"] += 1
                if cells["CPGD-Cardio"] < cells["CSDI"]:
                    csdi["w"] += 1
                else:
                    csdi["ex"].append(f"{label}/{p}")
            s = "  ".join(f"{m}={cells[m]:.3f}" + ("*" if m == best else "")
                          for m in METHODS if m in cells)
            print(f"{label:19s} {p:11s} {s}")
    print(f"CSDI: CPGD-Cardio better in {csdi['w']}/{csdi['n']} cells; "
          f"exceptions: {csdi['ex']}")
    print(f"best-per-cell counts: {wins}")

    print("\n=== Table seeds: SensSmartTech ECG over seeds "
          f"{args.seeds} (mean +- sd) ===")
    frames = []
    for s in args.seeds:
        f = R(f"eval_sens_ecg_s{s}.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f).assign(seed=s))
    if frames:
        ts = pd.concat(frames, ignore_index=True)
        g = ts.groupby(["protocol", "method"]).agg(
            mse_mu=("mse_norm", "mean"), mse_sd=("mse_norm", "std"),
            crps=("crps", "mean"), cov=("cov90", "mean"),
            mae=("mae_norm", "mean"))
        for p in ["blackout20", "seg5x20", "sensordrop"]:
            for m in METHODS:
                if (p, m) not in g.index:
                    continue
                r = g.loc[(p, m)]
                extra = f"  crps={r['crps']:.3f} cov={r['cov']:.2f}" \
                    if m in ("CPGD-Cardio", "CSDI") else ""
                print(f"{p:11s} {m:12s} {r['mse_mu']:.2f} +- {r['mse_sd']:.2f}"
                      + extra)
        summary = {
            "csdi_wins": csdi["w"], "csdi_total": csdi["n"],
            "csdi_exceptions": csdi["ex"], "wins_by_method": wins,
            "crps_blackout20": round(float(g.loc[("blackout20",
                                                   "CPGD-Cardio")]["crps"]), 3),
            "cov_blackout20": round(float(g.loc[("blackout20",
                                                 "CPGD-Cardio")]["cov"]), 2),
            "mse_blackout20": "%.2f +- %.2f" % (
                float(g.loc[("blackout20", "CPGD-Cardio")]["mse_mu"]),
                float(g.loc[("blackout20", "CPGD-Cardio")]["mse_sd"])),
            "gru_mae_blackout20": round(float(
                g.loc[("blackout20", "GRU")]["mae"]), 3),
            "csdi_point10_mse": round(float(
                g.loc[("point10", "CSDI")]["mse_mu"]), 2),
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("\n" + json.dumps(summary, indent=2))
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
