import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from config import Config
from src.dataset import build_datasets, resolve_data_root

PROTOS = ["blackout20", "seg5x20", "point10", "sensordrop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint whose config defines the dataset")
    ap.add_argument("--method", default="CPGD-Cardio")
    ap.add_argument("--baseline", default="CSDI")
    ap.add_argument("--data_root", default=None,
                    help="override the dataset root stored in the checkpoint")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in known})
    cfg.data_root = resolve_data_root(cfg, args.data_root)
    _, _, te, _ = build_datasets(cfg)
    df = pd.read_csv(args.eval_csv)

    starts = {}
    for w in df.window.unique():
        if w < len(te.index):
            starts[w] = te.index[w][1]
    df = df[df.window.isin(starts)].copy()
    df["start"] = df.window.map(starts)
    overlap = "non-overlapping by construction" \
        if cfg.stride >= cfg.window else f"window={cfg.window} stride={cfg.stride}"
    sub = df if cfg.stride >= cfg.window else df[df.start % cfg.window == 0]
    print(f"{args.eval_csv}: {overlap}; "
          f"{sub.window.nunique()}/{df.window.nunique()} windows in the "
          "non-overlapping subset")
    for p in PROTOS:
        a = sub[(sub.protocol == p) & (sub.method == args.method)] \
            .set_index("window")["mse_norm"]
        b = sub[(sub.protocol == p) & (sub.method == args.baseline)] \
            .set_index("window")["mse_norm"]
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        _, pv = wilcoxon(a.loc[common].values, b.loc[common].values)
        print(f"{p:11s} n={len(common):3d}  "
              f"{args.method} {a.loc[common].mean():.3f} vs "
              f"{args.baseline} {b.loc[common].mean():.3f}  "
              f"p={pv:.2e}")


if __name__ == "__main__":
    main()
