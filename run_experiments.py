import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def ckpt(model, dataset, modality, seed):
    return os.path.join(HERE,
                        f"results/ckpt/{model}_{dataset}_{modality}_s{seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+", default=["ecg", "acc", "pcg"])
    ap.add_argument("--dataset", default="sens",
                    choices=["sens", "mitdb", "ptbxl"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--models", nargs="+", default=["cpgd", "csdi", "gru"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--max_windows", type=int, default=200)
    args = ap.parse_args()

    train_extra = ["--epochs", str(args.epochs)] if args.epochs else []
    ds_args = (["--dataset", args.dataset] if args.dataset != "sens" else [])
    ds_tag = args.dataset + "_"
    for mod in args.modalities:
        for seed in args.seeds:
            for model in args.models:
                path = ckpt(model, args.dataset, mod, seed)
                if not args.skip_train and not os.path.exists(path):
                    run([PY, os.path.join(HERE, "train.py"), "--model", model,
                         "--modality", mod, "--seed", str(seed)] + ds_args
                        + train_extra)
            eval_cmd = [PY, os.path.join(HERE, "evaluate.py"),
                        "--ckpt", ckpt("cpgd", args.dataset, mod, seed),
                        "--csdi_ckpt", ckpt("csdi", args.dataset, mod, seed),
                        "--gru_ckpt", ckpt("gru", args.dataset, mod, seed),
                        "--max_windows", str(args.max_windows),
                        "--out", os.path.join(
                            HERE, f"results/eval_{ds_tag}{mod}_s{seed}.csv")]
            run(eval_cmd)

    frames = []
    for mod in args.modalities:
        for seed in args.seeds:
            p = os.path.join(
                HERE, f"results/eval_{ds_tag}{mod}_s{seed}.csv")
            if os.path.exists(p):
                d = pd.read_csv(p)
                d["modality"], d["seed"] = mod, seed
                frames.append(d)
    if not frames:
        print("no eval csvs found -- nothing to aggregate")
        return
    all_df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.join(HERE, "results/tables"), exist_ok=True)
    all_df.to_csv(os.path.join(HERE, "results/tables/all_runs.csv"),
                  index=False)
    g = all_df.groupby(["modality", "protocol", "method"]).agg(
        mse_mu=("mse", "mean"), mse_sd=("mse", "std"),
        mae_mu=("mae", "mean"), crps_mu=("crps", "mean"),
        cov90=("cov90", "mean"), n=("mse", "count")).reset_index()
    g.to_csv(os.path.join(HERE, "results/tables/summary.csv"), index=False)
    try:
        md = g.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        md = g.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    with open(os.path.join(HERE, "results/tables/summary.md"), "w",
              encoding="utf-8") as f:
        f.write("# CPGD-Cardio experiment summary (mean over seeds)\n\n" + md + "\n")
    print("\naggregated -> results/tables/summary.csv, summary.md")
    print(g.head(40).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
