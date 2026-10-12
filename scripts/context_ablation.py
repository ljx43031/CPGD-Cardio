import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch

from config import Config
from evaluate import load_model, select_eval_windows
from src.dataset import build_datasets, resolve_data_root
from src.metrics import masked_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FS = 100
GAP_FRAC = 0.2


def mask_with_gap(L, C, pos):
    g = int(L * GAP_FRAC)
    starts = {"head": 0, "middle": (L - g) // 2, "tail": L - g}
    M = np.ones((C, L), dtype=np.float32)
    M[:, starts[pos]:starts[pos] + g] = 0.0
    return M


def run(args):
    model, cfg, _, _ = load_model(args.ckpt, DEV)
    cfg.data_root = resolve_data_root(cfg, getattr(args, "data_root", None))
    cfg2 = Config(data_root=cfg.data_root, modality=cfg.modality, fs=cfg.fs,
                  window=cfg.window, stride=cfg.stride, split=cfg.split,
                  seed=cfg.seed, cache_dir=cfg.cache_dir)
    _, _, te, _ = build_datasets(cfg2)
    C, L = te.n_channels, cfg.window
    sel = select_eval_windows(te, args.max_windows, shuffle=True, eval_seed=0,
                              per_record_cap=5)
    rows = []
    for wi in sel:
        xn = te[wi].numpy()
        for pos in ("head", "middle", "tail"):
            mask = mask_with_gap(L, C, pos)
            torch.manual_seed(10007 + wi)
            torch.cuda.manual_seed_all(10007 + wi)
            with torch.no_grad():
                s = model.impute(torch.from_numpy(xn)[None].to(DEV),
                                 torch.from_numpy(mask)[None].to(DEV),
                                 n_samples=16)
            pred = s.mean(0)[0].cpu().numpy()
            sq_n, _ = masked_errors(pred, xn, 1.0 - mask)
            rows.append(dict(window=wi, position=pos,
                             mse_norm=float(sq_n.mean())))
    df = pd.DataFrame(rows)
    summ = df.groupby("position")["mse_norm"].mean().to_dict()
    mid = summ["middle"]
    summ_out = {p: round(v, 4) for p, v in summ.items()}
    summ_out["head_vs_middle_pct"] = round(100 * (summ["head"] / mid - 1), 1)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/context_ablation.csv", index=False)
    with open("results/tables/context_ablation.json", "w",
              encoding="utf-8") as f:
        json.dump({"summary": summ_out, "n_windows": len(sel)}, f, indent=1)
    print(df.groupby("position")["mse_norm"].agg(["mean", "std"]))
    print("saved: results/tables/context_ablation.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ckpt/cpgd_sens_ecg_s42.pt")
    ap.add_argument("--max_windows", type=int, default=24)
    ap.add_argument("--data_root", default=None,
                    help="override the dataset root stored in the checkpoint")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
