import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch

from config import Config
from evaluate import load_model, select_eval_windows
from src.dataset import build_datasets, resolve_data_root
from src.masking import eval_mask

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ckpt/cpgd_sens_ecg_s42.pt")
    ap.add_argument("--csdi_ckpt", default="results/ckpt/csdi_sens_ecg_s42.pt")
    ap.add_argument("--modality", default="ecg")
    ap.add_argument("--max_windows", type=int, default=200)
    ap.add_argument("--out", default="results/tables/interval_width.csv")
    ap.add_argument("--data_root", default=None,
                    help="override the dataset root stored in the checkpoint")
    args = ap.parse_args()

    models = {}
    ck0 = args.ckpt
    for name, ck in (("CPGD-Cardio", args.ckpt), ("CSDI", args.csdi_ckpt)):
        models[name], cfg, _, _ = load_model(ck, DEV)

    cfg.data_root = resolve_data_root(cfg, args.data_root)
    cfg2 = Config(data_root=cfg.data_root, modality=cfg.modality, fs=cfg.fs,
                  window=cfg.window, stride=cfg.stride, split=cfg.split,
                  seed=cfg.seed, cache_dir=cfg.cache_dir)
    _, _, te, _ = build_datasets(cfg2)
    C, L = te.n_channels, cfg.window
    sel = select_eval_windows(te, args.max_windows, shuffle=True, eval_seed=0,
                              per_record_cap=25)

    rows = []
    for wid in sel:
        xn = te[wid].numpy()
        mask = eval_mask("blackout20", L, C, window_id=wid, fs=FS)
        mt = torch.from_numpy(mask)[None].to(DEV)
        for name, model in models.items():
            torch.manual_seed(10007 + wid)
            torch.cuda.manual_seed_all(10007 + wid)
            with torch.no_grad():
                s = model.impute(torch.from_numpy(xn)[None].to(DEV), mt,
                                 n_samples=16).cpu().numpy()[:, 0]
            miss = mask == 0
            lo = np.quantile(s, 0.05, axis=0)
            hi = np.quantile(s, 0.95, axis=0)
            width = (hi - lo)[miss]
            truth = xn[miss]
            cov = ((truth >= lo[miss]) & (truth <= hi[miss])).mean()
            rows.append(dict(window=wid, method=name,
                             mean_width=float(width.mean()),
                             coverage=float(cov),
                             n_points=int(miss.sum())))
        if (len(rows) // 2) % 25 == 0:
            print(f"{len(rows)//2} windows ...", flush=True)

    df = pd.DataFrame(rows)
    summ = df.groupby("method")[["mean_width", "coverage"]].mean()
    print(summ.round(4).to_string())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
