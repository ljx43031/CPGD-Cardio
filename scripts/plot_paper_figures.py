import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from evaluate import load_model
from src.dataset import build_datasets, resolve_data_root
from src.masking import eval_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csdi_ckpt", default=None)
    ap.add_argument("--modality", default=None)
    ap.add_argument("--window_id", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--csv", default=None,
                    help="eval CSV for the metric bar figure "
                         "(default: results/eval_<dataset>_<modality>_s<seed>.csv)")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--data_root", default=None,
                    help="override the dataset root stored in the checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, mean, std = load_model(args.ckpt, device)
    C, L = model.C, cfg.window
    if mean.size != C:
        mean = np.full(C, float(mean.ravel()[0]), np.float32)
        std = np.full(C, float(std.ravel()[0]), np.float32)
    cfg.data_root = resolve_data_root(cfg, args.data_root)
    cfg2 = Config(data_root=cfg.data_root, modality=cfg.modality, fs=cfg.fs,
                  window=cfg.window, stride=cfg.stride, split=cfg.split,
                  seed=cfg.seed, cache_dir=cfg.cache_dir)
    _, _, te, _ = build_datasets(cfg2, synthetic=False)
    xb = next(iter(DataLoader(te, batch_size=max(args.window_id + 1, 8),
                              num_workers=0))).numpy()
    os.makedirs(args.outdir, exist_ok=True)

    ch = 0
    mask = eval_mask("blackout20", L, C, window_id=args.window_id)
    x = xb[args.window_id] * std[:, None] + mean[:, None]
    xn = torch.from_numpy(xb[args.window_id:args.window_id + 1]).to(device)

    def samples_of(path):
        m, _, _, _ = load_model(path, device)
        with torch.no_grad():
            s = m.impute(xn, torch.from_numpy(mask)[None].to(device),
                         n_samples=args.n_samples)
        return s.cpu().numpy()[:, 0] * std[None, :, None] + \
            mean[None, :, None]

    panels = [("CPGD-Cardio", samples_of(args.ckpt))]
    if args.csdi_ckpt and os.path.exists(args.csdi_ckpt):
        panels.append(("CSDI", samples_of(args.csdi_ckpt)))

    fig, axes = plt.subplots(len(panels) + 1, 1,
                             figsize=(11, 2.6 * (len(panels) + 1)),
                             sharex=True)
    t = np.arange(L) / cfg.fs
    axes[0].plot(t, x[ch], color="0.3", lw=1.0, label="ground truth")
    axes[0].plot(t[mask[ch] == 0], x[ch][mask[ch] == 0], "r.", ms=2,
                 label="missing (eval)")
    axes[0].set_ylabel("signal")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(f"{cfg.modality.upper()}  ch{ch}  blackout20 protocol")
    for ax, (name, s) in zip(axes[1:], panels):
        mean_p, lo, hi = s.mean(0), *np.quantile(s, [0.05, 0.95], axis=0)
        ax.fill_between(t, lo[ch], hi[ch], alpha=0.30, label="90% band")
        ax.plot(t, mean_p[ch], lw=1.0, label=f"{name} mean")
        ax.plot(t, x[ch], color="0.3", lw=0.7, alpha=0.6)
        ax.set_ylabel(name, fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    f1 = os.path.join(args.outdir, f"fig1_reconstruction_{cfg.modality}.png")
    fig.savefig(f1, dpi=300)
    print("saved", f1)

    cand = args.csv or (f"results/eval_{cfg.dataset}_{cfg.modality}"
                        f"_s{cfg.seed}.csv")
    if os.path.exists(cand):
        df = pd.read_csv(cand)
        piv = df.groupby(["protocol", "method"])["mse_norm"].mean().unstack()
        ax = piv.plot(kind="bar", figsize=(8, 3.6), logy=True)
        ax.set_ylabel("standardised MSE (log)")
        ax.set_title("Imputation error by missing-data protocol")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0, frameon=False)
        fig2 = ax.get_figure()
        f2 = os.path.join(args.outdir, f"fig2_metrics_{cfg.modality}.png")
        fig2.savefig(f2, dpi=300, bbox_inches="tight")
        print("saved", f2)
    else:
        print(f"no eval csv at {cand}; fig2 skipped")


if __name__ == "__main__":
    main()
