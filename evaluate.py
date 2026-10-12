import argparse
import dataclasses
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from src.dataset import build_datasets, resolve_data_root
from src.masking import eval_mask, protocols_for
from src.metrics import (crps_empirical, fmt_p, interval_coverage,
                         masked_errors, paired_wilcoxon)

CLASSICAL = ["linear", "spline", "mean", "locf"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="CPGD-Cardio checkpoint")
    ap.add_argument("--csdi_ckpt", default=None)
    ap.add_argument("--gru_ckpt", default=None)
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--max_windows", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--shuffle", action="store_true",
                    help="fixed-seed shuffle of the evaluation-window order")
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--per_record_cap", type=int, default=0,
                    help="max windows kept per record before shuffling (0 = off)")
    ap.add_argument("--data_root", default=None,
                    help="override the dataset root stored in the checkpoint")
    return ap.parse_args()


def select_eval_windows(te, max_windows, shuffle=False, eval_seed=0,
                        per_record_cap=0):
    idx = list(range(len(te)))
    if per_record_cap and per_record_cap > 0 and hasattr(te, "record_ids"):
        rng = np.random.RandomState(eval_seed)
        by_rec = {}
        for i in idx:
            by_rec.setdefault(te.record_ids[te.index[i][0]], []).append(i)
        idx = []
        for rec in sorted(by_rec):
            w = by_rec[rec]
            if len(w) > per_record_cap:
                w = list(rng.choice(w, per_record_cap, replace=False))
            idx.extend(w)
    if shuffle:
        idx = list(np.random.RandomState(eval_seed + 1).permutation(idx))
    return idx[:max_windows]


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in known})
    from src.model import build_model
    model, _ = build_model(ck["kind"], cfg, ck["n_channels"], device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, np.asarray(ck["mean"], np.float32), \
        np.asarray(ck["std"], np.float32)


def classical_impute(x: np.ndarray, mask: np.ndarray, method: str,
                     fill: np.ndarray = None) -> np.ndarray:
    out = x.copy()
    L = x.shape[1]
    t = np.arange(L)
    fill = np.zeros(x.shape[0]) if fill is None else np.asarray(fill, float)
    for c in range(x.shape[0]):
        obs = mask[c] == 1
        if obs.all():
            continue
        if obs.sum() < 2:
            out[c][~obs] = fill[c]
            continue
        if method == "mean":
            out[c][~obs] = x[c][obs].mean()
        elif method == "locf":
            last = np.maximum.accumulate(np.where(obs, t, -1))
            out[c][~obs] = x[c][np.clip(last[~obs], 0, None)]
        elif method == "linear":
            out[c][~obs] = np.interp(t[~obs], t[obs], x[c][obs])
        elif method == "spline":
            if obs.sum() >= 4:
                out[c][~obs] = CubicSpline(t[obs], x[c][obs])(t[~obs])
            else:
                out[c][~obs] = np.interp(t[~obs], t[obs], x[c][obs])
    return out


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, mean, std = load_model(args.ckpt, device)
    C, L = model.C, cfg.window
    if mean.size != C:
        mean = np.full(C, float(mean.ravel()[0]), np.float32)
        std = np.full(C, float(std.ravel()[0]), np.float32)
    if not args.synthetic:
        cfg.data_root = resolve_data_root(cfg, args.data_root)
    cfg2 = Config(data_root=cfg.data_root, modality=cfg.modality,
                  dataset=cfg.dataset, fs=cfg.fs,
                  window=cfg.window, stride=cfg.stride, split=cfg.split,
                  seed=cfg.seed, cache_dir=cfg.cache_dir)
    _, _, te, _ = build_datasets(cfg2, synthetic=args.synthetic)
    protos = protocols_for(C)

    def sample_with(mdl, batch, mask, n):
        with torch.no_grad():
            s = mdl.impute(torch.from_numpy(batch).to(device),
                           torch.from_numpy(mask)[None].to(device),
                           n_samples=n)
        return s.cpu().numpy() * std[None, None, :, None] + \
            mean[None, None, :, None]

    prob_methods = {"CPGD-Cardio": lambda b, m, n: sample_with(model, b, m, n)}
    if args.csdi_ckpt and os.path.exists(args.csdi_ckpt):
        csdi_model, _, _, _ = load_model(args.csdi_ckpt, device)
        prob_methods["CSDI"] = lambda b, m, n: sample_with(csdi_model, b, m, n)
    if args.gru_ckpt and os.path.exists(args.gru_ckpt):
        gru_model, _, _, _ = load_model(args.gru_ckpt, device)
        prob_methods["GRU"] = lambda b, m, n: sample_with(gru_model, b, m, n)

    rows, n_seen = [], 0
    t0 = time.time()
    sel = select_eval_windows(te, args.max_windows, args.shuffle,
                              args.eval_seed, args.per_record_cap)
    for bi in range(0, len(sel), args.batch_size):
        ids = sel[bi:bi + args.batch_size]
        xb = np.stack([te[i].numpy() for i in ids])
        for b in range(xb.shape[0]):
            wid = ids[b]
            rec = te.record_ids[te.index[wid][0]] \
                if hasattr(te, "record_ids") else ""
            x = xb[b] * std[:, None] + mean[:, None]
            xn = xb[b]
            for proto in protos:
                mask = eval_mask(proto, L, C, window_id=wid, fs=cfg.fs)
                miss = mask == 0
                if miss.sum() == 0:
                    continue
                torch.manual_seed(10007 + wid)
                torch.cuda.manual_seed_all(10007 + wid)
                for name, fn in prob_methods.items():
                    s = fn(xb[b:b + 1], mask, args.n_samples)
                    pred = s.mean(0)[0]
                    sq, ab = masked_errors(pred, x, 1.0 - mask)
                    sq_n, ab_n = masked_errors(pred, xn, 1.0 - mask)
                    cr = crps_empirical(s[:, 0], x)
                    cov = interval_coverage(s[:, 0], x)
                    rows.append(dict(window=wid, record=rec, protocol=proto,
                                     method=name, mse=float(sq.mean()),
                                     mae=float(ab.mean()),
                                     mse_norm=float(sq_n.mean()),
                                     mae_norm=float(ab_n.mean()),
                                     crps=float(cr[miss].mean()),
                                     cov90=float(cov[miss].mean())))
                for name in CLASSICAL:
                    pred = classical_impute(x, mask, name, fill=mean)
                    sq, ab = masked_errors(pred, x, 1.0 - mask)
                    sq_n, ab_n = masked_errors((pred - mean[:, None]) / std[:, None],
                                               xn, 1.0 - mask)
                    rows.append(dict(window=wid, record=rec, protocol=proto,
                                     method=name, mse=float(sq.mean()),
                                     mae=float(ab.mean()),
                                     mse_norm=float(sq_n.mean()),
                                     mae_norm=float(ab_n.mean()),
                                     crps=np.nan, cov90=np.nan))
        n_seen += xb.shape[0]
        if n_seen % (args.batch_size * 10) < args.batch_size:
            print(f"  {n_seen} windows ... {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = args.out or \
        f"results/eval_{cfg.dataset}_{cfg.modality}_s{cfg.seed}.csv"
    df.to_csv(out, index=False)

    print(f"\n== Summary ({n_seen} test windows, {len(protos)} protocols) ==")
    summ = df.groupby(["protocol", "method"]).agg(
        mse_n=("mse_norm", "mean"), mae_n=("mae_norm", "mean"),
        crps=("crps", "mean"), cov90=("cov90", "mean")).reset_index()
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sig_rows = []
    for proto in protos:
        base = df[(df.protocol == proto) & (df.method == "CPGD-Cardio")] \
            .set_index("window")["mse_norm"]
        for m in df.method.unique():
            if m == "CPGD-Cardio":
                continue
            other = df[(df.protocol == proto) & (df.method == m)] \
                .set_index("window")["mse_norm"]
            common = base.index.intersection(other.index)
            _, p = paired_wilcoxon(base.loc[common].values,
                                   other.loc[common].values)
            sig_rows.append(dict(
                protocol=proto, baseline=m,
                mean_diff=float((base.loc[common] - other.loc[common]).mean()),
                p=p, sig=fmt_p(p)))
    sig = pd.DataFrame(sig_rows)
    sig_out = out.replace(".csv", "_sig.csv")
    sig.to_csv(sig_out, index=False)
    print("\n== Paired Wilcoxon (CPGD-Cardio minus baseline, norm. MSE) ==")
    print(sig.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nsaved: {out}\nsaved: {sig_out}")


if __name__ == "__main__":
    main()
