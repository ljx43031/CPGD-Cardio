import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from config import Config
from evaluate import load_model, select_eval_windows
from src.dataset import build_datasets
from src.masking import eval_mask
from unified_fidelity import detect_true_beats, gap_eval

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FS, MATCH = 100, 15
PROTOS = ["blackout20", "seg5x20", "point10", "sensordrop"]


def mse_block(variants):
    full = pd.read_csv(variants[0][1]).query("method=='CPGD-Cardio'")
    rows = []
    for proto in PROTOS:
        f = full[full.protocol == proto].set_index("window")["mse_norm"]
        cells = {"full": f}
        for name, path in variants[1:]:
            a = pd.read_csv(path).query("method=='CPGD-Cardio'")
            cells[name] = a[a.protocol == proto].set_index("window")["mse_norm"]
        common = f.index
        for nm in cells:
            common = common.intersection(cells[nm].index)
        for name, path in variants:
            s = cells[name].loc[common]
            d = pd.read_csv(path).query(
                f"method=='CPGD-Cardio' & protocol=='{proto}'")
            d = d[d.window.isin(common)]
            crps = float(d.crps.mean())
            cov = float(d.cov90.mean())
            if name == "full":
                p = ""
            else:
                _, pv = wilcoxon(cells[name].loc[common].values,
                                 cells["full"].loc[common].values)
                p = "***" if pv < 0.001 else "**" if pv < 0.01 else \
                    "*" if pv < 0.05 else "ns"
            rows.append(dict(variant=name, protocol=proto,
                             mse=float(s.mean()), crps=crps, cov90=cov, p=p))
    return pd.DataFrame(rows)


def rhythm_block(ckpt_full, ckpt_noimpose, max_windows, cap):
    cfg = Config(modality="ecg")
    _, _, te, _ = build_datasets(cfg)
    C, L = te.n_channels, 960
    sel = select_eval_windows(te, max_windows, shuffle=True, eval_seed=0,
                              per_record_cap=cap)
    out = {}
    for name, ck in (("full", ckpt_full), ("w/o event implant", ckpt_noimpose)):
        model, _, _, _ = load_model(ck, DEV)
        meds, hits = [], []
        for wi in sel:
            x = te[wi].numpy()
            mask = eval_mask("blackout20", L, C, window_id=wi, fs=FS)
            xt = torch.from_numpy(x)[None].to(DEV)
            mt = torch.from_numpy(mask)[None].to(DEV)
            for ch in range(C):
                g = mask[ch] == 0
                if g.sum() < 60:
                    continue
                true_gap = x[ch][g]
                if len(detect_true_beats(true_gap)) < 2:
                    continue
                torch.manual_seed(5)
                torch.cuda.manual_seed_all(5)
                with torch.no_grad():
                    rec = model.impute(xt, mt, n_samples=16
                                       ).mean(0)[0].cpu().numpy()[ch][g]
                r = gap_eval(rec, true_gap)
                if r is None:
                    continue
                _, med, hit, _, _, _ = r
                meds.append(med)
                hits.append(hit)
        out[name] = dict(timing_samples=float(np.mean(meds)),
                         hit_rate=float(np.mean(hits)), n_gaps=len(meds))
        print(name, out[name], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_csv", default="results/eval_sens_ecg_s42.csv")
    ap.add_argument("--noimpose_csv",
                    default="results/eval_sens_ecg_s42_no_impose.csv")
    ap.add_argument("--nocomp_csv",
                    default="results/eval_sens_ecg_s42_no_comp.csv")
    ap.add_argument("--ckpt_full",
                    default="results/ckpt/cpgd_sens_ecg_s42.pt")
    ap.add_argument("--ckpt_noimpose",
                    default="results/ckpt/cpgd_sens_ecg_s42_no_impose.pt")
    ap.add_argument("--max_windows", type=int, default=40)
    ap.add_argument("--per_record_cap", type=int, default=5)
    ap.add_argument("--out", default="results/tables/ablation_v7.csv")
    args = ap.parse_args()

    variants = [("full", args.full_csv),
                ("w/o event implant", args.noimpose_csv),
                ("w/o GRU compensation", args.nocomp_csv)]
    df = mse_block(variants)
    rhythm = rhythm_block(args.ckpt_full, args.ckpt_noimpose,
                          args.max_windows, args.per_record_cap)
    for name, r in rhythm.items():
        label = "full" if name == "full" else name
        for k, v in r.items():
            df.loc[df.variant == label, k] = v
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
