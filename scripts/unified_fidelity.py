import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from scipy.signal import find_peaks

from config import Config
from src.dataset import build_datasets
from src.masking import eval_mask
from evaluate import classical_impute, load_model, select_eval_windows

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FS, MATCH = 100, 15
PEAK_H, DIST = 1.2, 25


def band(x, lo=0.5, hi=3.0):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=len(x))


def detect_true_beats(true_gap):
    v = band(true_gap)
    if v.std() < 1e-9:
        return np.array([], int)
    pk = find_peaks(v, height=PEAK_H * v.std(), distance=DIST)[0]
    return pk


def gap_eval(rec_gap, true_gap):
    pk_t = detect_true_beats(true_gap)
    if len(pk_t) < 2:
        return None
    pk_r = detect_true_beats(rec_gap)
    errs, amps = [], []
    for pt in pk_t:
        if len(pk_r) == 0:
            errs.append(MATCH + 1)
            amps.append(0.0)
            continue
        d = np.min(np.abs(pk_r - pt))
        if d <= MATCH:
            errs.append(float(d))
            amps.append(abs(rec_gap[pk_r[np.argmin(np.abs(pk_r - pt))]]) /
                        max(abs(true_gap[pt]), 1e-9))
        else:
            errs.append(float(MATCH + 1))
            amps.append(0.0)
    pr = np.sqrt((band(rec_gap) ** 2).mean())
    pt_pow = np.sqrt((band(true_gap) ** 2).mean())
    band_ratio = pr / max(pt_pow, 1e-9)
    a = rec_gap - rec_gap.mean()
    b = true_gap - true_gap.mean()
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else np.nan
    med_err = float(np.median(errs))
    hit_rate = float(np.mean([e <= MATCH for e in errs]))
    return len(pk_t), med_err, hit_rate, float(np.mean(amps)), corr, band_ratio


def run(label, root, ds, stride, v7ck, csdick, gruck, max_windows, cap):
    model, c, _, _ = load_model(v7ck, DEV)
    csdi, _, _, _ = load_model(csdick, DEV)
    gru, _, _, _ = load_model(gruck, DEV)
    cfg = Config(dataset=ds, data_root=root, modality="ecg", fs=100,
                 window=960, stride=stride, split=c.split, seed=c.seed,
                 cache_dir=c.cache_dir)
    _, _, te, _ = build_datasets(cfg)
    C, L = te.n_channels, 960
    methods = ["CPGD-Cardio", "GRU", "CSDI", "linear"]
    acc = {m: dict(n_true=[], med=[], hit=[], amp=[], band=[], corr=[])
           for m in methods}
    sel = select_eval_windows(te, max_windows, shuffle=True, eval_seed=0,
                              per_record_cap=cap)
    n_win = 0
    for wi in sel:
        x = te[wi].numpy()
        mask = eval_mask("blackout20", L, C, window_id=wi, fs=FS)
        xt = torch.from_numpy(x)[None].to(DEV)
        for ch in range(C):
            g = mask[ch] == 0
            if g.sum() < 60:
                continue
            true_gap = x[ch][g]
            pk_t = detect_true_beats(true_gap)
            if len(pk_t) < 2:
                continue
            mt = torch.from_numpy(mask)[None].to(DEV)
            outs = {}
            torch.manual_seed(5)
            torch.cuda.manual_seed_all(5)
            with torch.no_grad():
                outs["CPGD-Cardio"] = model.impute(
                    xt, mt, n_samples=16).mean(0)[0].cpu().numpy()[ch][g]
                outs["GRU"] = gru.impute(xt, mt, n_samples=1
                                         )[0, 0].cpu().numpy()[ch][g]
                outs["CSDI"] = csdi.impute(xt, mt, n_samples=16
                                           ).mean(0)[0].cpu().numpy()[ch][g]
            outs["linear"] = classical_impute(x, mask, "linear")[ch][g]
            for m, rg in outs.items():
                r = gap_eval(rg, true_gap)
                if r is None:
                    continue
                n_true, med, hit, amp, corr, band_r = r
                a = acc[m]
                a["n_true"].append(n_true)
                a["med"].append(med)
                a["hit"].append(hit)
                a["amp"].append(amp)
                a["band"].append(band_r)
                a["corr"].append(corr)
        n_win += 1
    print(f"\n==== {label} ({len(acc['linear']['med'])} gaps, "
          f"{n_win} windows) ====")
    print(f"{'method':13s}{'timing(med)':>12s}{'hit@150ms':>11s}"
          f"{'amp ratio':>10s}{'band':>8s}{'wave corr':>10s}")
    for m in methods:
        a = acc[m]
        f = lambda k: np.mean(a[k]) if len(a[k]) else float('nan')
        med = np.median(a['amp']) if len(a['amp']) else float('nan')
        print(f"{m:13s}{f('med'):>12.1f}{100*f('hit'):>10.0f}%"
              f"{med:>10.2f}{f('band'):>8.2f}{f('corr'):>10.2f}")
    return label, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_ckpt", default="results/ckpt/cpgd_sens_ecg_s42.pt")
    ap.add_argument("--sens_csdi", default="results/ckpt/csdi_sens_ecg_s42.pt")
    ap.add_argument("--sens_gru", default="results/ckpt/gru_sens_ecg_s42.pt")
    ap.add_argument("--mitdb_ckpt", default="results/ckpt/cpgd_mitdb_ecg_s42.pt")
    ap.add_argument("--mitdb_csdi", default="results/ckpt/csdi_mitdb_ecg_s42.pt")
    ap.add_argument("--mitdb_gru", default="results/ckpt/gru_mitdb_ecg_s42.pt")
    ap.add_argument("--max_windows", type=int, default=40)
    ap.add_argument("--per_record_cap", type=int, default=5)
    ap.add_argument("--out", default="results/tables/unified_fidelity.json")
    args = ap.parse_args()

    out = {}
    for label, acc in (
            run("SensSmartTech ECG", Config().data_root, "sens", 480,
                args.sens_ckpt, args.sens_csdi, args.sens_gru,
                args.max_windows, args.per_record_cap),
            run("MIT-BIH", os.path.join("data", "physionet", "mitdb"), "mitdb",
                960, args.mitdb_ckpt, args.mitdb_csdi, args.mitdb_gru,
                args.max_windows, args.per_record_cap)):
        out[label] = {m: {k: [None if v != v else round(float(v), 4)
                              for v in vals]
                          for k, vals in a.items()}
                      for m, a in acc.items()}
    summ = {}
    for label, acc in out.items():
        summ[label] = {}
        for m, a in acc.items():
            nz = lambda k: [v for v in a[k] if v is not None]
            summ[label][m] = {
                "timing_err_samples_mean_of_gap_medians":
                    round(float(np.mean(nz("med"))), 3) if nz("med") else None,
                "hit_rate_mean": round(float(np.mean(nz("hit"))), 4)
                    if nz("hit") else None,
                "amp_ratio_median_of_gap_means":
                    round(float(np.median(nz("amp"))), 4) if nz("amp") else None,
                "band_rms_ratio_mean": round(float(np.mean(nz("band"))), 4)
                    if nz("band") else None,
                "wave_corr_mean": round(float(np.mean(nz("corr"))), 4)
                    if nz("corr") else None,
                "n_gaps": len(a["med"]),
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fj:
        json.dump({"per_gap": out, "summary": summ}, fj,
                  ensure_ascii=False, indent=1)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
