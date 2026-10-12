import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise RuntimeError(f"FAILED: {' '.join(cmd)}")


def main():
    for model in ("cpgd", "csdi", "gru"):
        run([PY, "train.py", "--model", model, "--synthetic", "--tiny",
             "--seed", "42"])
    run([PY, "evaluate.py",
         "--ckpt", os.path.join(HERE, "results/ckpt/cpgd_sens_ecg_s42_syn.pt"),
         "--csdi_ckpt", os.path.join(HERE, "results/ckpt/csdi_sens_ecg_s42_syn.pt"),
         "--gru_ckpt", os.path.join(HERE, "results/ckpt/gru_sens_ecg_s42_syn.pt"),
         "--synthetic", "--n_samples", "4", "--max_windows", "16",
         "--out", os.path.join(HERE, "results/eval_smoke.csv")])

    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, "results/eval_smoke.csv"))
    assert len(df) > 0, "no evaluation rows produced"
    num = df[["mse", "mae", "crps", "cov90"]].dropna()
    assert np_like_finite(num.values), "non-finite metric values found"
    print("\nSMOKE TEST PASSED -- pipeline is runnable end to end.")
    print(df.groupby(["protocol", "method"])[["mse", "mae", "crps", "cov90"]]
          .mean().to_string(float_format=lambda v: f"{v:.4f}"))


def np_like_finite(a):
    import numpy as np
    return bool(np.isfinite(np.asarray(a, dtype=float)).all())


if __name__ == "__main__":
    main()
