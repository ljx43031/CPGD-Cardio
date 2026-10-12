import argparse
import os
import sys
import time
import urllib.request
from io import BytesIO

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import wfdb  # noqa: E402
import pandas as pd  # noqa: E402
from src.dataset import resample_uniform  # noqa: E402

DB_VERSION = {"mitdb": "1.0.0", "ptbxl": "1.0.3"}


def fetch_url(url: str, retries: int = 3) -> bytes:
    import subprocess
    data = b""
    for i in range(retries * 2):
        if i % 2 == 0:
            r = subprocess.run(["curl", "-s", "--max-time", "120", url],
                               capture_output=True)
            if r.returncode == 0 and r.stdout:
                return r.stdout
        else:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r2:
                    data = r2.read()
                if data:
                    return data
            except Exception as e:
                print(f"  url retry {i//2+1} ({e})", flush=True)
        time.sleep(3)
    return data


def read_records(db: str, attempts: int = 10) -> list:
    ver = DB_VERSION[db]
    url = f"https://physionet.org/files/{db}/{ver}/RECORDS"
    for i in range(attempts):
        lines = []
        try:
            lines = [l.strip() for l in fetch_url(url).decode().splitlines()
                     if l.strip()]
        except Exception as e:
            print(f"  RECORDS retry {i+1} ({e})", flush=True)
        if lines:
            return lines
        time.sleep(5)
    raise RuntimeError(f"RECORDS fetch failed after {attempts} attempts")


def ensure_ptbxl_csv(min_rows: int = 21800) -> pd.DataFrame:
    ver = DB_VERSION["ptbxl"]
    local = os.path.join(HERE, "data", "ptbxl_database.csv")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    url = f"https://physionet.org/files/ptbxl/{ver}/ptbxl_database.csv"
    import subprocess
    stuck = 0
    last_size = -1
    for attempt in range(300):
        if os.path.exists(local) and os.path.getsize(local) > 0:
            try:
                df = pd.read_csv(local)
                if len(df) >= min_rows and "filename_lr" in df.columns:
                    print(f"  csv usable: {len(df)} rows", flush=True)
                    return df
            except Exception:
                pass
            size = os.path.getsize(local)
            stuck = stuck + 1 if size == last_size else 0
            last_size = size
            if stuck >= 25:
                raise RuntimeError(
                    f"ptbxl_database.csv stalled at {size} bytes after "
                    f"{attempt+1} attempts; delete it and re-run")
        print(f"  csv attempt {attempt+1}: size="
              f"{os.path.getsize(local) if os.path.exists(local) else 0}",
              flush=True)
        subprocess.run(["curl", "-s", "-C", "-", "--max-time", "900",
                        "-o", local, url], check=False)
        time.sleep(3)
    raise RuntimeError("ptbxl_database.csv incomplete")


def ptbxl_patient_map() -> dict:
    df = ensure_ptbxl_csv()
    pcol = "patient_id" if "patient_id" in df.columns else "patientid"
    df = df.dropna(subset=["filename_lr", pcol])
    return dict(zip(df["filename_lr"], df[pcol].astype(str)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=["mitdb", "ptbxl"], required=True)
    ap.add_argument("--fs", type=int, default=100)
    ap.add_argument("--n_records", type=int, default=600,
                    help="ptbxl subset size (mitdb fetches all 48)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(HERE, "data", "physionet", args.db)
    os.makedirs(outdir, exist_ok=True)
    patients = None
    if args.db == "ptbxl":
        print("fetching ptbxl_database.csv (patient ids + record list) ...",
              flush=True)
        df = ensure_ptbxl_csv(min_rows=21000)
        pcol = "patient_id" if "patient_id" in df.columns else "patientid"
        df = df.dropna(subset=["filename_lr", pcol])
        patients = dict(zip(df["filename_lr"], df[pcol].astype(str)))
        sel = sorted(f for f in patients
                     if "records100" in f and f.endswith("_lr"))
        if len(sel) > args.n_records:
            idx = np.linspace(0, len(sel) - 1, args.n_records).round().astype(int)
            sel = [sel[i] for i in sorted(set(idx))]
    else:
        sel = read_records(args.db)
        print(f"{args.db}: {len(sel)} records in index")

    done = skipped = failed = 0
    t0 = time.time()
    for k, rec in enumerate(sel):
        subdir, base = os.path.split(rec)
        safe = base.replace("/", "_")
        out_path = os.path.join(outdir, safe + ".npz")
        if os.path.exists(out_path):
            skipped += 1
            continue
        pn_dir = f"{args.db}/{DB_VERSION[args.db]}" + (f"/{subdir}" if subdir else "")
        ok = False
        for attempt in range(3):
            try:
                if patients is not None and rec not in patients:
                    failed += 1
                    break
                sig, info = wfdb.rdsamp(base, pn_dir=pn_dir)
                sig = np.nan_to_num(np.asarray(sig, np.float32))
                if info["fs"] != args.fs:
                    t = np.arange(sig.shape[0]) / float(info["fs"])
                    _, sig = resample_uniform(t, sig, args.fs)
                patient = patients.get(rec, base)
                tmp = out_path + ".tmp.npz"
                np.savez_compressed(tmp, sig=sig.astype(np.float32),
                                    fs=np.int64(args.fs),
                                    patient=np.str_(patient),
                                    sig_name=np.array(info["sig_name"]))
                os.replace(tmp, out_path)
                done += 1
                ok = True
                break
            except Exception as e:
                if attempt == 2:
                    failed += 1
                    print(f"  FAIL {rec}: {e}", flush=True)
                else:
                    time.sleep(10)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(sel)}  done={done} skip={skipped} "
                  f"fail={failed}  {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.15)
    print(f"finished: done={done} skipped={skipped} failed={failed} "
          f"-> {outdir}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
