import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DEFAULT_DATA_ROOT

MODALITY_SUFFIX = {"ecg": "_ecg.csv", "acc": "_acc.csv",
                   "pcg": "_pcg.csv", "ppg": "_ppg.csv"}


def default_data_root(dataset: str) -> str:
    return (DEFAULT_DATA_ROOT if dataset == "sens"
            else os.path.join("data", "physionet", dataset))


def resolve_data_root(cfg, override: str = None) -> str:
    """Checkpoints can carry the training machine's data_root; fall back to
    the packaged default location when it does not exist here."""
    if override:
        return override
    if os.path.isdir(cfg.data_root):
        return cfg.data_root
    fb = default_data_root(getattr(cfg, "dataset", "sens"))
    print(f"[data] checkpoint data_root '{cfg.data_root}' not found on this "
          f"machine; using '{fb}'")
    return fb


def list_records(root: str, modality: str):
    files = sorted(glob.glob(os.path.join(root, "*" + MODALITY_SUFFIX[modality])))
    n_cut = len(MODALITY_SUFFIX[modality])
    return [(os.path.basename(f)[:-n_cut], f) for f in files]


def load_csv_signal(path: str):
    df = pd.read_csv(path)
    if "t" not in df.columns:
        df = df.rename(columns={df.columns[0]: "t"})
    t = df["t"].to_numpy(dtype=np.float64) / 1000.0
    vals = df.drop(columns=["t"]).to_numpy(dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(vals).all(axis=1)
    t, vals = t[finite], vals[finite]
    order = np.argsort(t)
    return t[order], vals[order].astype(np.float32)


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: int):
    t0, t1 = float(t[0]), float(t[-1])
    n = max(int(round((t1 - t0) * fs)) + 1, 4)
    grid = t0 + np.arange(n, dtype=np.float64) / fs
    out = np.empty((n, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        col = x[:, c]
        out[:, c] = np.interp(grid, t, col)
    return grid, out


def _split_subjects(subjects, split, seed, flag):
    subjects = sorted(subjects)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(subjects))
    subjects = [subjects[i] for i in perm]
    n_tr = int(len(subjects) * split[0])
    n_va = int(len(subjects) * split[1])
    if flag == "train":
        return set(subjects[:n_tr])
    if flag == "val":
        return set(subjects[n_tr:n_tr + n_va])
    return set(subjects[n_tr + n_va:])


class _WindowDataset(Dataset):
    def __init__(self, cfg, flag: str, cache_dir: str):
        assert flag in ("train", "val", "test")
        self.cfg, self.flag = cfg, flag
        os.makedirs(cache_dir, exist_ok=True)
        recs = self._load_records(cache_dir)
        if not recs:
            raise FileNotFoundError(f"{type(self).__name__}: no records found")
        keep = _split_subjects({s for s, _ in recs}, cfg.split, cfg.seed, flag)
        if getattr(cfg, "per_record_norm", False):
            recs = [(s, (sig - sig.mean(0)) / (sig.std(0) + 1e-6))
                    for s, sig in recs]
        self.stats = self._load_stats(cache_dir, recs, keep)

        self.signals, self.index, self.record_ids = [], [], []
        for subj, sig in recs:
            if subj not in keep:
                continue
            sig = (sig - self.stats["mean"]) / self.stats["std"]
            self.signals.append(sig.astype(np.float32))
            self.record_ids.append(subj)
            n = sig.shape[0]
            for start in range(0, max(n - cfg.window, 0) + 1, cfg.stride):
                self.index.append((len(self.signals) - 1, start))
        if not self.index:
            raise RuntimeError(f"{flag}: no windows produced")

    def _load_records(self, cache: str):
        raise NotImplementedError

    def _load_stats(self, cache: str, recs, keep):
        stats_path = os.path.join(cache, f"stats_sd{self.cfg.seed}.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                st = json.load(f)
            return {"mean": np.array(st["mean"], np.float32),
                    "std": np.clip(np.array(st["std"], np.float32), 1e-6, None)}
        if self.flag != "train":
            raise FileNotFoundError(
                f"{stats_path} missing -- build the train split first")
        big = np.concatenate([sig for subj, sig in recs if subj in keep], axis=0)
        mean = big.mean(0).astype(np.float32)
        std = np.clip(big.std(0), 1e-6, None).astype(np.float32)
        with open(stats_path, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
        return {"mean": mean, "std": std}

    @property
    def n_channels(self):
        return self.signals[0].shape[1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, start = self.index[i]
        x = self.signals[fi][start:start + self.cfg.window]
        return torch.from_numpy(x.T.copy())


class SensSmartTech(_WindowDataset):
    def __init__(self, cfg, flag: str = "train"):
        cache = os.path.join(cfg.cache_dir,
                             f"sens_{cfg.modality}_fs{cfg.fs}_w{cfg.window}_s{cfg.stride}")
        super().__init__(cfg, flag, cache)

    def _load_records(self, cache: str):
        recs = list_records(self.cfg.data_root, self.cfg.modality)
        out = []
        for subj, path in recs:
            true_subject = subj.split("_")[0]
            out.append((true_subject, self._load_resampled(path, cache)))
        return out

    def _load_resampled(self, path: str, cache: str) -> np.ndarray:
        stem = os.path.splitext(os.path.basename(path))[0]
        cached = os.path.join(cache, stem + ".npy")
        if os.path.exists(cached):
            return np.load(cached)
        t, x = load_csv_signal(path)
        _, sig = resample_uniform(t, x, self.cfg.fs)
        np.save(cached, sig)
        return sig


class SyntheticCardio(Dataset):
    def __init__(self, cfg, flag="train", length=512, n_channels=2):
        self.cfg, self.flag, self.length, self.n_channels = cfg, flag, length, n_channels

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.RandomState((self.cfg.seed * 100003 + i * 7919) & 0x7FFFFFFF)
        L, C, fs = self.cfg.window, self.n_channels, self.cfg.fs
        t = np.arange(L) / fs
        f0 = rng.uniform(0.8, 1.5)
        harm_w = 1.0 / np.arange(1, 6)
        phase = rng.uniform(0, 2 * np.pi, size=5)
        x = np.zeros((C, L), np.float32)
        for c in range(C):
            phi_c = c * rng.uniform(0.2, 0.5)
            base = sum(harm_w[k] * np.sin(2 * np.pi * (k + 1) * f0 * t + phase[k] + phi_c)
                       for k in range(5))
            trend = 0.7 * np.sin(2 * np.pi * rng.uniform(0.02, 0.1) * t + rng.uniform(0, 6.28))
            noise = rng.normal(0, 0.15, L).astype(np.float64)
            x[c] = (0.8 * base + trend + noise).astype(np.float32)
        return torch.from_numpy(x)


def build_datasets(cfg, synthetic=False):
    if synthetic:
        tr, va, te = (SyntheticCardio(cfg, f) for f in ("train", "val", "test"))
        return tr, va, te, {"n_channels": tr.n_channels,
                            "std": np.ones(1), "mean": np.zeros(1)}
    if cfg.dataset == "sens":
        cls = SensSmartTech
    else:
        from .wfdb_dataset import WfdbDataset
        cls = WfdbDataset
    tr, va, te = cls(cfg, "train"), cls(cfg, "val"), cls(cfg, "test")
    return tr, va, te, {"n_channels": tr.n_channels,
                        "std": tr.stats["std"], "mean": tr.stats["mean"]}
