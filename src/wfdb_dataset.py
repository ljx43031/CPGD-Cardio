import glob
import os

import numpy as np

from .dataset import _WindowDataset, resample_uniform

MITDB_PATIENT_MAP = {"202": "201", "208": "207"}


class WfdbDataset(_WindowDataset):
    def __init__(self, cfg, flag: str = "train"):
        cache = os.path.join(
            cfg.cache_dir,
            f"{cfg.dataset}_fs{cfg.fs}_w{cfg.window}_s{cfg.stride}")
        super().__init__(cfg, flag, cache)

    def _load_records(self, cache: str):
        files = sorted(glob.glob(os.path.join(self.cfg.data_root, "*.npz")))
        out = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            sig, fs = z["sig"].astype(np.float32), int(z["fs"])
            patient = str(z["patient"]) if "patient" in z else \
                os.path.splitext(os.path.basename(f))[0]
            if self.cfg.dataset == "mitdb":
                patient = MITDB_PATIENT_MAP.get(patient, patient)
            if fs != self.cfg.fs and sig.shape[0] > 4:
                t = np.arange(sig.shape[0]) / fs
                _, sig = resample_uniform(t, sig, self.cfg.fs)
            out.append((patient, sig))
        return out
