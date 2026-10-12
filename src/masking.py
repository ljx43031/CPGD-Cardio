import numpy as np

EVAL_PROTOCOLS = ("point10", "seg5x20", "blackout20", "sensordrop")
PROTOCOL_IDS = {p: i for i, p in enumerate(EVAL_PROTOCOLS)}


def sample_train_mask(rng: np.random.RandomState, L: int, C: int, cfg) -> np.ndarray:
    M = np.ones((C, L), dtype=np.float32)
    if C > 1 and rng.rand() < cfg.channel_drop_prob:
        M[rng.randint(C)] = 0.0
    if rng.rand() < cfg.blackout_prob:
        for _ in range(int(rng.randint(1, cfg.max_segments + 1))):
            l = int(rng.randint(cfg.seg_min, cfg.seg_max + 1))
            s = int(rng.randint(0, max(L - l, 1)))
            M[:, s:s + l] = 0.0
    if cfg.point_prob > 0:
        M *= (rng.rand(C, L) > cfg.point_prob).astype(np.float32)
    return M


def eval_mask(protocol: str, L: int, C: int, window_id: int = 0,
              fs: int = 100) -> np.ndarray:
    rng = np.random.RandomState(
        (window_id * 1000003 + PROTOCOL_IDS[protocol] * 7919 + 17) & 0x7FFFFFFF)
    M = np.ones((C, L), dtype=np.float32)
    if protocol == "point10":
        idx = rng.rand(C, L) < 0.10
        M[idx] = 0.0
    elif protocol == "seg5x20":
        l = max(1, int(round(0.2 * fs)))
        for _ in range(5):
            s = int(rng.randint(0, max(L - l, 1)))
            M[:, s:s + l] = 0.0
    elif protocol == "blackout20":
        l = int(L * 0.2)
        s = int(rng.randint(0, max(L - l, 1)))
        M[:, s:s + l] = 0.0
    elif protocol == "sensordrop":
        if C < 2:
            raise ValueError("sensordrop needs >=2 channels")
        M[rng.randint(C)] = 0.0
    else:
        raise ValueError(f"unknown protocol {protocol}")
    return M


def protocols_for(C: int):
    ps = list(EVAL_PROTOCOLS)
    if C < 2:
        ps.remove("sensordrop")
    return ps
