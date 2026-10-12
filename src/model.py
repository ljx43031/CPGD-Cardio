import math

import numpy as np
import torch
import torch.nn as nn

from .csdi_baseline import CSDI
from .diffusion import GaussianDiffusion
from .prior import CausalPrior, decomposition_channels
from .unet1d import UNet1D, timestep_embedding


def bandpass_torch(x: torch.Tensor, fs: float,
                   lo: float = 0.5, hi: float = 3.0) -> torch.Tensor:
    L = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(L, 1.0 / fs).to(x.device)
    keep = ((freqs >= lo) & (freqs <= hi)).float()
    return torch.fft.irfft(X * keep, n=L, dim=-1)


def masked_autocorr_period(bp: torch.Tensor, w: torch.Tensor, fs: float):
    B, C, L = bp.shape
    nfft = 2 * L
    r = torch.fft.rfft(bp * w, n=nfft, dim=-1)
    ac = torch.fft.irfft(r * torch.conj(r), n=nfft, dim=-1)[..., :L].real
    cnt = torch.fft.irfft(torch.fft.rfft(w, n=nfft, dim=-1)
                          * torch.conj(torch.fft.rfft(w, n=nfft, dim=-1)),
                          n=nfft, dim=-1)[..., :L].real
    var = (w * bp * bp).sum(-1) / w.sum(-1).clamp(min=1.0)
    ac = ac / (cnt + 1.0) / (var[..., None] + 1e-8)
    lo = int(round(fs / 3.0))
    hi = min(int(round(fs / 0.5)), L - 2)
    window = torch.zeros_like(ac)
    window[:, :, lo:hi] = 1.0
    ac_m = ac * window - 1e9 * (1.0 - window)
    gmax = ac_m.max(dim=-1, keepdim=True).values.clamp(min=0.0)
    thr = 0.8 * gmax
    local_max = ((ac >= torch.roll(ac, 1, dims=-1))
                 & (ac >= torch.roll(ac, -1, dims=-1))
                 & (window > 0) & (ac >= thr)).float()
    has_peak = local_max.sum(dim=-1) > 0
    idx = torch.where(has_peak, local_max.argmax(dim=-1),
                      ac_m.argmax(dim=-1))
    idx_c = idx.clamp(1, L - 2).long()
    lft = torch.gather(ac, -1, (idx_c - 1)[..., None])
    mid = torch.gather(ac, -1, idx_c[..., None])
    rgt = torch.gather(ac, -1, (idx_c + 1)[..., None])
    den = lft - 2 * mid + rgt
    delta = torch.where(den.abs() > 1e-9, 0.5 * (lft - rgt) / den,
                        torch.zeros_like(den)).clamp(-0.5, 0.5)
    period = idx_c.float() + delta.squeeze(-1)
    prom = mid.squeeze(-1)
    return period, prom


def template_impose(x_obs: torch.Tensor, mask: torch.Tensor,
                    period: torch.Tensor, prom: torch.Tensor):
    B, C, L = x_obs.shape
    gate = (prom > 0.2).float()
    NB = 64
    dev = x_obs.device

    t = torch.arange(L, device=dev, dtype=x_obs.dtype)[None, None, :]
    P = period.clamp(min=4.0)

    phase = torch.remainder(t, P[..., None])
    pos = phase / P[..., None] * NB
    j = pos.floor().long().clamp(0, NB - 1)
    owner = (torch.arange(B * C, device=dev).view(B, C, 1)
             .expand(B, C, L).reshape(-1))
    jj = j.reshape(-1)
    w = mask.reshape(-1)
    ssum = torch.zeros(B * C * NB, device=dev, dtype=x_obs.dtype)
    cnt = torch.zeros(B * C * NB, device=dev, dtype=x_obs.dtype)
    ssum.index_add_(0, owner * NB + jj, (x_obs * mask).reshape(-1) * w)
    cnt.index_add_(0, owner * NB + jj, w)
    tmpl = (ssum / cnt.clamp(min=1.0)).reshape(B * C, NB)
    have = (cnt > 0).reshape(B * C, NB)

    for _ in range(4):
        val_l = torch.where(have, tmpl, torch.zeros_like(tmpl))
        acc = (torch.roll(val_l, 1, dims=1) + torch.roll(val_l, -1, dims=1))
        nbn = (torch.roll(have.float(), 1, dims=1)
               + torch.roll(have.float(), -1, dims=1)).clamp(min=1.0)
        cand = acc / nbn
        tmpl = torch.where(have, tmpl, cand)
        have = have | (nbn > 0)
    tmpl = tmpl.reshape(B, C, NB)

    p0 = pos.floor().long().clamp(0, NB - 1)
    p1 = (p0 + 1) % NB
    frac = (pos - pos.floor()).clamp(0, 1)
    x_struct = (tmpl.gather(-1, p0) * (1 - frac)
                + tmpl.gather(-1, p1) * frac)
    return x_struct * gate[..., None], gate


def _detect(sig: np.ndarray, obs: np.ndarray, fs: int = 100):
    from scipy.signal import find_peaks
    idx = np.where(obs)[0]
    if len(idx) < 50:
        return np.array([], int)
    filled = np.interp(np.arange(len(sig)), idx, sig[idx])
    X = np.fft.rfft(filled)
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    X[(f < 5) | (f > 25)] = 0
    s = np.fft.irfft(X, n=len(sig)) ** 2
    mwi = np.convolve(s, np.ones(8) / 8.0, mode="same")
    thr = 0.25 * np.percentile(mwi[obs], 95)
    pk = find_peaks(mwi, height=thr, distance=25)[0]
    ref = []
    for p in pk:
        w0, w1 = max(p - 8, 0), min(p + 9, len(sig))
        ref.append(w0 + int(np.argmax(np.abs(sig[w0:w1]))))
    pk = np.array(sorted(set(ref)))
    return pk[obs[pk]]


def _longest_gap(obs):
    best = None
    run = None
    for i, o in enumerate(obs):
        if not o:
            if run is None:
                run = i
        else:
            if run is not None:
                if best is None or i - run > best[1] - best[0]:
                    best = (run, i)
                run = None
    if run is not None:
        if best is None or len(obs) - run > best[1] - best[0]:
            best = (run, len(obs))
    return best


def _gen_times(pre, post, g0, g1, rng):
    ipre, ipost = np.diff(pre), np.diff(post)
    if len(ipre) < 2 or len(ipost) < 2:
        return None
    span = max(g1 - g0, 1)
    pool = np.concatenate([ipre, ipost])
    q = np.quantile(pool, [1 / 3, 2 / 3])
    st = np.digitize(pool, q)
    seq = np.digitize(np.concatenate([ipre, ipost]), q)
    T = np.ones((3, 3)) * 0.5
    for a, b in zip(seq[:-1], seq[1:]):
        T[a, b] += 1
    irr = float(np.median(np.abs(ipre - np.median(ipre)))) / \
        max(np.median(ipre), 1e-9)
    strat = "markov" if irr >= 0.12 else "interp"

    def step(prev_rr, frac):
        if strat == "interp":
            rs, re_ = np.median(ipre[-3:]), np.median(ipost[:3])
            return rs + (re_ - rs) * frac
        s = 0 if prev_rr <= q[0] else (1 if prev_rr <= q[1] else 2)
        cand = pool[st == s]
        return float(rng.choice(cand)) if len(cand) else float(rng.choice(pool))

    fw, t, pr = [], pre[-1], float(ipre[-1])
    while True:
        r = step(pr, min(max((t - g0) / span, 0), 1)); pr = r
        t += r
        if t >= g1 - 10: break
        fw.append(t)
    bw, t, pr = [], post[0], float(ipost[0])
    while True:
        r = step(pr, min(max((post[0] - t) / span, 0), 1)); pr = r
        t -= r
        if t <= g0 + 10: break
        bw.append(t)
    t = np.array(sorted(fw + bw))
    rr_med = max(float(np.median(pool)), 20.0)
    keep = np.ones(len(t), bool)
    for i, dd in enumerate(np.diff(t)):
        if dd < 0.45 * rr_med:
            keep[i + 1] = False
    return t[keep]


def transplant_window(sig: np.ndarray, obs: np.ndarray, fs: int = 100):
    L = len(sig)
    gap = _longest_gap(obs)
    if gap is None:
        return None
    g0, g1 = gap
    pk = _detect(sig, obs, fs)
    pre, post = pk[pk < g0], pk[pk >= g1]
    pre, post = pre[-6:], post[:6]
    if len(pre) < 3 or len(post) < 3:
        return None
    rng = np.random.RandomState(7)
    times = _gen_times(pre, post, g0, g1, rng)
    if times is None or len(times) == 0:
        return None
    out = np.interp(np.arange(L), np.where(obs)[0], sig[obs])
    rr_med = max(float(np.median(np.diff(pk))), 20.0)
    for t_f in times:
        t = int(round(t_f))
        placed = False
        for j in np.argsort(np.abs(pk - t)):
            p = int(pk[j])
            a0, a1 = p - int(round(0.45 * rr_med)), p + int(round(0.55 * rr_med))
            if a0 < 0 or a1 > L or not obs[a0:a1].all():
                continue
            src = sig[a0:a1]
            d0, d1 = max(t - int(0.45 * rr_med), g0), \
                min(t + int(0.55 * rr_med), g1)
            if d1 - d0 <= 0:
                continue
            s0 = (p - a0) + (d0 - t)
            s0 = max(s0, 0)
            seg = src[s0:s0 + (d1 - d0)]
            if len(seg) != d1 - d0:
                continue
            out[d0:d1] = seg
            placed = True
            break
        if not placed:
            return None
    return out


class CPGDCardio(nn.Module):
    def __init__(self, cfg, n_channels: int, diffusion: GaussianDiffusion):
        super().__init__()
        self.cfg = cfg
        self.C = n_channels
        self.diff = diffusion
        self.residual = True
        self.n_harm = int(getattr(cfg, "n_harmonics", 6))
        self.prior = CausalPrior(n_channels, cfg.gru_hidden,
                                 cfg.gru_layers, cfg.feat_dim)
        self.time_dim = 128
        K = 3 * n_channels
        in_ch = 3 * n_channels + K
        cond_dim = self.time_dim + cfg.feat_dim
        self.eps_net = UNet1D(in_ch, out_ch=n_channels, base=cfg.base_dim,
                              mults=tuple(cfg.dim_mults), cond_dim=cond_dim)

    def _impose_template(self, x_obs: torch.Tensor, mask: torch.Tensor):
        bp = bandpass_torch(x_obs, self.cfg.fs)
        period, prom = masked_autocorr_period(bp, mask, self.cfg.fs)
        x_struct, gate = template_impose(x_obs, mask, period, prom)
        return x_struct, gate, period, prom

    def impose(self, x_obs: torch.Tensor, mask: torch.Tensor):
        B, C, L = x_obs.shape
        dev = x_obs.device
        if getattr(self.cfg, "no_impose", False):
            zeros = torch.zeros_like(x_obs)
            gate = torch.ones(B, C, device=dev)
            per = torch.full((B, C), float(self.cfg.period_samples), device=dev)
            return zeros, gate, per, gate
        out = torch.empty_like(x_obs)
        ok = torch.zeros(B, C, dtype=torch.bool)
        for b in range(B):
            for c in range(C):
                sig = x_obs[b, c].detach().cpu().numpy()
                msk = mask[b, c].detach().cpu().numpy().astype(bool)
                r = transplant_window(sig, msk)
                if r is None:
                    continue
                out[b, c] = torch.from_numpy(r.astype(np.float32))
                ok[b, c] = True
        if not bool(ok.all()):
            fb = self._impose_template(x_obs, mask)
            out[~ok] = fb[0][~ok]
        gate = torch.ones(B, C, device=dev)
        period = torch.full((B, C), float(self.cfg.period_samples), device=dev)
        prom = torch.ones(B, C, device=dev)
        return out, gate, period, prom

    def _point(self, x: torch.Tensor, mask: torch.Tensor):
        x_obs = x * mask
        x_struct, gate, period, prom = self.impose(x_obs, mask)
        comp_in = x_obs - x_struct * mask
        inp = torch.cat([comp_in, mask], dim=1)
        h, _ = self.prior.gru(inp.transpose(1, 2))
        comp = self.prior.head(h).transpose(1, 2)
        if getattr(self.cfg, "no_comp", False):
            comp = torch.zeros_like(comp)
        f = self.prior.feat(h[:, -1])
        point = x_struct * gate[..., None] + comp
        missing = 1.0 - mask
        aux = ((point - x) ** 2 * missing).sum() / missing.sum().clamp(min=1.0)
        return point, f, aux

    def _build_condition(self, x: torch.Tensor, mask: torch.Tensor,
                         point: torch.Tensor):
        x_obs = x * mask
        cond = torch.cat([point, mask], dim=1)
        trend, seasonal, skeleton = decomposition_channels(
            point, self.cfg.period_samples, self.cfg.skeleton_topk)
        cond = torch.cat([cond, trend, seasonal, skeleton], dim=1)
        return cond, x_obs

    def _cond_vec(self, t: torch.Tensor, f):
        cv = timestep_embedding(t, self.time_dim)
        return torch.cat([cv, f], dim=-1)

    def forward_loss(self, x: torch.Tensor, mask: torch.Tensor,
                     t: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        if t is None:
            t = torch.randint(0, self.diff.T, (B,), device=x.device)
        point, f, aux = self._point(x, mask)
        target = x - point.detach()
        noise = torch.randn_like(target)
        x_t = self.diff.q_sample(target, t, noise)
        cond, _ = self._build_condition(x, mask, point)
        eps_hat = self.eps_net(torch.cat([x_t, cond], dim=1),
                               self._cond_vec(t, f))
        loss = self.diff.loss(eps_hat, noise, missing=1.0 - mask)
        sa = self.diff.sqrt_ac[t][:, None, None]
        so = self.diff.sqrt_om[t][:, None, None]
        x0_hat = point.detach() + (x_t - so * eps_hat) / sa
        lam = float(getattr(self.cfg, "spec_weight", 0.5))
        if lam > 0:
            bp_hat = bandpass_torch(x0_hat, self.cfg.fs)
            bp_true = bandpass_torch(x, self.cfg.fs)
            missing = 1.0 - mask
            loss = loss + lam * (((bp_hat - bp_true) ** 2) * missing).sum() \
                / missing.sum().clamp(min=1.0)
        return loss + aux

    @torch.no_grad()
    def impute(self, x: torch.Tensor, mask: torch.Tensor, n_samples: int = 16,
               chunk: int = 4) -> torch.Tensor:
        point, f, _ = self._point(x, mask)
        B, C, L = x.shape
        cond, _ = self._build_condition(x, mask, point)
        outs, done = [], 0
        while done < n_samples:
            m = min(chunk, n_samples - done)

            def eps_fn(x_t, t, cond=cond, f=f):
                cv = self._cond_vec(t, f)
                return self.eps_net(torch.cat([x_t, cond], dim=1), cv)

            outs.append(self.diff.ddim_sample(
                eps_fn, (B, C, L), self.cfg.ddim_steps, n_samples=m))
            done += m
        samples = point[None] + torch.cat(outs, dim=0)
        return torch.where(mask[None] > 0, x[None], samples)


class GRUImputer(nn.Module):
    def __init__(self, cfg, n_channels: int, diffusion: GaussianDiffusion = None):
        super().__init__()
        self.gru = nn.GRU(2 * n_channels, cfg.gru_hidden, cfg.gru_layers,
                          batch_first=True)
        self.head = nn.Sequential(nn.Linear(cfg.gru_hidden, cfg.gru_hidden),
                                  nn.SiLU(), nn.Linear(cfg.gru_hidden, n_channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x * mask, mask], dim=1).transpose(1, 2)
        h, _ = self.gru(inp)
        return self.head(h).transpose(1, 2)

    def forward_loss(self, x: torch.Tensor, mask: torch.Tensor,
                     t=None) -> torch.Tensor:
        pred = self.forward(x, mask)
        missing = 1.0 - mask
        return ((pred - x) ** 2 * missing).sum() / missing.sum().clamp(min=1.0)

    @torch.no_grad()
    def impute(self, x: torch.Tensor, mask: torch.Tensor, n_samples: int = 1,
               chunk: int = 1) -> torch.Tensor:
        pred = self.forward(x, mask)
        return pred[None].repeat(n_samples, 1, 1, 1)


def build_model(kind: str, cfg, n_channels: int, device: str):
    diff = GaussianDiffusion(cfg.diff_T, device, cfg.clip_x0)
    if kind == "cpgd":
        model = CPGDCardio(cfg, n_channels, diff)
    elif kind == "predecessor":
        from .model_predecessor import CPGDCardioPredecessor
        model = CPGDCardioPredecessor(cfg, n_channels, diff)
    elif kind == "csdi":
        model = CSDI(cfg, n_channels, diff)
    elif kind == "gru":
        model = GRUImputer(cfg, n_channels)
    else:
        raise ValueError(kind)
    return model.to(device), diff


def freeze_prior(model: nn.Module, frozen: bool):
    if hasattr(model, "prior"):
        for p in model.prior.parameters():
            p.requires_grad = not frozen
