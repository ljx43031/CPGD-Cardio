import numpy as np
import torch


def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ac = f / f[0]
    return ac[1:].clamp(1e-5, 0.9999).float()


class GaussianDiffusion:
    def __init__(self, T: int, device: str = "cuda", clip_x0: float = 4.0):
        self.T = T
        self.clip_x0 = clip_x0
        self.device = device
        ac = cosine_alphas_cumprod(T).to(device)
        self.alphas_cumprod = ac
        self.sqrt_ac = ac.sqrt()
        self.sqrt_om = (1.0 - ac).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        sa = self.sqrt_ac[t][:, None, None]
        so = self.sqrt_om[t][:, None, None]
        return sa * x0 + so * noise

    def loss(self, eps_hat: torch.Tensor, eps: torch.Tensor,
             missing: torch.Tensor) -> torch.Tensor:
        w = missing.sum().clamp(min=1.0)
        return ((eps_hat - eps) ** 2 * missing).sum() / w

    @torch.no_grad()
    def ddim_sample(self, eps_fn, shape, steps: int, n_samples: int = 1):
        B, C, L = shape[0], shape[1], shape[2]
        ts = np.linspace(0, self.T - 1, steps).round().astype(np.int64)
        ts = np.unique(ts)[::-1]
        out = torch.empty((n_samples, B, C, L), device=self.device)
        for s in range(n_samples):
            x = torch.randn(B, C, L, device=self.device)
            for i, t_ in enumerate(ts):
                t = torch.full((B,), int(t_), device=self.device, dtype=torch.long)
                eps = eps_fn(x, t)
                sa = self.sqrt_ac[t][:, None, None]
                so = self.sqrt_om[t][:, None, None]
                x0 = ((x - so * eps) / sa).clamp(-self.clip_x0, self.clip_x0)
                if i == len(ts) - 1:
                    x = x0
                else:
                    t_next = torch.full((B,), int(ts[i + 1]), device=self.device,
                                        dtype=torch.long)
                    sa_n = self.sqrt_ac[t_next][:, None, None]
                    so_n = self.sqrt_om[t_next][:, None, None]
                    x = sa_n * x0 + so_n * eps
            out[s] = x
        return out
