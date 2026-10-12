import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(100) *
                      torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None] * 100.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(timestep_embedding(t, self.dim))


class CSDILayer(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.feat_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        B, L, C, d = h.shape
        ht = h.permute(0, 2, 1, 3).reshape(B * C, L, d)
        te = temb.repeat_interleave(C, dim=0)[:, None, :].expand(B * C, L, d)
        a, _ = self.time_attn(ht + te, ht, ht, need_weights=False)
        ht = self.n1(ht + a)
        hf = ht.reshape(B, C, L, d).permute(0, 2, 1, 3).reshape(B * L, C, d)
        te2 = temb.repeat_interleave(L, dim=0)[:, None, :].expand(B * L, C, d)
        a2, _ = self.feat_attn(hf + te2, hf, hf, need_weights=False)
        hf = self.n2(hf + a2)
        h = hf.reshape(B, L, C, d)
        return self.n3(h + self.ff(h))


class CSDI(nn.Module):
    def __init__(self, cfg, n_channels: int, diffusion):
        super().__init__()
        self.cfg = cfg
        self.diff = diffusion
        d = cfg.csdi_dim
        self.temb = DiffusionEmbedding(d)
        self.input_proj = nn.Linear(2, d)
        self.layers = nn.ModuleList([CSDILayer(d, cfg.csdi_heads)
                                     for _ in range(cfg.csdi_layers)])
        self.out = nn.Linear(d, 1)

    def _eps(self, x_t: torch.Tensor, mask: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
        B, C, L = x_t.shape
        temb = self.temb(t)
        feats = torch.stack([x_t.transpose(1, 2), mask.transpose(1, 2)], dim=-1)
        h = self.input_proj(feats) + temb[:, None, None, :]
        skips = []
        for layer in self.layers:
            h = layer(h, temb)
            skips.append(h)
        h = (h + torch.stack(skips, 0).sum(0)) / math.sqrt(len(skips))
        return self.out(h).squeeze(-1).transpose(1, 2)

    def forward_loss(self, x: torch.Tensor, mask: torch.Tensor,
                     t: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        if t is None:
            t = torch.randint(0, self.diff.T, (B,), device=x.device)
        noise = torch.randn_like(x)
        x_t = self.diff.q_sample(x, t, noise)
        eps_hat = self._eps(x_t, mask, t)
        return self.diff.loss(eps_hat, noise, missing=1.0 - mask)

    @torch.no_grad()
    def impute(self, x: torch.Tensor, mask: torch.Tensor, n_samples: int = 16,
               chunk: int = 4) -> torch.Tensor:
        B, C, L = x.shape
        outs, done = [], 0
        while done < n_samples:
            m = min(chunk, n_samples - done)

            def eps_fn(x_t, t, mask=mask, m=m):
                return self._eps(x_t, mask, t)

            outs.append(self.diff.ddim_sample(
                eps_fn, (B, C, L), self.cfg.ddim_steps, n_samples=m))
            done += m
        return torch.cat(outs, dim=0)
