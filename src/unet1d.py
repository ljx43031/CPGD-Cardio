import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _gn(channels: int) -> nn.GroupNorm:
    for g in (8, 4, 2):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond)[:, :, None].chunk(2, dim=1)
        return x * (1.0 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 64,
                 mults=(1, 2, 4, 8), cond_dim: int = 128):
        super().__init__()
        dims = [base * m for m in mults]
        self.stem = nn.Conv1d(in_ch, dims[0], 3, padding=1)
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        skip_ch = [dims[0]]
        for i in range(len(dims) - 1):
            self.downs.append(nn.ModuleDict({
                "block": ResBlock(dims[i], dims[i + 1], cond_dim),
                "down": nn.Conv1d(dims[i + 1], dims[i + 1], 4, stride=2, padding=1),
            }))
            skip_ch.append(dims[i + 1])
        self.mid = nn.ModuleList([ResBlock(dims[-1], dims[-1], cond_dim),
                                  ResBlock(dims[-1], dims[-1], cond_dim)])
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(ResBlock(dims[i] + skip_ch[i], dims[i - 1], cond_dim))
        self.head = nn.Sequential(_gn(dims[0]),
                                  nn.SiLU(),
                                  nn.Conv1d(dims[0], out_ch, 3, padding=1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        skips = [h]
        for d in self.downs:
            h = d["block"](h, cond)
            skips.append(h)
            h = d["down"](h)
        for block in self.mid:
            h = block(h, cond)
        for i, up in enumerate(self.ups):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            skip = skips[-(i + 1)]
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = up(torch.cat([h, skip], dim=1), cond)
        return self.head(h)
