import torch
import torch.nn as nn
import torch.nn.functional as F


def moving_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    w = kernel if kernel % 2 == 1 else kernel + 1
    pad = w // 2
    B, C, L = x.shape
    xf = F.pad(x.reshape(B * C, 1, L), (pad, pad), mode="replicate")
    k = torch.full((1, 1, w), 1.0 / w, device=x.device, dtype=x.dtype)
    return F.conv1d(xf, k).reshape(B, C, L)


def seasonal_fold(detrended: torch.Tensor, period: int) -> torch.Tensor:
    B, C, L = detrended.shape
    folds = max(1, L // period)
    L_eff = folds * period
    d = detrended[:, :, :L_eff].reshape(B, C, folds, period)
    seas = d.mean(dim=2, keepdim=True).expand(B, C, folds, period).reshape(B, C, L_eff)
    if L_eff < L:
        seas = F.pad(seas, (0, L - L_eff), mode="replicate")
    return seas


def harmonic_skeleton(detrended: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    B, C, L = detrended.shape
    X = torch.fft.rfft(detrended, dim=-1)
    amp = X.abs()
    amp[..., 0] = 0.0
    k = min(top_k, amp.shape[-1] - 1)
    idx = amp.topk(k, dim=-1).indices
    mask = torch.zeros_like(amp)
    mask.scatter_(-1, idx, 1.0)
    return torch.fft.irfft(X * mask, n=L, dim=-1)


def decomposition_channels(fill: torch.Tensor, period: int, top_k: int = 8):
    trend = moving_average(fill, kernel=2 * period)
    seasonal = seasonal_fold(fill - trend, period)
    skeleton = harmonic_skeleton(fill - trend, top_k=top_k)
    return trend, seasonal, skeleton


class CausalPrior(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 256, layers: int = 2,
                 feat_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=n_channels * 2, hidden_size=hidden,
                          num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_channels)
        self.feat = nn.Linear(hidden, feat_dim)

    def forward(self, x_masked: torch.Tensor, valid: torch.Tensor):
        inp = torch.cat([x_masked.transpose(1, 2) * valid.transpose(1, 2),
                         valid.transpose(1, 2)], dim=-1)
        h, _ = self.gru(inp)
        prior = self.head(h).transpose(1, 2)
        f = self.feat(h[:, -1])
        return prior, f
