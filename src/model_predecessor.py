import torch
import torch.nn as nn

from .diffusion import GaussianDiffusion
from .prior import CausalPrior, decomposition_channels
from .unet1d import UNet1D, timestep_embedding


class CPGDCardioPredecessor(nn.Module):
    def __init__(self, cfg, n_channels: int, diffusion: GaussianDiffusion):
        super().__init__()
        self.cfg = cfg
        self.C = n_channels
        self.diff = diffusion
        self.use_gru = cfg.use_gru_prior
        self.use_freq = cfg.use_freq_prior
        self.residual = bool(getattr(cfg, "residual", True))
        if self.use_gru:
            self.prior = CausalPrior(n_channels, cfg.gru_hidden, cfg.gru_layers,
                                     cfg.feat_dim)
        self.time_dim = 128
        K = 3 * n_channels if self.use_freq else 0
        in_ch = 3 * n_channels + K
        cond_dim = self.time_dim + (cfg.feat_dim if self.use_gru else 0)
        self.eps_net = UNet1D(in_ch, out_ch=n_channels, base=cfg.base_dim,
                              mults=tuple(cfg.dim_mults), cond_dim=cond_dim)

    def _build_condition(self, x: torch.Tensor, mask: torch.Tensor):
        valid = mask
        x_obs = x * mask
        prior, f, aux = x_obs, None, None
        if self.use_gru:
            prior, f = self.prior(x_obs, valid)
            missing = 1.0 - mask
            denom = missing.sum().clamp(min=1.0)
            aux = ((prior - x) ** 2 * missing).sum() / denom
        cond = torch.cat([prior, valid], dim=1)
        if self.use_freq:
            trend, seasonal, skeleton = decomposition_channels(
                prior.detach() if self.use_gru else x_obs,
                self.cfg.period_samples, self.cfg.skeleton_topk)
            cond = torch.cat([cond, trend, seasonal, skeleton], dim=1)
        if self.training and self.cfg.cond_dropout > 0:
            keep = (torch.rand(x.shape[0], 1, 1, device=x.device)
                    > self.cfg.cond_dropout).float()
            cond = torch.cat([cond[:, :self.C] * keep,
                              cond[:, self.C:2 * self.C],
                              cond[:, 2 * self.C:] * keep], dim=1)
            if f is not None:
                f = f * keep[:, 0]
        return prior, f, cond, aux

    def _cond_vec(self, t: torch.Tensor, f):
        cv = timestep_embedding(t, self.time_dim)
        if f is not None:
            cv = torch.cat([cv, f], dim=-1)
        return cv

    def forward_loss(self, x: torch.Tensor, mask: torch.Tensor,
                     t: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        if t is None:
            t = torch.randint(0, self.diff.T, (B,), device=x.device)
        prior, f, cond_ch, aux = self._build_condition(x, mask)
        target = (x - prior.detach()) if self.residual else x
        noise = torch.randn_like(target)
        x_t = self.diff.q_sample(target, t, noise)
        eps_hat = self.eps_net(torch.cat([x_t, cond_ch], dim=1),
                               self._cond_vec(t, f))
        loss = self.diff.loss(eps_hat, noise, missing=1.0 - mask)
        if aux is not None:
            loss = loss + aux
        return loss

    @torch.no_grad()
    def impute(self, x: torch.Tensor, mask: torch.Tensor, n_samples: int = 16,
               chunk: int = 4) -> torch.Tensor:
        prior, f, cond_ch, _ = self._build_condition(x, mask)
        B, C, L = x.shape
        base = prior if (self.residual and self.use_gru) else \
            torch.zeros_like(prior)
        outs = []
        done = 0
        while done < n_samples:
            m = min(chunk, n_samples - done)

            def eps_fn(x_t, t, cond_ch=cond_ch, f=f, m=m):
                cv = self._cond_vec(t, f)
                return self.eps_net(torch.cat([x_t, cond_ch], dim=1), cv)

            outs.append(self.diff.ddim_sample(
                eps_fn, (B, C, L), self.cfg.ddim_steps, n_samples=m))
            done += m
        samples = base[None] + torch.cat(outs, dim=0)
        return torch.where(mask[None] > 0, x[None], samples)
