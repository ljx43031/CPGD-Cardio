import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from src.dataset import build_datasets
from src.masking import eval_mask, sample_train_mask
from src.model import build_model, freeze_prior


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    choices=["cpgd", "csdi", "gru", "predecessor"],
                    default="cpgd")
    ap.add_argument("--modality", default="ecg")
    ap.add_argument("--dataset", choices=["sens", "mitdb", "ptbxl"],
                    default=None)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--val_max_windows", type=int, default=64)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_impose", action="store_true",
                    help="ablation: disable the beat-transplant imposition")
    ap.add_argument("--no_comp", action="store_true",
                    help="ablation: disable the GRU compensation")
    ap.add_argument("--no_freq_prior", action="store_true",
                    help="predecessor model only: disable the frequency prior")
    ap.add_argument("--no_gru_prior", action="store_true",
                    help="predecessor model only: disable the GRU prior")
    ap.add_argument("--no_residual", action="store_true",
                    help="predecessor model only: direct-space diffusion")
    ap.add_argument("--channel_drop_prob", type=float, default=None,
                    help="train-time whole-channel dropout probability")
    ap.add_argument("--fs", type=int, default=None,
                    help="resample rate in Hz")
    ap.add_argument("--seg_min", type=int, default=None,
                    help="shortest training blackout in samples")
    ap.add_argument("--seg_max", type=int, default=None,
                    help="longest training blackout in samples")
    ap.add_argument("--freeze_prior_epochs", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--period_sec", type=float, default=None)
    ap.add_argument("--diff_T", type=int, default=None)
    ap.add_argument("--ddim_steps", type=int, default=None)
    ap.add_argument("--base_dim", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path")
    ap.add_argument("--tiny", action="store_true",
                    help="one-flag tiny config for smoke tests (synthetic)")
    args = ap.parse_args()
    legacy = [f for f in ("no_freq_prior", "no_gru_prior", "no_residual")
              if getattr(args, f)]
    if legacy and args.model != "predecessor":
        ap.error(f"{legacy} are only valid with --model predecessor")
    return args


def apply_overrides(cfg: Config, args) -> Config:
    if getattr(args, "tiny", False):
        cfg.window, cfg.period_sec = 256, 0.64
        cfg.diff_T, cfg.ddim_steps = 100, 5
        cfg.base_dim, cfg.dim_mults = 16, (1, 2, 2)
        cfg.gru_hidden, cfg.gru_layers, cfg.feat_dim = 64, 1, 32
        cfg.epochs, cfg.batch_size, cfg.lr = 2, 8, 2e-3
        cfg.n_samples = 4
        cfg.val_every, cfg.val_max_windows = 1, 8
    if args.dataset and args.dataset != "sens":
        cfg.dataset = args.dataset
        if not args.data_root:
            cfg.data_root = os.path.join("data", "physionet", args.dataset)
    if args.data_root:
        cfg.data_root = args.data_root
    if args.model == "csdi" and args.batch_size is None:
        cfg.batch_size = 16
    for k in ("modality", "epochs", "batch_size", "lr", "seed", "device",
              "freeze_prior_epochs", "window", "stride", "period_sec", "diff_T",
              "ddim_steps", "base_dim", "channel_drop_prob", "seg_min",
              "seg_max", "fs"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.no_impose:
        cfg.no_impose = True
    if args.no_comp:
        cfg.no_comp = True
    if args.no_freq_prior:
        cfg.use_freq_prior = False
    if args.no_gru_prior:
        cfg.use_gru_prior = False
    if args.no_residual:
        cfg.residual = False
    return cfg.validate()


def ckpt_path(cfg: Config, model_kind: str, args) -> str:
    if args.ckpt:
        return args.ckpt
    os.makedirs("results/ckpt", exist_ok=True)
    return (f"results/ckpt/{model_kind}_{cfg.dataset}_{cfg.modality}_s{cfg.seed}"
            f"{cfg.ablation_tag()}{'_syn' if args.synthetic else ''}.pt")


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(
            {k: v.detach().clone() for k, v in model.state_dict().items()})

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


@torch.no_grad()
def validate(model, loader, cfg, device, mean, std, n_val_samples=4,
             max_windows=64):
    model.eval()
    mses, n_seen = [], 0
    for wi, x in enumerate(loader):
        if n_seen >= max_windows:
            break
        x = x.to(device)
        mask = torch.from_numpy(
            eval_mask("blackout20", cfg.window, x.shape[1], window_id=wi)
        )[None].expand(x.shape[0], -1, -1).contiguous().to(device)
        samples = model.impute(x, mask, n_samples=n_val_samples)
        pred = samples.mean(0).cpu().numpy()
        true = x.cpu().numpy()
        miss = (1 - mask.cpu().numpy()) == 1
        mses.append(float(((pred - true) ** 2)[miss].mean()))
        n_seen += x.shape[0]
    model.train()
    return float(np.mean(mses))


def main():
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    tr, va, _, meta = build_datasets(cfg, synthetic=args.synthetic)
    mean, std = np.asarray(meta["mean"], np.float32), np.asarray(meta["std"], np.float32)
    C = meta["n_channels"]
    loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(va, batch_size=max(cfg.batch_size // 2, 4),
                            num_workers=cfg.num_workers)

    model, diff = build_model(args.model, cfg, C, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.model}] modality={cfg.modality} C={C} windows="
          f"{len(tr)}/{len(va)} params={n_params/1e6:.2f}M device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema_decay)
    path = ckpt_path(cfg, args.model, args)
    log_path = path.replace(".pt", "_log.csv")
    best_val = float("inf")
    start = time.time()

    with open(log_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", "val_mse_denorm", "seconds"])
        for epoch in range(cfg.epochs):
            if args.model == "cpgd" and cfg.freeze_prior_epochs > 0:
                freeze_prior(model, epoch < cfg.freeze_prior_epochs)
            tot, nb = 0.0, 0
            model.train()
            for x in loader:
                x = x.to(device)
                B, Cb, L = x.shape
                masks = np.stack([
                    sample_train_mask(np.random.RandomState(
                        cfg.seed * 1000003 + epoch * 7919 + i), L, Cb, cfg)
                    for i in range(B)])
                mask = torch.from_numpy(masks).to(device)
                loss = model.forward_loss(x, mask)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)
                tot += float(loss.detach())
                nb += 1
            do_val = ((epoch + 1) % cfg.val_every == 0
                      or epoch == cfg.epochs - 1)
            if do_val:
                val_mse = validate(model, val_loader, cfg, device, mean, std,
                                   max_windows=cfg.val_max_windows)
            wr.writerow([epoch, tot / max(nb, 1),
                         val_mse if do_val else "",
                         round(time.time() - start, 1)])
            f.flush()
            marker = ""
            if do_val and val_mse < best_val:
                best_val = val_mse
                torch.save({"model": model.state_dict(), "ema": ema.shadow,
                            "cfg": cfg.to_dict(), "kind": args.model,
                            "val_mse": val_mse, "n_channels": C,
                            "mean": mean.tolist(), "std": std.tolist()}, path)
                marker = "  <- best"
            print(f"epoch {epoch:3d}  loss {tot/max(nb,1):.4f}  "
                  + (f"val_mse {val_mse:.5f}" if do_val else "(no val)")
                  + marker, flush=True)
    print(f"done in {(time.time()-start)/60:.1f} min; best val_mse {best_val:.5f};"
          f" ckpt -> {path}")


if __name__ == "__main__":
    main()
