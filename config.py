from dataclasses import dataclass, asdict

DEFAULT_DATA_ROOT = "data/sensmarttech"


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    modality: str = "ecg"
    dataset: str = "sens"
    fs: int = 100
    window: int = 960
    stride: int = 480
    split: tuple = (0.7, 0.15, 0.15)
    per_record_norm: bool = True
    seed: int = 42
    cache_dir: str = "cache"

    blackout_prob: float = 0.75
    max_segments: int = 4
    seg_min: int = 20
    seg_max: int = 192
    point_prob: float = 0.10
    channel_drop_prob: float = 0.15

    diff_T: int = 500
    ddim_steps: int = 25
    base_dim: int = 64
    dim_mults: tuple = (1, 2, 4, 8)
    n_samples: int = 16
    clip_x0: float = 4.0

    gru_hidden: int = 256
    gru_layers: int = 2
    feat_dim: int = 128
    period_sec: float = 1.2
    skeleton_topk: int = 8
    use_gru_prior: bool = True
    use_freq_prior: bool = True
    residual: bool = True
    cond_dropout: float = 0.15
    no_impose: bool = False
    no_comp: bool = False

    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    freeze_prior_epochs: int = 0
    val_every: int = 5
    val_max_windows: int = 64
    num_workers: int = 0
    device: str = "cuda"

    csdi_layers: int = 4
    csdi_heads: int = 4
    csdi_dim: int = 64

    def validate(self):
        period = self.period_samples
        if self.window % period != 0:
            raise ValueError(
                f"window ({self.window}) must be divisible by "
                f"period_samples ({period}); adjust window or period_sec/fs")
        return self

    @property
    def period_samples(self) -> int:
        return max(4, int(round(self.period_sec * self.fs)))

    def ablation_tag(self) -> str:
        tag = ""
        if self.no_impose:
            tag += "_no_impose"
        if self.no_comp:
            tag += "_no_comp"
        if not self.use_gru_prior:
            tag += "_no_gru"
        if not self.use_freq_prior:
            tag += "_no_freq"
        if not self.residual:
            tag += "_direct"
        return tag

    def to_dict(self):
        return asdict(self)
