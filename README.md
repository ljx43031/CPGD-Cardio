# CPGD-Cardio

CPGD-Cardio (**C**onstrained-**P**rior **G**RU-**D**iffusion) — code for the
paper **"CPGD-Cardio: Constrained-Prior GRU-Diffusion for Physiologically
Faithful Reconstruction of Missing Data in Cardiovascular Signals"**.  The
name tracks the cascade: the beat-transplant operator is a prior fixed by
arithmetic constraints, the GRU absorbs the residual it leaves, and the
diffusion model quantifies only what remains.

CPGD-Cardio reconstructs and probabilistically forecasts cardiovascular signals
(ECG / PCG / PPG / acceleration) when segments or whole channels are missing.
Each window passes through a three-track cascade:

1. **Beat transplant (rhythm track)** — R-peaks are detected on the observed
   signal, in-gap beat times are predicted from the observed R–R trend
   (blend interpolation, or state-conditional resampling for irregular
   rhythm), and fully observed beat cycles are transplanted into the gap.
2. **GRU compensation (point estimate)** — a causal GRU fits the residuals
   left by the transplant and yields the point estimate.
3. **Residual diffusion (uncertainty)** — an epsilon-prediction diffusion with
   a cosine schedule and DDIM sampling models the residual around the point
   estimate, giving calibrated 90% prediction intervals.

## Repository layout

```
├── config.py               # all hyperparameters (dataclass Config)
├── train.py                # train one model on one dataset/modality
├── evaluate.py             # evaluate all methods under fixed missing protocols
├── run_experiments.py      # full experiment matrix + aggregation
├── smoke_test.py           # end-to-end check on tiny synthetic data
├── requirements.txt
├── scripts/
│   ├── fetch_physionet.py        # download MIT-BIH / PTB-XL into NPZ records
│   ├── make_paper_tables.py      # main + seed tables and headline counts
│   ├── unified_fidelity.py       # beat-timing/energy fidelity metrics
│   ├── make_ablation_table.py    # ablation table
│   ├── measure_interval_width.py # mean 90% interval width
│   ├── context_ablation.py       # gap-position (offline vs. online) study
│   ├── nonoverlap_check.py       # non-overlapping-subset significance check
│   └── plot_paper_figures.py     # Fig. 1 and Fig. 2
└── src/
    ├── model.py            # CPGD-Cardio model, GRU imputer, model factory
    ├── model_predecessor.py# earlier configuration (design-space studies)
    ├── csdi_baseline.py    # CSDI baseline (compact reimplementation)
    ├── diffusion.py        # Gaussian diffusion (cosine schedule, DDIM)
    ├── prior.py            # causal GRU prior + frequency-condition channels
    ├── unet1d.py           # 1D UNet denoiser with FiLM conditioning
    ├── dataset.py          # windowing, normalization, subject-level splits
    ├── wfdb_dataset.py     # PhysioNet NPZ record loader
    ├── masking.py          # train-time mask augmentation + eval protocols
    └── metrics.py          # masked errors, CRPS, coverage, Wilcoxon
```

## Installation

Python 3.9+ is recommended (developed on 3.10). A CUDA GPU is strongly
recommended for the diffusion models.

```bash
pip install -r requirements.txt
```

On Windows with a conda environment, importing scipy together with torch can
raise `OMP: Error #15 (libiomp5md.dll already initialized)`; if that happens,
set the environment variable `KMP_DUPLICATE_LIB_OK=TRUE` before running.

## Datasets

Three datasets are supported. All splits are **subject-level** (inter-record)
so that no subject appears in both train and test. Normalization statistics,
window caches (`cache/`), checkpoints (`results/ckpt/`) and evaluation tables
(`results/`) are generated automatically on first use.

### 1. MIT-BIH Arrhythmia Database (`--dataset mitdb`)

48 two-lead ECG records at 360 Hz, resampled to 100 Hz on fetch. Download all
records into `data/physionet/mitdb/` as per-record NPZ files:

```bash
python scripts/fetch_physionet.py --db mitdb --fs 100
```

Records `202` and `208` share patients with `201` and `207`; the loader groups
them so the split stays patient-disjoint.

### 2. PTB-XL (`--dataset ptbxl`)

12-lead ECG, 100 Hz (`records100/*_lr`) subset. Downloads a uniformly spread
subset (default 600 records) plus `ptbxl_database.csv`, which provides the
patient ids used for the split; records without a patient id are skipped:

```bash
python scripts/fetch_physionet.py --db ptbxl --n_records 600 --fs 100
```

### 3. SensSmartTech (`--dataset sens`)

A public PhysioNet database: *SensSmartTech database of cardiovascular
signals synchronously recorded by an electrocardiograph, phonocardiograph,
photoplethysmograph and accelerometer*, version 1.0.0 (Lazović et al., 2024,
PhysioNet, [DOI 10.13026/fy9p-n277](https://doi.org/10.13026/fy9p-n277),
CC BY 4.0). It contains 338 recordings of 30 s from 32 healthy volunteers
with synchronously recorded 4-lead ECG, PCG, dual-site PPG and z-axis
acceleration. Dataset page: <https://physionet.org/content/senssmarttech/1.0.0/>.

The code reads the per-record CSV files (the dataset's `CSV/` folder), named
`<subject>_<HHMMSS>_<modality>.csv` with a millisecond timestamp column `t`
followed by the signal channels:

| File            | Columns after `t`                                              |
|-----------------|----------------------------------------------------------------|
| `*_ecg.csv`     | `lead_I`, `lead_II`, `v3`, `v4`                                |
| `*_pcg.csv`     | `pcg`                                                          |
| `*_ppg.csv`     | `carotid_880nm`, `carotid_660nm`, `brachial_880nm`, `brachial_660nm` |
| `*_acc.csv`     | `az`                                                           |

Download (any one of the following):

```bash
# Option A: fetch only the CSV files the code reads (AWS, no sign-in needed)
aws s3 sync --no-sign-request s3://physionet-open/senssmarttech/1.0.0/CSV/ data/sensmarttech/

# Option B: mirror the full dataset (CSV + WFDB + demographics, ~265 MB)
wget -r -N -c -np https://physionet.org/files/senssmarttech/1.0.0/
mkdir -p data/sensmarttech
cp physionet.org/files/senssmarttech/1.0.0/CSV/*.csv data/sensmarttech/

# Option C: download the ZIP manually from the dataset page in a browser
# and unzip its CSV/ folder into data/sensmarttech/
```

The `WFDB/` folder, `Demographics.csv` and `RECORDS` in the dataset are not
used by the code. The subject id is the filename part before the first `_`
(32 subjects; the 70/15/15 split is subject-level, so the same volunteer
never appears in both train and test). Signals are linearly resampled to
`--fs` (default 100 Hz) and z-scored per record before global
standardization. Alternatively, place the CSVs anywhere and point at them
with `--data_root`:

```bash
python train.py --model cpgd --modality ecg --data_root /path/to/csv
```

Expected data layout after setup:

```
data/
├── physionet/
│   ├── mitdb/*.npz
│   └── ptbxl/*.npz
└── sensmarttech/*_ecg.csv ...
```

Checkpoints store the `data_root` they were trained with. Every evaluation
entry point (`evaluate.py` and `scripts/`) accepts `--data_root` to override
it; when the stored root does not exist on the current machine, they fall
back automatically to the packaged `data/` locations above, so a fresh
clone works out of the box.

## Quick start

```bash
python smoke_test.py      # trains all three models for 2 epochs on synthetic
                          # data and evaluates them (~2-4 min on GPU, no
                          # dataset needed) — run this first
```

Train and evaluate one dataset:

```bash
python train.py --model cpgd --dataset mitdb --seed 42
python train.py --model csdi --dataset mitdb --seed 42
python train.py --model gru  --dataset mitdb --seed 42

python evaluate.py --ckpt results/ckpt/cpgd_mitdb_ecg_s42.pt \
    --csdi_ckpt results/ckpt/csdi_mitdb_ecg_s42.pt \
    --gru_ckpt  results/ckpt/gru_mitdb_ecg_s42.pt
```

Full matrix over modalities/seeds with aggregation to
`results/tables/summary.csv`:

```bash
python run_experiments.py --dataset mitdb --seeds 42 7 123
python run_experiments.py --dataset sens --modalities ecg acc pcg --seeds 42 7 123
```

Useful `train.py` flags: `--fs 500` (PCG), `--window/--stride/--period_sec`,
`--epochs/--batch_size/--lr`, `--device cpu`, and `--ckpt` for explicit
checkpoint paths.

## Compared methods

All methods are trained by `train.py` and compared by `evaluate.py` in a
single run:

| Method           | Kind                          | Flag           |
|------------------|-------------------------------|----------------|
| CPGD-Cardio      | proposed (constrained-prior GRU-diffusion) | `--model cpgd` |
| CSDI             | diffusion imputation baseline (Tashiro et al., NeurIPS 2021, compact reimplementation) | `--model csdi` |
| GRU imputer      | discriminative masked-MSE baseline | `--model gru` |
| Linear / cubic-spline interpolation, mean fill, LOCF | classical baselines | evaluated automatically, no training |

## Evaluation protocols and metrics

Missingness is imposed deterministically per test window under four fixed
protocols: `point10` (10% i.i.d. point dropout), `seg5x20` (five 0.2 s
segments), `blackout20` (one contiguous 20% gap, the main protocol) and
`sensordrop` (one whole channel; multi-channel data only). Protocol lengths
are defined in seconds and scale with the sampling rate.

Reported metrics: masked MSE/MAE (normalized and denormalized), sample-based
CRPS, 90% interval coverage, and paired Wilcoxon signed-rank tests of
CPGD-Cardio against every baseline. Results are written to
`results/eval_<dataset>_<modality>_s<seed>.csv` (+ `_sig.csv` for the tests).

## Ablations

```bash
python train.py --model cpgd --dataset mitdb --no_impose   # beat transplant off
python train.py --model cpgd --dataset mitdb --no_comp     # GRU compensation off
```

## Reproducing the paper

All experiments were trained for 150 epochs on a single CUDA GPU: three seeds
(42 / 7 / 123) on SensSmartTech ECG, seed 42 elsewhere. Evaluation is
deterministic given a checkpoint — masks are keyed to the original window
index and the per-window sampling seed is fixed — so a released checkpoint
reproduces its numbers exactly; retraining reproduces results
statistically (the pipeline is fully seeded, but CUDA kernels are not
bit-deterministic across GPU models and library versions). Re-running
everything takes roughly a day of GPU time per stage.

### 1. Train and evaluate the main comparison

```bash
# SensSmartTech ECG / ACC / PCG (repeat the ECG block with --seed 7 and 123)
python train.py --model cpgd --dataset sens --modality ecg --epochs 150
python train.py --model csdi --dataset sens --modality ecg --epochs 150
python train.py --model gru  --dataset sens --modality ecg --epochs 150

# MIT-BIH (non-overlapping windows)
python train.py --model cpgd --dataset mitdb --stride 960 --epochs 150
python train.py --model csdi --dataset mitdb --stride 960 --epochs 150
python train.py --model gru  --dataset mitdb --stride 960 --epochs 150

# PTB-XL (shorter windows, denser stride)
python train.py --model cpgd --dataset ptbxl --window 480 --stride 240 --epochs 150
python train.py --model csdi --dataset ptbxl --window 480 --stride 240 --epochs 150
python train.py --model gru  --dataset ptbxl --window 480 --stride 240 --epochs 150

# Evaluation protocol used for every table (200 test windows)
python evaluate.py --ckpt results/ckpt/cpgd_<dataset>_<modality>_s<seed>.pt \
    --csdi_ckpt results/ckpt/csdi_<dataset>_<modality>_s<seed>.pt \
    --gru_ckpt  results/ckpt/gru_<dataset>_<modality>_s<seed>.pt \
    --shuffle --eval_seed 0 --per_record_cap 25 --max_windows 200
```

`run_experiments.py` drives this whole matrix (models x seeds x modalities)
and aggregates the evaluation CSVs across seeds.

### 2. Reproduce the tables and figures

| Paper item | Command |
|---|---|
| Main table (18 MSE cells) and three-seed table + headline counts | `python scripts/make_paper_tables.py` |
| Fidelity table (beat timing, hit rate, band ratio, waveform corr.) | `python scripts/unified_fidelity.py` |
| Ablation table (see step 3) | `python scripts/make_ablation_table.py` |
| Mean 90% interval-width column | `python scripts/measure_interval_width.py` |
| Gap-position (offline vs. online) numbers | `python scripts/context_ablation.py` |
| Non-overlapping-subset significance | `python scripts/nonoverlap_check.py --eval_csv results/eval_sens_ecg_s42.csv --ckpt results/ckpt/cpgd_sens_ecg_s42.pt` (then the PTB-XL pair) |
| Fig. 1 (reconstruction) and Fig. 2 (metric bars) | `python scripts/plot_paper_figures.py --ckpt results/ckpt/cpgd_sens_ecg_s42.pt --csdi_ckpt results/ckpt/csdi_sens_ecg_s42.pt` |

### 3. Ablations (final configuration)

```bash
python train.py --model cpgd --dataset sens --modality ecg --epochs 150 --no_impose
python train.py --model cpgd --dataset sens --modality ecg --epochs 150 --no_comp
python evaluate.py --ckpt results/ckpt/cpgd_sens_ecg_s42_no_impose.pt \
    --out results/eval_sens_ecg_s42_no_impose.csv \
    --shuffle --eval_seed 0 --per_record_cap 25 --max_windows 100
python evaluate.py --ckpt results/ckpt/cpgd_sens_ecg_s42_no_comp.pt \
    --out results/eval_sens_ecg_s42_no_comp.csv \
    --shuffle --eval_seed 0 --per_record_cap 25 --max_windows 100
python scripts/make_ablation_table.py
```

### 4. Predecessor-configuration studies

The design-space studies — dataset dependence of the spectral prior,
predecessor ablations, PCG at 500 Hz, and the channel-dropout sensitivity —
were run on the earlier configuration, kept for reference as
`--model predecessor`. They were run before the subject-level split was
corrected, so the released pipeline reproduces their design-space
conclusions, but not the exact percentages printed in those sections.

```bash
# spectral-prior study (repeat for mitdb / ptbxl, and --seed 7 123 on ECG)
python train.py --model predecessor --dataset sens --modality ecg --epochs 150
python train.py --model predecessor --dataset sens --modality ecg --epochs 150 --no_freq_prior
python evaluate.py --ckpt results/ckpt/predecessor_sens_ecg_s42_no_freq.pt \
    --max_windows 200 --out results/eval_sens_ecg_s42_no_freq.csv

# predecessor ablations
python train.py --model predecessor --dataset sens --modality ecg --epochs 150 --no_residual
python train.py --model predecessor --dataset sens --modality ecg --epochs 150 --no_gru_prior

# PCG at 500 Hz (4.8 s non-overlapping windows, gap durations scaled)
python train.py --model predecessor --dataset sens --modality pcg --fs 500 \
    --window 2400 --stride 2400 --seg_min 100 --seg_max 960 --epochs 150 \
    --ckpt results/ckpt/predecessor_sens_pcg500_s42.pt
python train.py --model csdi --dataset sens --modality pcg --fs 500 \
    --window 2400 --stride 2400 --seg_min 100 --seg_max 960 --epochs 150 \
    --ckpt results/ckpt/csdi_sens_pcg500_s42.pt
python train.py --model gru --dataset sens --modality pcg --fs 500 \
    --window 2400 --stride 2400 --seg_min 100 --seg_max 960 --epochs 150 \
    --ckpt results/ckpt/gru_sens_pcg500_s42.pt
python evaluate.py --ckpt results/ckpt/predecessor_sens_pcg500_s42.pt \
    --csdi_ckpt results/ckpt/csdi_sens_pcg500_s42.pt \
    --gru_ckpt results/ckpt/gru_sens_pcg500_s42.pt --max_windows 200

# channel-dropout sensitivity (Discussion; mask ranges of that period)
python train.py --model predecessor --dataset mitdb --stride 960 --epochs 150 \
    --channel_drop_prob 0.30 --seg_min 30 --seg_max 120
```

## Citation

If you use this code, please cite the paper (citation block to be added),
the SensSmartTech database,

> Lazović, A., Tadić, P., Đorđević, N., Atanasoski, V., Tiosavljevic, M.,
> Ivanovic, M., Hadzievski, L., Ristic, A., Vukcevic, V., & Petrovic, J.
> (2024). SensSmartTech database of cardiovascular signals synchronously
> recorded by an electrocardiograph, phonocardiograph, photoplethysmograph
> and accelerometer (version 1.0.0). PhysioNet.
> <https://doi.org/10.13026/fy9p-n277>

and, if you use the diffusion-imputation baseline, Tashiro et al.,
"CSDI: Conditional Score-based Diffusion Models for Probabilistic Time
Series Imputation", NeurIPS 2021. The MIT-BIH Arrhythmia Database and
PTB-XL are also distributed by PhysioNet and should be cited per their
dataset pages.
