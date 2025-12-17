# Refactoring Summary and Usage Guide

This document summarizes the architectural refactor and provides practical instructions to run the new pipelines: DDSP training, control export, and Prior training.

## Overview

- Modular architecture centered on an explicit ControlSpace and component registries.
- DDSP model requires ControlSpace and supports feature-only, latent-only, and hybrid configurations.
- Flexible loss configuration; adversarial regime optional via config.
- Export pipeline produces concatenated control sequences for Prior training.
- Prior model learns next-step discrete distributions via Mu-Law quantization while consuming continuous controls at runtime.

## Key Changes

### Interfaces and Registries
- `ddsp/interfaces.py`
  - ControlSpace and ControlField schema (names, dimensions, control rate).
  - Protocol-like interfaces for FeatureExtractor, Encoder/Decoder, SynthBlock, Loss.
- `ddsp/registry.py`
  - Category registries (`SYNTHS`, `FEATURE_EXTRACTORS`, `LOSSES`, etc.).
  - `build_from_config` helpers to assemble components from Hydra/OmegaConf configs.

### ControlSpace
- Explicit, mandatory ControlSpace defines control parameters, dimensions, and rate.
- Feature extractors operate at audio rate; dataset downsamples once to control rate.
- Used consistently across dataset and DDSP model to derive `feature_dim`/`latent_dim`.

### Dataset
- `ddsp/audio_feature_dataset.py`
  - Builds feature extractors from ControlSpace via registry.
  - Caches audio-rate features; normalization via mean/std or min/max.
  - Single downsample to control rate in `__getitem__`.

### DDSP Model
- `ddsp/ddsp.py`
  - Constructor requires ControlSpace; derives dims; optional encoder.
  - Decoder merges features and latents; synthesis routing is explicit per synth (no inspect-based dispatch).
  - Adversarial schedule: optional discriminator, epoch-gated weights.
  - Flexible loss list built from config; `_reconstruction_loss` aggregates weighted entries.

### Losses
- `ddsp/losses/__init__.py`
  - Registers `AttributeRegularizationLoss`, `SlicedWassersteinLoss`, `MultiScaleSlicedWassersteinLoss`.
  - MRSTFT loss supported via auraloss; default config uses MRSTFT-only.

### CLI and Configs
- `cli/train.py`
  - Hydra-driven DDSP training; constructs dataset and model from `configs/experiment*.yaml`.
- `configs/experiment.yaml` and variants
  - Audio and control_space definitions; model and adversarial settings.
  - `losses` block configurable as a list; default MRSTFT-only.

### Prior Model and Pipeline
- `ddsp/prior/prior.py`
  - Transformer encoder over token embeddings; Mu-Law quantization (`MuLawEncoding/MuLawDecoding`).
  - `normalize`/`denormalize` guards for missing normalization dict.
  - `generate` uses internal device; scheduler monitors `loss` for simple runs.
- `ddsp/prior/exporter.py`
  - Exports concatenated `controls` and separate `latents` to HDF5.
  - Robust to encoder interfaces; disables streaming if supported.
- `ddsp/prior/dataset.py`
  - Loads `controls` (preferred) or `latents` from HDF5; synthetic fallback available.
- `cli/train_prior.py`
  - Hydra CLI to train Prior from HDF5 controls.
- `cli/export_prior_latents.py`
  - Hydra CLI to export controls using the same experiment config used for DDSP training.

### Tests
- `tests/test_prior_smoke.py`: Prior forward/generate shapes and finiteness.
- `tests/test_prior_exporter.py`: Exporter HDF5 contents (`controls`, `latents`).
- `tests/test_prior_integration.py`: Export → load controls → train Prior one epoch and assert finite loss.
- Existing DDSP tests remain: shapes, ControlSpace, synthesis parameter slicing, loss/gradient flow, adversarial schedule, config integration.

## How to Run

### 1) DDSP Training (Hydra)
- Default training:
```
python -m cli.train
```
- Select a variant (features-only, latent-only, hybrid):
```
python -m cli.train +experiment=experiment_features_only
python -m cli.train +experiment=experiment_latent_only
python -m cli.train +experiment=experiment_hybrid
```
- Override common options:
```
python -m cli.train losses=[{name: mrstft, weight: 1.0}] adversarial.enabled=false
```

### 2) Export Controls for Prior
- Export using the same experiment config (ControlSpace is the source of truth):
```
python -m cli.export_prior_latents prior_export.out_path=/absolute/path/latents.h5
```
- Output HDF5 contains:
  - `controls`: [num_sequences, seq_len, control_size]
  - `latents`: [num_sequences, seq_len, latent_size]

### 3) Train the Prior (Hydra)
- Train from exported HDF5:
```
python -m cli.train_prior dataset.hdf5_path=/absolute/path/latents.h5
```
- Config defaults in `configs/experiment_prior.yaml`:
  - `prior`: latent_size, embedding_dim, quantization_channels, nhead, num_layers, dropout, max_len, lr
  - `dataset`: hdf5_path, num_sequences, seq_len, latent_size (used for synthetic only)
  - `training`: batch_size, num_workers, max_epochs

## Configuration Notes

- ControlSpace drives feature names, dimensions, and rate; dataset and model must share the same ControlSpace.
- Losses: configured as a list of entries with `name` and `weight`; MRSTFT-only is a sensible default.
- Adversarial: gated via epoch thresholds in config; can be disabled entirely.
- Prior: Expects sequences shaped `[batch, seq_len, control_size]`; internally quantizes targets with Mu-Law for classification and reports `loss`, `mse`, `acc`.

## Design Rationale

- Explicit ControlSpace ensures consistent feature extraction and control-rate alignment.
- Registries enable plugin-like extension for synths, losses, and extractors without changing core code.
- Exporting controls to HDF5 decouples DDSP training from Prior training, improving scalability and reproducibility.
- Tests cover shape contracts, pipeline correctness, and basic training loops for confidence.

## Tips and Troubleshooting

- GPU usage: CLIs auto-select GPU if available; tests use CPU for portability.
- If validation metrics are not available, the Prior scheduler monitors `loss` by default to avoid misconfiguration.
- HDF5 export requires sufficient audio length; check `sequence_length` and dataset duration.
- To extend ControlSpace or add new features/losses, register them via `ddsp/registry.py` and reference in configs.

## Files and Entry Points

- Model: `ddsp/ddsp.py`, `ddsp/prior/prior.py`
- Datasets: `ddsp/audio_feature_dataset.py`, `ddsp/prior/dataset.py`
- Export: `ddsp/prior/exporter.py`, `cli/export_prior_latents.py`
- Training CLIs: `cli/train.py`, `cli/train_prior.py`
- Configs: `configs/experiment*.yaml`, `configs/experiment_prior.yaml`
- Tests: `tests/` (DDSP + Prior + exporter + integration)

---
If you need tailored examples for your specific dataset/config layout, or want me to add a README quickstart with command snippets, let me know.
