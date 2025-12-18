# Performative Latents for Adaptive Unsupervised DDSP (PLAUD)

PLAUD is a modular PyTorch framework for Differentiable Digital Signal Processing (DDSP). It targets small, personal datasets and emphasizes playability, modularity, and real-time use. The system is configurable end-to-end: datasets, feature/control spaces, synthesis blocks, losses, training regimes (including optional adversarial), and a transformer-based Prior model for generative control.

**Core idea**: A single, explicit ControlSpace defines the controls (features and/or latents), their dimensions, and control rate. The dataset and model are built to respect this schema, making components composable and exports consistent.

## Highlights

- **Explicit ControlSpace**: Authoritative source for control dimensions and rate.
- **Modular synth routing**: Choose synth blocks and their parameters via config (no inspect-based magic).
- **Configurable losses**: Flexible list (e.g., MRSTFT-only by default) with per-component logging.
- **Optional adversarial regime**: Gate discriminator and generator loss contributions by epoch.
- **Prior integration**: Transformer Prior learns on exported control sequences; continuous controls at runtime.
- **Exports**: TorchScript or ONNX models for `nn~` (Max/MSP, PureData).

## Install

```zsh
pip install -r requirements.txt
pip install -e .
```

## Concepts

- `ControlSpace`: Declares control fields (name, dim, source, extractor, normalization) and the rate. The DDSP model and dataset derive `feature_dim` and `latent_dim` strictly from this.
- **Registries**: Features, synth blocks, and losses are registered and instantiated from Hydra configs.
- **Dataset**: Computes audio-rate features, caches them, and downsamples once to control rate at load time.
- **Prior**: Transformer with Mu-Law tokens for training, consumes continuous controls at inference.

## Quickstart

### 1) Train DDSP (Hydra)

Select a config and point to your dataset path. Use `--config-name` (or `-cn`) to choose a config file, and `++` to override existing fields.

```zsh
# Features-only
python -m cli.train -cn experiment_features_only \
  data.dataset_path=/absolute/path/to/dataset \
  ++experiment.name=my_run

# Latent-only
python -m cli.train -cn experiment_latent_only \
  data.dataset_path=/absolute/path/to/dataset

# Hybrid
python -m cli.train -cn experiment_hybrid \
  data.dataset_path=/absolute/path/to/dataset
```

Notes:
- New keys require a leading `+` (struct mode): e.g., `+prior_export.out_path=...`.
- GPU is used when available. For CPU-only runs, set the Trainer `accelerator='cpu'` in the config or script, and ensure CUDA defaults are disabled.

### 2) Export controls for Prior

Build an HDF5 with concatenated `controls` (and `latents` when present). Use the same experiment config to keep ControlSpace consistent.

```zsh
python -m cli.export_prior_latents --config-name experiment_features_only \
  +prior_export.out_path=/absolute/path/controls_latents.h5
```

### 3) Train the Prior

Train the transformer Prior from exported HDF5 controls.

```zsh
python -m cli.train_prior dataset.hdf5_path=/absolute/path/controls_latents.h5 \
  training.max_epochs=10 training.batch_size=32
```

### 4) Export models (nn~ / TorchScript or ONNX)

Export the DDSP model (optionally with Prior) for real-time use.

```zsh
# TorchScript export
python -m cli.export \
  --model_directory /path/to/synth/training/dir \
  --output_path /path/to/output/model.ts \
  --type best \
  --target_fs 16000

# With Prior
python -m cli.export \
  --model_directory /path/to/synth/training/dir \
  --prior_directory /path/to/prior/training/dir \
  --output_path /path/to/output/model.ts \
  --type last \
  --target_fs 16000

# ONNX export
python -m cli.export \
  --model_directory /path/to/synth/training/dir \
  --output_path /path/to/output/model.onnx \
  --type best
```

Export details:
- The exporter reconstructs a minimal `ControlSpace` from checkpoint hparams (`feature_dim`, `latent_size`) and passes it to `DDSP.load_from_checkpoint(...)`.
- If those hparams are absent, export will fail fast with a clear message.

## Configuration

- Hydra configs live in `configs/experiment*.yaml`.
- Use `--config-name/-cn` to select a file; use `++path.to.field=value` to override existing fields; use `+new.key=value` to add new keys.
- Losses are a list with `name` and `weight`. MRSTFT-only is a sensible default.
- Adversarial regime is optional and epoch-gated.

## Testing

Run the test suite to validate shapes, losses, exporter, and integration.

```zsh
python -m pytest -q
```

## Troubleshooting

- HDF5 export requires enough audio for the window length; adjust `sequence_length` in dataset or use longer audio.
- If you see Hydra struct mode errors when adding keys, prefix overrides with `+`.
- For CPU-only environments, ensure CUDA defaults are disabled and set Trainer `accelerator='cpu'`.

## Max/MSP and PureData

PLAUD exports are compatible with `nn~` externals for Max/MSP and PureData. Install the externals from the [nn~ repository](https://github.com/acids-ircam/nn_tilde) and load the exported `.ts` model.
