# Performative Latents for Adaptive Unsupervised DDSP (PLAUD)

PLAUD is a modular PyTorch framework for Differentiable Digital Signal Processing (DDSP). It targets small, personal datasets and emphasizes playability, modularity, and real-time use. The system is configurable end-to-end: datasets, feature/control spaces, synthesis blocks, losses, training regimes (including optional adversarial), and a transformer-based Prior model for generative control.

**Core idea**: A single, explicit ControlSpace defines the controls (features and/or latents), their dimensions, and control rate. The dataset and model are built to respect this schema, making components composable and exports consistent.

## Highlights

- **Explicit ControlSpace**: Authoritative source for control dimensions and rate.
- **Modular synth routing**: Choose synth blocks and their parameters via config (no inspect-based magic).
- **Multichannel output**: Set `audio.n_channels` to synthesize and export N audio channels (mono by default).
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
- New keys require a leading `+` (struct mode).
- GPU is used when available. For CPU-only runs, set the Trainer `accelerator='cpu'` in the config or script, and ensure CUDA defaults are disabled.

### 2) Train the Prior

Train the transformer Prior on control sequences derived from your dataset and a trained DDSP model.

On the first run, `cli.train_prior` automatically builds an LMDB cache next to the dataset directory (named like `prior_cache_<key>.lmdb`). It then trains the Prior on that cache, and subsequent runs reuse it.

Run it with the *same* config + overrides you used for DDSP training:

```zsh
python -m cli.train_prior -cn experiment_features_only \
  data.dataset_path=/absolute/path/to/dataset \
  ++experiment.name=my_run
```

Common overrides:

```zsh
python -m cli.train_prior -cn experiment_features_only \
  data.dataset_path=/absolute/path/to/dataset \
  ++experiment.name=my_run \
  prior.model.max_len=64 \
  prior.dataset.stride_factor=0.2 \
  prior.training.max_epochs=10 \
  prior.training.batch_size=256
```

Notes:
- If `prior.enabled=false` in the chosen config, the script will print a message and exit.
- To force rebuilding the cache, delete the `prior_cache_*.lmdb` directory (or change cache-relevant settings like `prior.model.max_len` / `prior.dataset.stride_factor`).

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
  --prior_directory /path/to/training/prior/my_run \
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

## Multichannel

`audio.n_channels` (default `1`) sets the number of audio channels the model synthesizes. Override it
like any other field:

```zsh
python -m cli.train -cn experiment_features_only \
  data.dataset_path=/absolute/path/to/dataset \
  ++audio.n_channels=2
```

Behavior:

- **Decoder per-channel heads**: a single shared latent/feature stream drives `n_channels` independent
  sets of synth parameters, so each channel is synthesized separately. Channels differ only through
  the decoder — the encoder and feature extractors consume a **mono downmix** of the input.
- **Data**: preprocess with channels preserved — `utils/dataset_converter.py` keeps the source channel
  layout by default (pass `--channels N` to force one). At load time, files with fewer channels than
  `n_channels` are **zero-padded** and files with more are **cropped** to the first `n_channels`.
- **Adversarial**: the discriminator runs per channel (each channel judged as an independent mono
  example) and is averaged.
- **Export**: `decode` exposes `n_channels` audio outputs (one nn~ signal outlet per channel), and
  `encode` accepts `n_channels` inputs.
- **Backward compatible**: at `n_channels=1` everything is identical to the mono model and existing
  checkpoints load unchanged. Changing `n_channels` rebuilds the dataset cache (it is part of the
  cache key).

## Testing

Run the test suite to validate shapes, losses, exporter, and integration.

```zsh
python -m pytest -q
```

## Troubleshooting

- If the prior cache build fails with “No control windows produced”, your dataset chunks are too short for `prior.model.max_len`; reduce `prior.model.max_len`, increase dataset duration, or adjust `audio.chunk_duration_s`.
- If you see Hydra struct mode errors when adding keys, prefix overrides with `+`.
- For CPU-only environments, ensure CUDA defaults are disabled and set Trainer `accelerator='cpu'`.


## Max/MSP and PureData

PLAUD exports are compatible with `nn~` externals for Max/MSP and PureData. Install the externals from the [nn~ repository](https://github.com/acids-ircam/nn_tilde) and load the exported `.ts` model.
