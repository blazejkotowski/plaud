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

## Synthesizers

The framework ships several differentiable synth blocks. Each is configured in the YAML `model.synths` list and instantiated via the synth registry.

| Synth                    | `n_params`         | Description |
|--------------------------|--------------------|-------------|
| `BendableNoiseBandSynth` | `n_filters`        | Filtered noise with optional waveshaping and component limiting. |
| `NoiseBandSynth`         | `n_filters`        | Classic noise-band synthesis. |
| `SineSynth`              | `3 * n_sines`      | Mixture of sine waves (freq, amp, phase per partial). |
| `SubbandSineSynth`       | `3 * n_sines`      | Subband sine synthesis with waveshaping. |
| `HarmonicSynth`          | `n_harmonics`      | Additive harmonic synthesis — see below. |
| `FilteredNoiseSynth`     | `n_bands`          | FFT-based filtered noise (DDSP reference) — see below. |

### HarmonicSynth

`HarmonicSynth` generates audio as a weighted sum of harmonically-related sine waves. Unlike other synths, **pitch (f0) and loudness are not predicted by the decoder** — they come from feature extractors (e.g., CREPE for pitch, librosa RMS for loudness) declared in the `ControlSpace`. The decoder only predicts `n_harmonics` raw amplitude parameters.

A `scale_function` (modified sigmoid: `2 * sigmoid(x)^log(10) + 1e-7`) is applied to produce positive amplitudes. Harmonics above the Nyquist frequency are zeroed, the remaining amplitudes are normalized to sum to 1, and the result is then scaled by the external loudness envelope. This means the decoder learns **timbral structure** (which harmonics are prominent), while overall energy is entirely controlled by the loudness feature.

This follows the DDSP reference implementation (Engel et al., ICLR 2020; ACIDS `ddsp_pytorch`).

This means HarmonicSynth **requires both a pitch and a loudness feature field** in the control space and cannot work in latent-only mode.

### FilteredNoiseSynth

`FilteredNoiseSynth` generates audio by filtering white noise through a learned frequency-domain filter, following the DDSP reference. The decoder predicts `n_bands` filter magnitude parameters at control rate. These are passed through `scale_function` (with a -5 bias to start quiet), converted to impulse responses via inverse FFT, and convolved with uniform white noise using FFT convolution.

### Reverb

A trainable `Reverb` module can be optionally enabled via `reverb: true` in the YAML config. It applies a learnable impulse response via FFT convolution as a post-processing step after all synths are summed. A learnable `dry_wet` parameter controls the mix (initialized mostly dry). Set `reverb_length` (in samples) to control the IR length.

**YAML example** (`experiment_features_only_harmonic.yaml`):

```yaml
model:
  control_space:
    - name: loudness        # required by HarmonicSynth
      dim: 1
      source: feature
      extractor: LibrosaFeatureExtractor
      params:
        feature_fn_name: rms
    - name: pitch          # required by HarmonicSynth
      dim: 1
      source: feature
      extractor: CrepePitchExtractor
      params:
        postprocess: true
  synths:
    - type: HarmonicSynth
      params:
        n_harmonics: 100
        # pitch_feature: pitch       # default — matches the field name above
        # loudness_feature: loudness  # default — matches the field name above
    - type: FilteredNoiseSynth
      params:
        n_bands: 65
  reverb: true
  reverb_length: 44100
```

Key design points:
- `pitch_feature` (default `"pitch"`) names the ControlSpace field that supplies f0 in Hz.
- `loudness_feature` (default `"loudness"`) names the ControlSpace field that supplies the amplitude envelope.
- Decoder output goes through `scale_function` (modified sigmoid) → Nyquist filtering → normalize by sum → scaled by loudness.
- Phase is accumulated via `cumsum` without wrapping (no `torch.remainder`) — `sin()` is already periodic, and wrapping before multiplying by harmonic index would create phase discontinuities.
- Harmonics above the Nyquist frequency are automatically zeroed via `remove_above_nyquist`.
- Streaming mode maintains phase continuity across chunks.
- `FilteredNoiseSynth` and `Reverb` can be combined with `HarmonicSynth` for the full DDSP pipeline (harmonic + noise + reverb).

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

- If the prior cache build fails with “No control windows produced”, your dataset chunks are too short for `prior.model.max_len`; reduce `prior.model.max_len`, increase dataset duration, or adjust `audio.chunk_duration_s`.
- If you see Hydra struct mode errors when adding keys, prefix overrides with `+`.
- For CPU-only environments, ensure CUDA defaults are disabled and set Trainer `accelerator='cpu'`.


## Max/MSP and PureData

PLAUD exports are compatible with `nn~` externals for Max/MSP and PureData. Install the externals from the [nn~ repository](https://github.com/acids-ircam/nn_tilde) and load the exported `.ts` model.
