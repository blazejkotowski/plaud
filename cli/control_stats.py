"""
Dataset statistics for the feature controls (loudness, centroid, ...), so that audio from outside the
training set can be analysed into the same control space the decoder was trained on.

Training turns each feature into [0, 1] between its 5th and 95th percentile over the WHOLE concatenated
dataset (LibrosaFeatureExtractor._calculate on AudioFeatureDataset's mono downmix). Those numbers are
not stored in the .ts, so a new file can't be placed on the same scale. This recomputes them exactly
the way training did, checks the result against the dataset's cached features, and writes them into
the model's JSON sidecar under "controls" (other keys are kept). The .ts itself is not touched, so the
model stays exactly as nn~ sees it. The latents come from the model's own encode() method.

  python -m cli.control_stats --config configs/night0908/05_putxet.yaml --json models/night0908/putxet.json

To analyse a new file into decoder controls with these numbers (what a host does):
  1. load at audio.fs, n_channels as in training, sum the channels to mono
  2. per feature: librosa.feature.<function>(y=mono, center=False) with the listed frame / hop sizes
     (librosa's default sr = 22050 for the centroid; the scale cancels out in the normalisation)
  3. interpolate the frames linearly to one value per sample (align_corners=False)
  4. clamp to [lo, hi], then (x - lo) / (hi - lo + 1e-8); then the optional causal moving average
     (smoothing_kernel_size samples) and the optional config normalisation
  5. interpolate to one value per resampling_factor samples (linear, align_corners=False)
"""

import argparse
import inspect
import json
import os
from typing import Any, Dict

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ddsp import AudioFeatureDataset
from ddsp.interfaces import build_control_space
from ddsp.feature_extractors.librosa_feature_extractor import LibrosaFeatureExtractor
from ddsp.feature_extractors.utils import _percentile_kth, normalize_feature, smoothen_feature

LOW_PERCENTILE, HIGH_PERCENTILE = 5.0, 95.0  # as in LibrosaFeatureExtractor._calculate


def _defaults(fn) -> Dict[str, Any]:
  out = {}
  for name, p in inspect.signature(fn).parameters.items():
    if name != 'center' and p.default is not inspect.Parameter.empty and isinstance(p.default, (int, float, str, bool)):
      out[name] = p.default
  return out


def compute_control_stats(cfg, verify: bool = True) -> Dict[str, Any]:
  """The "controls" section for a model trained with `cfg` (an OmegaConf config)."""
  fs = int(cfg.audio.fs)
  n_channels = int(getattr(cfg.audio, 'n_channels', 1))
  resampling_factor = int(cfg.model.resampling_factor)
  control_space = build_control_space(cfg.model.control_space)
  ds = AudioFeatureDataset(dataset_path=cfg.data.dataset_path,
                           n_signal=int(fs * float(cfg.audio.chunk_duration_s)),
                           sampling_rate=fs, resampling_factor=resampling_factor,
                           control_space=control_space, n_channels=n_channels, device='cpu')

  mono = ds._audio.float().cpu().sum(dim=0)  # the same downmix as AudioFeatureDataset._extract_features
  T = int(mono.shape[-1])
  features, column = [], 0
  for name, extractor, norm in ds._extractor_specs:
    if not isinstance(extractor, LibrosaFeatureExtractor):
      raise RuntimeError(f"control '{name}': only LibrosaFeatureExtractor is supported, got {type(extractor).__name__}")
    if not extractor._postprocess:
      print(f"note: control '{name}' has postprocess=false, but training normalises it anyway")
    frames = torch.tensor(extractor._feature_fn(y=mono.numpy(), center=False), dtype=torch.float32)
    if frames.ndim == 1:
      frames = frames.unsqueeze(0)
    if frames.shape[0] != 1:
      raise RuntimeError(f"control '{name}' has {frames.shape[0]} channels; only 1 is supported")
    x = F.interpolate(frames.unsqueeze(0), size=T, mode='linear', align_corners=False).squeeze(0)  # [1, T]
    lo = float(_percentile_kth(x, LOW_PERCENTILE, dim=-1).item())
    hi = float(_percentile_kth(x, HIGH_PERCENTILE, dim=-1).item())

    entry = {
      "name": name,
      "function": extractor._feature_fn.__name__,
      "librosa_defaults": _defaults(extractor._feature_fn),
      "center": False,
      "low_percentile": LOW_PERCENTILE, "high_percentile": HIGH_PERCENTILE,
      "lo": lo, "hi": hi,
      "smoothing_kernel_size": int(extractor._smoothing_kernel_size),
      "normalization": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in (norm or {}).items()} or None,
    }

    if verify:
      # rebuild the cached training feature from (lo, hi) and compare
      y = normalize_feature(x, low=LOW_PERCENTILE, high=HIGH_PERCENTILE, dim=-1)
      del x
      if extractor._smoothing_kernel_size > 1:
        y = smoothen_feature(y, window_size=extractor._smoothing_kernel_size)
      y = y.squeeze(0)
      if 'mean' in norm and 'std' in norm:
        y = (y - float(norm['mean'])) / (float(norm['std']) + 1e-8)
      elif 'min' in norm and 'max' in norm:
        y = (y - float(norm['min'])) / (float(norm['max']) - float(norm['min']) + 1e-8)
      cached = ds._features[:, column].cpu()
      n = min(len(cached), len(y))
      err = float((cached[:n] - y[:n]).abs().max().item())
      entry["max_abs_error_vs_training"] = err
      print(f"  {name}: lo={lo:.6g} hi={hi:.6g}  max |error| vs cached training feature = {err:.2e}")
      if err > 1e-4:
        raise RuntimeError(f"control '{name}' does not reproduce the training feature (error {err})")
    features.append(entry)
    column += 1

  return {
    "version": 1,
    "fs": fs,
    "n_channels": n_channels,
    "downmix": "sum",
    "resampling_factor": resampling_factor,
    "dataset_seconds": round(T / fs, 1),
    "features": features,
    "latents": {"method": "encode", "dim": sum(int(f.dim) for f in control_space.fields if f.source == 'latent')},
    "decode_inputs": [f["name"] for f in features] + [f"latent_{i}" for i in range(
      sum(int(f.dim) for f in control_space.fields if f.source == 'latent'))],
  }


def write_control_stats(config_path: str, json_path: str, verify: bool = True) -> Dict[str, Any]:
  cfg = OmegaConf.load(config_path)
  print(f"control stats for {cfg.experiment.name} ({cfg.data.dataset_path})")
  stats = compute_control_stats(cfg, verify=verify)
  data = {}
  if os.path.exists(json_path):
    with open(json_path) as f:
      data = json.load(f)
  data["controls"] = stats
  tmp = json_path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(data, f)
  os.replace(tmp, json_path)
  print(f"wrote controls -> {json_path}")
  return stats


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--config", required=True, help="the training config (yaml)")
  ap.add_argument("--json", required=True, help="the model's JSON sidecar (created if missing)")
  ap.add_argument("--no_verify", dest="verify", action="store_false", default=True,
                  help="skip the comparison with the cached training features")
  a = ap.parse_args()
  write_control_stats(a.config, a.json, verify=a.verify)
