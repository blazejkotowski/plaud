"""Perceptual scores for the data-size experiment's reconstructions (the *_orig.wav / *_recon.wav that
eval_recon.py writes), to find a metric that agrees with listening better than log-mel L1.

Per reconstruction and embedding model (fadtk loaders):
  cos   mean cosine distance between time-aligned frames of the original and the reconstruction
  fad   Frechet distance between the two passages' frame sets (frame-rich models only: EnCodec, MERT)
plus two time-aligned DSP scores, which FAD can't see (it compares frame sets, not their order):
  flux  1 - correlation of the positive spectral flux in 16 mel bands (onsets and transients: timing, attack)
  env   mean |difference| of the 16 mel-band log envelopes, dB (dynamics per band, 20 ms resolution)

Runs in the `percept` env (fadtk):
  python experiments/datasize/eval_percept.py --results experiments/datasize/results --out experiments/datasize/percept.json
"""

import argparse
import glob
import json
import logging
import os
import re

import librosa
import numpy as np
import torch
from scipy import linalg

logging.disable(logging.WARNING)

import soundfile as sf
import torchaudio
from fadtk.model_loader import get_all_models


def _sf_load(path, *a, **kw):        # torchaudio 2.11 load() needs torchcodec; EnCodec's loader calls it
  x, sr = sf.read(str(path), dtype="float32", always_2d=True)
  return torch.from_numpy(x.T.copy()), sr


torchaudio.load = _sf_load

MODELS = ["clap-laion-audio", "clap-laion-music", "MERT-v1-95M-4", "MERT-v1-95M-8", "MERT-v1-95M", "encodec-emb-48k"]
FAD_MODELS = {"MERT-v1-95M-4", "MERT-v1-95M-8", "MERT-v1-95M", "encodec-emb-48k"}


def frechet(a: np.ndarray, b: np.ndarray) -> float:
  mu1, mu2 = a.mean(0), b.mean(0)
  s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
  covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
  return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
  n = min(len(a), len(b))
  a, b = a[:n], b[:n]
  return float(np.mean(1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)))


def bands(x: np.ndarray, fs: int) -> np.ndarray:
  """[16, T] mel-band log energies, dB, 10 ms hop."""
  m = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=2048, hop_length=fs // 100, n_mels=16, fmin=30)
  return 10 * np.log10(m + 1e-8)


def aligned(orig: str, recon: str) -> dict:
  o, fs = librosa.load(orig, sr=None)
  r, _ = librosa.load(recon, sr=fs)
  bo, br = bands(o, fs), bands(r, fs)
  fo, fr = np.maximum(np.diff(bo, axis=1), 0), np.maximum(np.diff(br, axis=1), 0)
  corr = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(fo, fr) if a.std() > 0 and b.std() > 0])
  k = 2                                                             # 20 ms
  env = np.abs(bo[:, : bo.shape[1] // k * k].reshape(16, -1, k).mean(2) - br[:, : br.shape[1] // k * k].reshape(16, -1, k).mean(2))
  return {"flux": float(1 - corr), "env": float(env.mean())}


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--results", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  # one original per set (all evals of a set reconstruct the same core), every reconstruction
  recons = sorted(glob.glob(os.path.join(args.results, "*_recon.wav")))
  takes = []
  for r in recons:
    stem = os.path.basename(r)[: -len("_recon.wav")]
    m = re.match(r"(?:ds_)?(harbour|percussion|drone_sampled)_(full|s\d+)_e(\d+)$", stem)
    if m:
      takes.append(dict(stem=stem, set=m[1], size=m[2], epoch=int(m[3]), recon=r, orig=r.replace("_recon.wav", "_orig.wav")))
  out = {t["stem"]: dict(set=t["set"], size=t["size"], epoch=t["epoch"], **aligned(t["orig"], t["recon"])) for t in takes}

  if os.path.exists(args.out):                                    # resume: keep the models already scored
    for stem, v in json.load(open(args.out)).items():
      out.setdefault(stem, {}).update({k: x for k, x in v.items() if k not in out.get(stem, {})})
  models = {m.name: m for m in get_all_models()}
  for name in MODELS:
    if all(f"{name}/cos" in v for v in out.values()):
      continue
    model = models[name]
    model.load_model()
    cache = {}
    emb = lambda p: cache.setdefault(p, np.asarray(model.get_embedding(model.load_wav(p)), dtype=np.float64).reshape(-1, model.num_features))
    for t in takes:
      o, r = emb(t["orig"]), emb(t["recon"])
      out[t["stem"]][f"{name}/cos"] = cos_dist(o, r)
      if name in FAD_MODELS:
        out[t["stem"]][f"{name}/fad"] = frechet(o, r)
    print(name, "done", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    del model

  json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
  main()
