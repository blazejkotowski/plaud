"""Score the style experiments' passage reconstructions (eval_passages.py), and combine.

Per passage and take (results/<run>/e<epoch>/pNN_{orig,recon}.wav):
  tex    log EnCodec-48k FAD between the original and the reconstruction (sound of the material)
  range  |dynamic range difference| per mel band (90th - 10th percentile level), weighted by the
         original's band power, dB (the "mush")
  floor  rise of the 10th percentile level above the original's, log-power weighted, dB (the hiss)
The combined score standardises each of the three per passage over all takes, sums them and
averages over passages (lower is better). It only ranks the takes scored together. On the data-size
listening test it ordered all 23 judged pairs as heard (experiments/datasize).

Runs in the `percept` env (fadtk); caches per take in <results>/scores.json.
  python experiments/styles/score_passages.py --results experiments/styles/results
"""

import argparse
import glob
import json
import logging
import math
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import linalg

logging.disable(logging.WARNING)


def _sf_load(path, *a, **kw):        # torchaudio 2.11 load() needs torchcodec; EnCodec's loader calls it
  x, sr = sf.read(str(path), dtype="float32", always_2d=True)
  return torch.from_numpy(x.T.copy()), sr


torchaudio.load = _sf_load
from fadtk.model_loader import get_all_models


def frechet(a, b):
  s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
  covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
  return float(((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def mel_db(x, fs):
  return 10 * np.log10(librosa.feature.melspectrogram(y=x, sr=fs, n_fft=2048, hop_length=512, n_mels=64, fmin=30) + 1e-10)


def dyn(orig, recon):
  o, fs = librosa.load(orig, sr=None)
  r, _ = librosa.load(recon, sr=fs)
  Mo, Mr = mel_db(o, fs), mel_db(r, fs)
  lo, hi, rlo, rhi = (np.percentile(M, q, 1) for M, q in ((Mo, 10), (Mo, 90), (Mr, 10), (Mr, 90)))
  w = 10 ** (np.percentile(Mo, 50, 1) / 10); w = w / w.sum()
  wl = np.log1p(w * len(w)); wl = wl / wl.sum()
  return float((np.abs((hi - lo) - (rhi - rlo)) * w).sum()), float((np.maximum(rlo - lo, 0) * wl).sum())


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--results", required=True)
  args = ap.parse_args()
  cache_path = os.path.join(args.results, "scores.json")
  cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

  todo = [r for r in sorted(glob.glob(os.path.join(args.results, "*", "*", "p*_recon.wav")))
          if os.path.relpath(r, args.results) not in cache]
  if todo:
    model = {m.name: m for m in get_all_models()}["encodec-emb-48k"]
    model.load_model()
    emb = {}
    E = lambda p: emb.setdefault(p, np.asarray(model.get_embedding(model.load_wav(p)), dtype=np.float64).reshape(-1, model.num_features))
    for r in todo:
      o = r.replace("_recon.wav", "_orig.wav")
      rng, floor = dyn(o, r)
      cache[os.path.relpath(r, args.results)] = dict(tex=math.log(frechet(E(o), E(r))), range=rng, floor=floor)
    json.dump(cache, open(cache_path, "w"), indent=1)

  # combine: per passage, z over all takes; then mean over passages
  takes = sorted({os.path.dirname(k) for k in cache})
  passages = sorted({os.path.basename(k)[:3] for k in cache})
  table = {t: dict(tex=[], range=[], floor=[], combined=[]) for t in takes}
  for p in passages:
    keys = {t: f"{t}/{p}_recon.wav" for t in takes if f"{t}/{p}_recon.wav" in cache}
    z = {}
    for m in ("tex", "range", "floor"):
      v = np.array([cache[k][m] for k in keys.values()])
      for t, k in keys.items():
        z.setdefault(t, 0.0)
        z[t] += (cache[k][m] - v.mean()) / (v.std() + 1e-9)
        table[t][m].append(cache[k][m])
    for t in keys:
      table[t]["combined"].append(z[t])
  summary = {t: {m: float(np.mean(v)) for m, v in d.items()} | dict(n=len(d["combined"])) for t, d in table.items()}
  json.dump(summary, open(os.path.join(args.results, "summary.json"), "w"), indent=1)
  print(f"{'take':34s} {'combined':>9s} {'tex':>7s} {'range':>7s} {'floor':>7s}  n")
  for t, s in sorted(summary.items(), key=lambda kv: kv[1]["combined"]):
    print(f"{t:34s} {s['combined']:9.2f} {s['tex']:7.2f} {s['range']:7.2f} {s['floor']:7.2f}  {s['n']}")


if __name__ == "__main__":
  main()
