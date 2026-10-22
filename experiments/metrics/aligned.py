"""Time-aligned reconstruction measures: the original and its reconstruction compared frame by frame, at the full
bandwidth (no embedding model in between), each aimed at one thing heard in the listening tests.

Bands: 48 log-spaced bands from 40 Hz to 24 kHz on a 2048-point STFT with a 128-sample hop (2.7 ms at 48 kHz),
level in dB, floored 80 dB below the original's loudest band frame (so inaudible differences don't count).

  env20, env100   mean |level difference| per band over time, envelopes smoothed over 20 ms / 100 ms, bands
                  weighted toward the ones the original uses (log of the energy share); dB
  hf20            env20 over the bands above 4 kHz only; dB
  dyncorr         1 - mean over bands of the correlation between the two 20 ms envelopes (does the level move
                  when the original's does: the mush)
  attack          how much less the level rises at the original's onsets (20 ms before vs after), dB, >= 0
  tail_cut        how much quieter the decays after onsets are (10-400 ms after the peak), dB, >= 0
  tail_long       how much louder they are (smeared or held tails, hiss under the decay), dB, >= 0
  floor           rise of the 10th percentile level, log-energy weighted (as in the earlier score), dB, >= 0
Lower is better for all.

  python experiments/metrics/aligned.py ORIG.wav RECON.wav
"""

import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

N_FFT, HOP, N_BANDS, FMIN, RANGE_DB = 2048, 128, 48, 40.0, 80.0


def _load(path, fs=None):
  x, sr = sf.read(path, dtype="float32", always_2d=True)
  x = x.mean(1)
  if fs and sr != fs:
    x = librosa.resample(x, orig_sr=sr, target_sr=fs)
    sr = fs
  return x, sr


def band_power(x, fs):
  S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP, center=True)) ** 2
  f = librosa.fft_frequencies(sr=fs, n_fft=N_FFT)
  edges = np.geomspace(FMIN, fs / 2, N_BANDS + 1)
  P = np.stack([S[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(edges[:-1], edges[1:])])
  return P + 1e-12, np.sqrt(edges[:-1] * edges[1:])


def _db(P, floor):
  return np.maximum(10 * np.log10(P), floor)


def scores(orig, recon):
  xo, fs = _load(orig)
  xr, _ = _load(recon, fs)
  n = min(len(xo), len(xr))
  Po, fc = band_power(xo[:n], fs)
  Pr, _ = band_power(xr[:n], fs)
  fr = fs / HOP
  floor = 10 * np.log10(Po.max()) - RANGE_DB
  sm = lambda P, ms: uniform_filter1d(P, max(1, int(ms / 1000 * fr)), axis=1)

  share = Po.mean(1) / Po.mean(1).sum()
  w = np.log1p(share * N_BANDS); w = w / w.sum()                       # band weights, log of energy share
  out = {}
  for ms in (20, 100):
    Do, Dr = _db(sm(Po, ms), floor), _db(sm(Pr, ms), floor)
    out[f"env{ms}"] = float((np.abs(Dr - Do).mean(1) * w).sum())
    if ms == 20:
      hf = fc >= 4000
      out["hf20"] = float((np.abs(Dr - Do)[hf].mean(1) * w[hf]).sum() / max(w[hf].sum(), 1e-9))
      c = []
      for b in range(N_BANDS):
        a, r = Do[b] - Do[b].mean(), Dr[b] - Dr[b].mean()
        d = np.sqrt((a * a).sum() * (r * r).sum())
        c.append((a * r).sum() / d if d > 0 else 0.0)
      out["dyncorr"] = float(1 - (np.array(c) * w).sum())

  # onsets and decays on the broadband level (5 ms smoothing)
  Bo = _db(sm(Po.sum(0, keepdims=True), 5), floor)[0]
  Br = _db(sm(Pr.sum(0, keepdims=True), 5), floor)[0]
  on = librosa.onset.onset_detect(y=xo[:n], sr=fs, hop_length=HOP, backtrack=False, units="frames")
  k20, k10, k400 = int(0.02 * fr), int(0.01 * fr), int(0.4 * fr)
  att, cut, lng = [], [], []
  for i, t in enumerate(on):
    if t - k20 < 0 or t + k20 >= len(Bo):
      continue
    rise = lambda B: B[t:t + k20].max() - B[t - k20:t].min()
    att.append(max(rise(Bo) - rise(Br), 0.0))
    pk = t + int(np.argmax(Bo[t:t + k20]))
    end = min(pk + k400, on[i + 1] if i + 1 < len(on) else len(Bo), len(Bo))
    seg = slice(pk + k10, end)
    if end - (pk + k10) < k10:
      continue
    live = Bo[seg] > floor + 6                                          # where the original's tail is audible
    if live.any():
      d = (Br[seg] - Bo[seg])[live]
      cut.append(np.maximum(-d, 0).mean()); lng.append(np.maximum(d, 0).mean())
  out["attack"] = float(np.mean(att)) if att else 0.0
  out["tail_cut"] = float(np.mean(cut)) if cut else 0.0
  out["tail_long"] = float(np.mean(lng)) if lng else 0.0
  out["n_onsets"] = len(att)

  # noise floor as in the earlier score (mel-free version on the same bands)
  Do, Dr = _db(sm(Po, 20), floor), _db(sm(Pr, 20), floor)
  lo, rlo = np.percentile(Do, 10, 1), np.percentile(Dr, 10, 1)
  out["floor"] = float((np.maximum(rlo - lo, 0) * w).sum())
  return out


if __name__ == "__main__":
  print(scores(sys.argv[1], sys.argv[2]))
