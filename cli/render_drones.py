"""Render slow drones from CC0 sustained-instrument samples (VSCO 2 Community Edition, VCSL).

Each take is a slow progression of mellow voicings (fifths, sus2, minor 7, add9, open octaves) held by
one or two instrument families: organ, string sections (sustain with vibrato), bowed psaltery, wine
glasses, harmonicas. Notes are the nearest sample re-pitched by up to a few semitones; tones are
extended by crossfade-looping the sustain; chords crossfade over several seconds; a long synthetic
hall goes on top. Sample pitch comes from the file name when it has a note, else from pYIN.

  python -m cli.render_drones --vsco .../VSCO-2-CE --vcsl .../VCSL --out .../drone_audition --takes 8 --seconds 45
"""

import argparse
import glob
import json
import os
import re

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

FS = 48000
NOTE = re.compile(r"(?<![A-Za-z])([A-G])(#|b)?(-?\d)(?![\d])")

FAMILIES = {
    "organ": ["VSCO/Keys/Organ/Quiet", "VSCO/Keys/Organ/Loud"],
    "strings": ["VSCO/Strings/Cello Section/susvib", "VSCO/Strings/Viola Section/susvib", "VSCO/Strings/Violin Section/susVib",
                "VSCO/Strings/Solo Contrabass/SusNV"],
    "psaltery": ["VCSL/Chordophones/Zithers/Psaltery, Bowed and Plucked:LongBow"],
    "glass": ["VCSL/Idiophones/Friction Idiophones/Wine Glasses"],
    "harmonica": ["VCSL/Aerophones/Free Aerophones/Harmonica-Hohner-Special20-C", "VCSL/Aerophones/Free Aerophones/Harmonica-Hohner-Super64"],
}
VOICINGS = [[0, 7, 12], [0, 7, 14], [0, 2, 7, 12], [0, 3, 10, 14], [0, 5, 7, 12], [0, 7, 12, 16], [0, 4, 11, 14], [0, 12, 19]]
PROGRESSIONS = [[0, 5, 0, 7], [0, -2, -4, -2], [0, 3, 5, 3], [0, 0, 5, 5], [0, -5, -3, 0], [0, 2, 0, -2]]


def midi_of(path: str, x: np.ndarray):
  m = NOTE.search(os.path.basename(path).replace("_", " "))
  if m:
    name = m.group(1) + (m.group(2) or "") + m.group(3)
    try:
      return float(librosa.note_to_midi(name))
    except Exception:
      pass
  seg = x[int(0.3 * FS): int(2.3 * FS)]
  if len(seg) < FS // 2:
    return None
  f0, voiced, _ = librosa.pyin(seg[::2], fmin=40, fmax=2000, sr=FS // 2)
  f0 = f0[voiced] if voiced.any() else f0[~np.isnan(f0)]
  return float(librosa.hz_to_midi(np.median(f0))) if len(f0) else None


class Family:
  def __init__(self, roots: dict, folders: list):
    self.notes = []   # (midi, samples)
    for spec in folders:
      lib, rest = spec.split("/", 1)
      folder, _, art = rest.partition(":")
      files = sorted(glob.glob(os.path.join(roots[lib], folder, "**", "*.wav"), recursive=True))
      if art:
        files = [f for f in files if art.lower() in os.path.basename(f).lower()]
      for f in files:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != FS:
          x = librosa.resample(x, orig_sr=sr, target_sr=FS)
        if len(x) < 2 * FS:
          continue
        m = midi_of(f, x)
        if m is not None:
          self.notes.append((m, x / (np.abs(x).max() + 1e-9), f))
    if not self.notes:
      raise FileNotFoundError(folders)
    self.range = (min(n[0] for n in self.notes), max(n[0] for n in self.notes))

  def tone(self, midi: float, seconds: float, rng) -> np.ndarray:
    m, x, _ = min(self.notes, key=lambda n: abs(n[0] - midi) + rng.uniform(0, 0.5))
    ratio = 2 ** ((midi - m) / 12)                 # re-pitch by resampling
    up, down = int(round(1000 / ratio)), 1000
    x = resample_poly(x, up, down).astype(np.float32)
    # sustain loop: from 25 % to 85 % of the sample, crossfaded by 0.8 s
    a, b = int(len(x) * 0.25), int(len(x) * 0.85)
    loop, xf = x[a:b], int(0.8 * FS)
    if len(loop) < 2 * xf:
      return np.resize(x, int(seconds * FS))
    out = x[:b].copy()
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    while len(out) < seconds * FS:
      out[-xf:] = out[-xf:] * (1 - ramp) + loop[:xf] * ramp
      out = np.concatenate([out, loop[xf:]])
    return out[: int(seconds * FS)].astype(np.float32)


def hall(rng, rt60: float) -> np.ndarray:
  n = int(FS * rt60 * 1.3)
  t = np.arange(n) / FS
  ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
  ir = sosfilt(butter(2, [120, 5500], "bandpass", fs=FS, output="sos"), ir)
  return ir / np.sqrt((ir ** 2).sum())


def render_take(fams: dict, names: list, seconds: float, rng) -> tuple[np.ndarray, dict]:
  lo = max(fams[n].range[0] for n in names)
  hi = min(fams[n].range[1] for n in names)
  root = float(rng.integers(int(lo) + 2, max(int(lo) + 3, int(min(hi - 14, lo + 18)))))
  prog = PROGRESSIONS[rng.integers(len(PROGRESSIONS))]
  chord_s = rng.uniform(10, 18)
  xfade = min(chord_s * 0.45, rng.uniform(4, 7))
  n_chords = int(np.ceil(seconds / chord_s)) + 1
  out = np.zeros(int((seconds + chord_s + 10) * FS), dtype=np.float32)
  used = []
  for c in range(n_chords):
    voicing = VOICINGS[rng.integers(len(VOICINGS))]
    base = root + prog[c % len(prog)]
    t0 = max(0.0, c * chord_s - xfade)
    dur = chord_s + 2 * xfade
    for k, interval in enumerate(voicing):
      fam = fams[names[k % len(names)]]
      midi = base + interval
      while midi > fam.range[1]:
        midi -= 12
      while midi < fam.range[0]:
        midi += 12
      x = fam.tone(midi, dur, rng) * (0.8 if k == 0 else 0.55)
      env = np.ones(len(x), dtype=np.float32)
      fi = int((xfade + rng.uniform(0, 1.5)) * FS)
      env[:fi] = np.linspace(0, 1, fi) ** 1.5
      env[-int(xfade * FS):] = np.linspace(1, 0, int(xfade * FS)) ** 1.5
      i0 = int(t0 * FS)
      out[i0:i0 + len(x)] += (x * env)[: len(out) - i0]
    used.append(dict(at=round(c * chord_s, 1), root=base, voicing=voicing))
  out = out[: int(seconds * FS)]
  rt60 = rng.uniform(3.0, 6.0)
  wet = fftconvolve(out, hall(rng, rt60))[: len(out)]
  mix = out * 0.7 + wet * rng.uniform(0.35, 0.6)
  fade = int(3 * FS)
  mix[:fade] *= np.linspace(0, 1, fade)
  mix[-fade:] *= np.linspace(1, 0, fade)
  mix = mix / (np.abs(mix).max() + 1e-9) * 0.8
  return mix.astype(np.float32), dict(families=names, rt60=round(rt60, 1), chord_s=round(chord_s, 1), chords=used)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--vsco", required=True)
  ap.add_argument("--vcsl", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--takes", type=int, default=8)
  ap.add_argument("--seconds", type=float, default=45.0)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  rng = np.random.default_rng(args.seed)
  roots = {"VSCO": args.vsco, "VCSL": args.vcsl}
  fams = {}
  for name, folders in FAMILIES.items():
    try:
      fams[name] = Family(roots, folders)
      print(f"{name}: {len(fams[name].notes)} notes, MIDI {fams[name].range[0]:.0f}-{fams[name].range[1]:.0f}", flush=True)
    except FileNotFoundError:
      print(f"{name}: no samples", flush=True)
  combos = [["organ"], ["strings"], ["organ", "strings"], ["psaltery"], ["glass", "strings"], ["harmonica", "organ"],
            ["strings", "psaltery"], ["glass"]]
  combos = [c for c in combos if all(n in fams for n in c)]
  os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
  with open(os.path.join(args.out, "manifest.jsonl"), "w") as man:
    for n in range(args.takes):
      names = combos[n % len(combos)]
      x, info = render_take(fams, names, args.seconds, rng)
      name = f"{n:03d}_{'+'.join(names)}.wav"
      sf.write(os.path.join(args.out, "audio", name), x, FS, subtype="PCM_24")
      man.write(json.dumps(dict(file="audio/" + name, **info, licence="CC0 (VSCO 2 CE / VCSL samples, generated)",
                                credit="VSCO 2 Community Edition and Versilian Community Sample Library (CC0), drone generated by cli/render_drones.py")) + "\n")
      print(name, {k: v for k, v in info.items() if k != "chords"}, flush=True)


if __name__ == "__main__":
  main()
