"""Render percussion performances from the CC0 Versilian Community Sample Library (VCSL).

Every sound is a recorded acoustic instrument (VCSL, CC0); the grooves come from this script (style
templates with per-bar variation, fills, swing, humanised timing and velocity, dynamics that rise and
fall), so the output is CC0 as well. Kits:

  kit     kick, snares, toms, hi-hat, cymbals
  hands   frame drum, darbuka, conga, bongos, cajon, shakers, tambourine, claves
  metal   anvil, brake drum, slit drum, cowbells, agogo, gongs, triangles, ratchet

Each take goes through a synthetic room; --field adds a gritty chain (tape saturation, band limit,
noise floor, crackle). Writes 48 kHz mono WAVs and a manifest.

  python -m cli.render_percussion --vcsl .../VCSL --out .../percussion_audition --takes 12 --seconds 30
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt

FS = 48000
S = "Membranophones/Struck Membranophones/"
I = "Idiophones/Struck Idiophones/"

# role -> list of (instrument folder, articulation regex) alternatives
KITS = {
    "kit": {
        "kick": [(S + "Bass Drum 1", "hit"), (S + "Bass Drum 2", "")],
        "snare": [(S + "Snare Drum, Modern 1", "Hit"), (S + "Snare Drum, Modern 2", "Hit"), (S + "Snare Drum, Modern 3", "Hit"),
                  (S + "Snare Drum, Rope Tension", "")],
        "ghost": [(S + "Snare Drum, Modern 1", "Hit"), (S + "Snare Drum, Modern 2", "Hit")],
        "hat": [(I + "Hi-Hat Cymbal", "HitC|Close")],
        "open": [(I + "Hi-Hat Cymbal", "HitO|Loose")],
        "tom": [(S + "Tom 1", "Hit"), (S + "Tom 2", "Hit")],
        "crash": [(I + "Suspended Cymbal 1", "hit"), (I + "Suspended Cymbal 2", "hit"), (I + "Clash Cymbals 1", "crash")],
    },
    "hands": {
        "kick": [(S + "Frame Drum", "Hit|Hand"), (I + "Cajon", "Bass|Low|Hit")],
        "snare": [(S + "Darbuka", ""), (I + "Cajon", "Slap|High|Snare")],
        "ghost": [(S + "Bongos", ""), (S + "Darbuka", "")],
        "hat": [(I + "Shaker, Small", ""), (I + "Shaker, Large", ""), (I + "Cabasa", "")],
        "open": [(I + "Tambourine 1", ""), (I + "Tambourine 2", "")],
        "tom": [(S + "Conga", ""), (S + "Bongos", "")],
        "crash": [(I + "Claves", ""), (I + "Woodblock", "")],
    },
    "metal": {
        "kick": [(I + "Slit Drum", ""), (S + "Timpani 1", "")],
        "snare": [(I + "Anvil", ""), (I + "Brake Drum", "")],
        "ghost": [(I + "Cowbells", ""), (I + "Agogo Bells", "")],
        "hat": [(I + "Triangles", ""), (I + "Finger Cymbals", ""), (I + "Sleigh Bells", "")],
        "open": [(I + "Finger Cymbals", ""), (I + "Hand Bells, Nepalese", "")],
        "tom": [(I + "Slit Drum", ""), (I + "Brake Drum", "")],
        "crash": [(I + "Gong 1", ""), (I + "Gong 2", "")],
    },
}

# 16-step probability grids per role (one bar of 4/4); triplet styles use 12 steps
STYLES = {
    "straight": dict(steps=16, kick=[.95, 0, .1, 0, 0, 0, .15, 0, .8, 0, .45, 0, 0, .1, .2, 0],
                     snare=[0, 0, 0, 0, .97, 0, 0, 0, 0, 0, 0, 0, .97, 0, 0, .1],
                     ghost=[0, .1, 0, .2, 0, 0, .25, .1, 0, .2, 0, .15, 0, 0, .25, 0],
                     hat=[.95, .3, .9, .3] * 4, open=[0, 0, 0, 0, 0, 0, 0, .15, 0, 0, 0, 0, 0, 0, 0, .3]),
    "halftime": dict(steps=16, kick=[.95, 0, 0, .2, 0, 0, .3, 0, 0, .4, .3, 0, 0, 0, .2, 0],
                     snare=[0, 0, 0, 0, 0, 0, 0, 0, .97, 0, 0, 0, 0, 0, 0, 0],
                     ghost=[0, 0, .15, 0, .2, 0, 0, .15, 0, 0, .2, 0, .25, .15, 0, .2],
                     hat=[.9, .2, .7, .2] * 4, open=[0] * 14 + [.25, 0]),
    "broken": dict(steps=16, kick=[.95, 0, 0, .35, 0, 0, .5, 0, 0, .55, 0, 0, .3, 0, .4, 0],
                   snare=[0, 0, 0, 0, .95, 0, 0, .2, 0, 0, 0, .25, .9, 0, 0, 0],
                   ghost=[0, .3, .2, 0, 0, .35, 0, 0, .3, 0, .35, 0, 0, .3, 0, .3],
                   hat=[.9, .5, .8, .6] * 4, open=[0, 0, 0, 0, 0, 0, .2, 0, 0, 0, 0, 0, 0, 0, .2, 0]),
    "shuffle": dict(steps=12, kick=[.95, 0, .2, 0, 0, 0, .7, 0, .35, 0, 0, .15],
                    snare=[0, 0, 0, .95, 0, 0, 0, 0, 0, .95, 0, 0],
                    ghost=[0, .25, .2, 0, .3, .2, 0, .2, .3, 0, .3, .25],
                    hat=[.95, 0, .8, .95, 0, .8, .95, 0, .8, .95, 0, .8], open=[0] * 11 + [.2]),
    "sixeight": dict(steps=12, kick=[.95, 0, 0, 0, 0, .2, .7, 0, 0, .2, 0, 0],
                     snare=[0, 0, 0, .8, 0, 0, 0, 0, 0, .8, 0, .2],
                     ghost=[0, .4, .3, 0, .4, .3, 0, .4, .3, 0, .4, .3],
                     hat=[.9, .6, .6] * 4, open=[0, 0, 0, 0, 0, 0, .2, 0, 0, 0, 0, 0]),
}


# articulations that are not single strikes (swells, rolls, bowing, scrapes, shakes)
NOT_A_HIT = re.compile(r"bow|cresc|roll|scrape|crank|slow|fast|swell|trem|shake|taps|rub|gliss|buzz", re.I)
DYNAMICS = {"ppp": 1, "pp": 2, "p": 3, "mp": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8}
MAX_S = dict(kick=1.2, snare=0.9, ghost=0.4, hat=0.3, open=0.9, tom=1.1, crash=2.5)
# kit drums are damped (concert bass drums and orchestral toms ring for seconds otherwise)
DAMP_S = dict(kick=0.16, snare=0.22, ghost=0.12, tom=0.3)


class Instrument:
  """One VCSL instrument folder: its samples by velocity layer, one microphone."""

  def __init__(self, folder: str, art: str):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))
    files = [f for f in files if not NOT_A_HIT.search(os.path.basename(f))]
    if art:
      matching = [f for f in files if re.search(art, os.path.basename(f), re.I)]
      files = matching or files
    if not files:
      raise FileNotFoundError(folder)
    mics = {}
    for f in files:
      m = re.search(r"_([A-Za-z]+)\.wav$", f)
      mics.setdefault(m.group(1) if m else "", []).append(f)
    pref = [m for m in ("Sum", "Mid", "Main", "Close", "") if m in mics]
    files = mics[pref[0]] if pref else max(mics.values(), key=len)
    self.layers = {}
    for f in files:
      base = os.path.basename(f)
      v = re.search(r"_v(\d+)", base)
      d = re.search(r"_(ppp|pp|p|mp|mf|f|ff|fff)\d*(?:_|\.wav)", base)
      level = int(v.group(1)) if v else DYNAMICS[d.group(1)] if d else 1
      self.layers.setdefault(level, []).append(f)
    self.keys = sorted(self.layers)
    if not self.keys:
      raise FileNotFoundError(folder)
    # one gain for the whole instrument (its loudest sample at 1), so velocity layers keep their levels
    self.cache = {f: self._load(f) for fs in self.layers.values() for f in fs}
    peak = max(float(np.abs(x).max()) for x in self.cache.values()) + 1e-9
    self.cache = {f: x / peak for f, x in self.cache.items()}

  @staticmethod
  def _load(f: str) -> np.ndarray:
    x, sr = sf.read(f, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != FS:
      import librosa
      x = librosa.resample(x, orig_sr=sr, target_sr=FS)
    return x

  def sample(self, vel: float, rng) -> np.ndarray:
    k = self.keys[min(len(self.keys) - 1, int(vel * len(self.keys)))]
    return self.cache[self.layers[k][rng.integers(len(self.layers[k]))]]


def room_ir(rng, rt60: float, pre_ms: float = 8.0) -> np.ndarray:
  n = int(FS * rt60 * 1.2)
  t = np.arange(n) / FS
  ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
  sos = butter(2, [200, 7000], "bandpass", fs=FS, output="sos")
  ir = sosfilt(sos, ir)
  ir = np.concatenate([np.zeros(int(FS * pre_ms / 1000)), ir])
  return ir / np.sqrt((ir ** 2).sum())


def field_chain(x: np.ndarray, rng) -> np.ndarray:
  x = np.tanh(x * 2.2) / np.tanh(2.2)                                  # tape saturation
  sos = butter(2, [70, rng.uniform(5000, 9000)], "bandpass", fs=FS, output="sos")
  x = sosfilt(sos, x)
  noise = np.cumsum(rng.standard_normal(len(x))) ; noise -= np.convolve(noise, np.ones(4800) / 4800, "same")
  noise = sosfilt(butter(1, 200, "highpass", fs=FS, output="sos"), noise)
  x = x + noise / (np.abs(noise).max() + 1e-9) * 10 ** (-38 / 20)    # noise floor
  clicks = np.zeros(len(x)); idx = rng.integers(0, len(x), int(len(x) / FS * rng.uniform(2, 9)))
  clicks[idx] = rng.uniform(-1, 1, len(idx)) * 10 ** (-24 / 20)
  return x + sosfilt(butter(2, 2500, "highpass", fs=FS, output="sos"), clicks)  # crackle


def render_take(kit: dict, style: str, seconds: float, rng, field: bool, kit_name: str = "") -> tuple[np.ndarray, dict]:
  st = STYLES[style]
  steps = st["steps"]
  bpm = rng.uniform(72, 132) if steps == 16 else rng.uniform(60, 104)
  step_s = 60.0 / bpm / (4 if steps == 16 else 3)
  swing = rng.uniform(0.0, 0.18) if steps == 16 else 0.0
  bars = int(np.ceil(seconds / (step_s * steps)))
  out = np.zeros(int((bars * steps * step_s + 4) * FS), dtype=np.float32)
  voices = {}
  for role, alts in kit.items():
    for folder, art in rng.permutation(np.array(alts, dtype=object)):
      try:
        voices[role] = Instrument(folder, art)
        break
      except FileNotFoundError:
        continue
  gains = {r: 10 ** (db / 20) for r, db in dict(kick=0, snare=-2, ghost=-12, hat=-14, open=-11, tom=-4, crash=-10).items()}
  intensity = rng.uniform(0.5, 1.0)
  for bar in range(bars):
    # dynamics drift; every 4th / 8th bar ends with a fill
    intensity = float(np.clip(intensity + rng.normal(0, 0.12), 0.35, 1.0))
    fill = (bar % 8 == 7 and rng.random() < 0.8) or (bar % 4 == 3 and rng.random() < 0.3)
    density = rng.uniform(0.8, 1.2)
    for s in range(steps):
      t = (bar * steps + s) * step_s + (swing * step_s if (steps == 16 and s % 2 == 1) else 0.0)
      hits = []
      for role in ("kick", "snare", "ghost", "hat", "open"):
        p = st[role][s] * (density if role in ("ghost", "hat") else 1.0)
        if fill and s >= steps // 2 and role in ("hat", "ghost"):
          p *= 0.3
        if rng.random() < p:
          hits.append(role)
      if fill and s >= steps // 2 and rng.random() < 0.6:
        hits.append("tom")
      if bar % 8 == 0 and s == 0 and bar > 0 and rng.random() < 0.7:
        hits.append("crash")
      for role in hits:
        if role not in voices:
          continue
        vel = np.clip(intensity * (0.55 if role == "ghost" else 1.0) * rng.uniform(0.75, 1.0), 0.05, 1.0)
        x = voices[role].sample(vel, rng)
        if kit_name == "kit" and role in DAMP_S:
          x = x * np.exp(-np.arange(len(x)) / (DAMP_S[role] * FS))
        n = int(MAX_S[role] * FS)
        if len(x) > n:   # cap the tail, with a 40 ms fade
          x = x[:n].copy()
          x[-1920:] *= np.linspace(1.0, 0.0, 1920)
        lag = rng.normal(0.0, 0.006) + (0.008 if role == "snare" else 0.0)    # laid-back snare
        i0 = max(0, int((t + lag) * FS))
        x = x[: len(out) - i0] * gains[role] * (0.6 + 0.4 * vel)
        out[i0:i0 + len(x)] += x
  out = out[: int(seconds * FS)]
  rt60 = rng.uniform(0.25, 0.9)
  wet = fftconvolve(out, room_ir(rng, rt60))[: len(out)]
  mix = out + wet * rng.uniform(0.06, 0.2)
  if field:
    mix = field_chain(mix / (np.abs(mix).max() + 1e-9) * 0.7, rng)
  mix = mix / (np.abs(mix).max() + 1e-9) * 0.9
  return mix.astype(np.float32), dict(bpm=round(bpm, 1), swing=round(swing, 3), rt60=round(rt60, 2),
                                      voices={r: os.path.basename(os.path.dirname(v.layers[v.keys[0]][0])) for r, v in voices.items()})


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--vcsl", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--takes", type=int, default=12)
  ap.add_argument("--seconds", type=float, default=30.0)
  ap.add_argument("--kits", default="kit,hands,metal")
  ap.add_argument("--field", choices=["off", "on", "both"], default="both")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--max-seconds", type=float, default=0.0, help="vary each take's length between --seconds and this")
  args = ap.parse_args()
  rng = np.random.default_rng(args.seed)
  os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
  kits = {k: {r: [(os.path.join(args.vcsl, f), a) for f, a in alts] for r, alts in KITS[k].items()} for k in args.kits.split(",")}
  styles = {"kit": ["straight", "halftime", "broken", "shuffle"], "hands": ["sixeight", "broken", "shuffle", "straight"],
            "metal": ["broken", "sixeight", "halftime"]}
  with open(os.path.join(args.out, "manifest.jsonl"), "w") as man:
    for n in range(args.takes):
      kit = list(kits)[n % len(kits)]
      style = styles[kit][(n // len(kits)) % len(styles[kit])]
      field = args.field == "on" or (args.field == "both" and (n // len(kits)) % 2 == 1)
      seconds = rng.uniform(args.seconds, args.max_seconds) if args.max_seconds > args.seconds else args.seconds
      x, info = render_take(kits[kit], style, seconds, rng, field, kit)
      name = f"{n:03d}_{kit}_{style}{'_field' if field else ''}.wav"
      sf.write(os.path.join(args.out, "audio", name), x, FS, subtype="PCM_24")
      man.write(json.dumps(dict(file="audio/" + name, kit=kit, style=style, field=field, **info,
                                licence="CC0 (VCSL samples, generated performance)",
                                credit="Versilian Community Sample Library (CC0), performance generated by cli/render_percussion.py")) + "\n")
      print(name, info, flush=True)


if __name__ == "__main__":
  main()
