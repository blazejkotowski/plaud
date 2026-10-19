"""Turn Cassini RPWS wideband (WBR) waveforms into audio for training.

The Radio and Plasma Wave Science wideband receiver sampled the electric field directly: 8-bit
samples every 36 us in the 10 kHz band (27.8 kHz rate) and every 4.5 us in the 75 kHz band
(222 kHz). NASA PDS data are public domain. Volumes are served by the University of Iowa
(https://space.physics.uiowa.edu/pds/CORPWS_nnnn/), described in DOCUMENT/WBRWFR/WBRWFR.TXT.

Each 125 ms the receiver captures one record (usually 2048 samples, 74 ms at 10 kHz). Records are
joined with short raised-cosine crossfades (gaps squeezed out, as in NASA's own Cassini audio),
DC removed and gain compensated per record. The 10 kHz band is resampled to 48 kHz (true speed);
the 75 kHz band is played at 48 kHz, i.e. 4.6x slower and lower, which brings its content down
into the audible range. Each file is level-normalised.

  python -m cli.source_cassini --out /mnt/mariadata/datasets/factory/_raw/cassini \\
      --day 2004183 --day 2008072 --mb-per-day 30
"""

import argparse
import json
import os
import re
import struct
import urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = "https://space.physics.uiowa.edu/pds/"
FS_OUT = 48000
CAL_DB = {2: 6.33, 3: 6.43}                       # Calibration_Factor (dB) by band
GAIN_DB = [0, 10, 20, 30, 40, 50, 60, 70]         # analog gain steps (WBR)
CREDIT = ("Cassini Radio and Plasma Wave Science (RPWS) wideband data, University of Iowa / "
          "NASA Planetary Data System (PPI node). Public domain.")


def get(url: str) -> bytes:
  with urllib.request.urlopen(url, timeout=120) as r:
    return r.read()


def listing(url: str) -> list:
  try:
    return re.findall(r'href="([^"?/][^"]*)"', get(url).decode("utf-8", "replace"))
  except Exception:
    return []


def find_day(day: str, volumes: dict) -> list:
  """URLs of the WBR files of one day (yyyyddd), searching the volumes whose folder holds it."""
  folder = f"T{day[:5]}XX"
  urls = []
  for vol, folders in volumes.items():
    if folder not in folders:
      continue
    base = f"{ROOT}CORPWS_{vol}/DATA/RPWS_WIDEBAND_FULL/{folder}/T{day}/"
    urls += [base + f for f in listing(base) if f.endswith("_WBRFR.DAT")]
  return sorted(set(urls))


def scan_volumes(cache: str, last: int = 260) -> dict:
  if os.path.exists(cache):
    with open(cache) as f:
      return json.load(f)
  volumes = {}
  for v in range(1, last + 1):
    vol = f"{v:04d}"
    folders = sorted({f.strip("/") for f in listing(f"{ROOT}CORPWS_{vol}/DATA/RPWS_WIDEBAND_FULL/") if f.startswith("T")})
    if folders:
      volumes[vol] = folders
  with open(cache, "w") as f:
    json.dump(volumes, f)
  return volumes


def decode(data: bytes, fade: int = 128):
  """Records -> one waveform (float, physical-ish units), its band and the number of records."""
  rec_bytes = struct.unpack(">H", data[12:14])[0]
  n_rec = len(data) // rec_bytes
  out, band = [], None
  ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
  for i in range(n_rec):
    r = data[i * rec_bytes:(i + 1) * rec_bytes]
    samples = struct.unpack(">H", r[14:16])[0]
    b, gain = r[20] & 0x0F, r[21] & 0x07
    if b not in CAL_DB or samples < 4 * fade or 32 + samples > len(r):
      continue
    if band is None:
      band = b
    if b != band:
      continue
    x = np.frombuffer(r[32:32 + samples], dtype=np.uint8).astype(np.float32)
    if x.max() - x.min() < 2:                     # dead / saturated record
      continue
    x = (x - x.mean()) / 127.5 / (10.0 ** ((CAL_DB[b] + GAIN_DB[gain]) / 20.0) / np.sqrt(2.0))
    if out:
      prev = out[-1]
      prev[-fade:] = prev[-fade:] * ramp[::-1] + x[:fade] * ramp
      x = x[fade:]
    out.append(x)
  if not out:
    return None, band, 0
  return np.concatenate(out), band, len(out)


def to_audio(x: np.ndarray, band: int) -> np.ndarray:
  if band == 2:
    x = resample_poly(x, 216, 125)               # 27 777.8 Hz -> 48 kHz, true speed
  # band 3 (222 kHz) is written as-is at 48 kHz: 4.6x slower and lower
  x = x - np.mean(x)
  rms = np.sqrt(np.mean(x ** 2)) + 1e-12
  x = x * (10 ** (-20 / 20) / rms)                 # -20 dBFS RMS
  return np.tanh(x * 1.5) / 1.5                     # soft limit the rare spikes (dust impacts)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", required=True)
  ap.add_argument("--day", action="append", required=True, help="yyyyddd")
  ap.add_argument("--bands", default="10KHZ,75KHZ")
  ap.add_argument("--mb-per-day", type=float, default=30.0, help="per band; files spread over the day")
  args = ap.parse_args()

  os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
  volumes = scan_volumes(os.path.join(args.out, "volumes.json"))
  manifest = open(os.path.join(args.out, "items.jsonl"), "a")
  for day in args.day:
    urls = find_day(day, volumes)
    for band in args.bands.split(","):
      files = [u for u in urls if f"_{band}" in u]
      sizes = {}
      for u in files:   # sizes from the listing are rough; HEAD is exact
        req = urllib.request.Request(u, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
          sizes[u] = int(r.headers.get("Content-Length", 0))
      files = [u for u in files if sizes[u] > 200_000]
      budget = args.mb_per_day * 1e6
      # spread over the day: largest-first would cluster on one event
      step = max(1, int(np.ceil(sum(sizes[u] for u in files) / budget))) if files else 1
      used = 0
      for u in files[::step]:
        if used >= budget:
          break
        name = os.path.basename(u).replace(".DAT", ".wav")
        path = os.path.join(args.out, "wav", name)
        if os.path.exists(path):
          used += sizes[u]
          continue
        x, b, n = decode(get(u))
        if x is None or len(x) < FS_OUT:
          continue
        y = to_audio(x, b)
        sf.write(path, y.astype(np.float32), FS_OUT, subtype="PCM_24")
        used += sizes[u]
        item = dict(file=os.path.relpath(path, args.out), source=u, day=day, band=band, records=n,
                    seconds=round(len(y) / FS_OUT, 1), licence="Public domain (NASA PDS)", credit=CREDIT)
        manifest.write(json.dumps(item) + "\n")
        manifest.flush()
        print(f"{name}: {n} records, {len(y) / FS_OUT:.0f} s", flush=True)


if __name__ == "__main__":
  main()
