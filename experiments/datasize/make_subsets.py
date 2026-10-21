"""Nested training subsets for the data-size experiment (does PLAUD reconstruct better on less data?).

From a factory set (<root>/<set>/audio), cut 16 s excerpts (8 training chunks of 2 s each, so no chunk
straddles two excerpts): one per file, round-robin over the files (shuffled with --seed) and then again
at other positions until the largest subset is filled. Each excerpt is the 16 s window whose RMS is
closest to the file's median 16 s RMS (no silence, no outlier peak). Subsets are nested prefixes of that
list, so the first excerpt, `core`, is in every subset and in the full set, and every model can be
scored on the same audio.

  python experiments/datasize/make_subsets.py --set harbour --sizes 16,120,960 \
      --out /mnt/mariadata/datasets/research/datasize

writes <out>/<set>/s<seconds>/audio/NNN_<file>_<start>.wav and <out>/<set>/excerpts.json.
"""

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf

EXCERPT_S = 16.0


def candidates(x: np.ndarray, fs: int, hop_s: float = 2.0) -> list:
  """(start, rms) of every 16 s window on a 2 s grid, ordered by distance to the median rms."""
  n, h = int(EXCERPT_S * fs), int(hop_s * fs)
  mono = x.mean(1) if x.ndim == 2 else x
  starts = list(range(0, max(1, len(mono) - n + 1), h))
  rms = np.array([np.sqrt(np.mean(mono[s:s + n] ** 2)) for s in starts])
  med = np.median(rms)
  return [(starts[i], float(rms[i])) for i in np.argsort(np.abs(np.log(rms + 1e-9) - np.log(med + 1e-9)))]


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--root", default="/mnt/mariadata/datasets/factory")
  ap.add_argument("--set", required=True)
  ap.add_argument("--sizes", default="16,120,960", help="subset sizes in seconds (multiples of 16)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  sizes = sorted(int(s) for s in args.sizes.split(","))
  n_needed = max(sizes) // int(EXCERPT_S)
  rng = np.random.default_rng(args.seed)

  files = sorted(glob.glob(os.path.join(args.root, args.set, "audio", "*.wav")))
  rng.shuffle(files)
  audio, cands = {}, {}
  for f in files:
    x, fs = sf.read(f, dtype="float32")
    audio[f] = x
    cands[f] = candidates(x, fs)

  # round-robin over files; within a file take the next candidate that doesn't overlap a taken excerpt
  n = int(EXCERPT_S * fs)
  taken = {f: [] for f in files}
  excerpts = []
  while len(excerpts) < n_needed:
    progress = False
    for f in files:
      if len(excerpts) >= n_needed:
        break
      for start, rms in cands[f]:
        if all(abs(start - t) >= n for t in taken[f]):
          taken[f].append(start)
          excerpts.append(dict(file=f, start=start, rms_db=round(20 * np.log10(rms + 1e-9), 1)))
          progress = True
          break
    if not progress:
      raise SystemExit(f"{args.set}: only {len(excerpts)} excerpts fit, {n_needed} needed")

  base = os.path.join(args.out, args.set)
  for size in sizes:
    d = os.path.join(base, f"s{size}", "audio")
    os.makedirs(d, exist_ok=True)
    for k, e in enumerate(excerpts[: size // int(EXCERPT_S)]):
      stem = os.path.splitext(os.path.basename(e["file"]))[0][:40]
      sf.write(os.path.join(d, f"{k:03d}_{stem}_{e['start'] // fs}.wav"), audio[e["file"]][e["start"]:e["start"] + n], fs,
               subtype="PCM_24")
  with open(os.path.join(base, "excerpts.json"), "w") as fh:
    json.dump(dict(set=args.set, fs=fs, excerpt_s=EXCERPT_S, sizes=sizes, seed=args.seed, excerpts=excerpts), fh, indent=1)
  print(f"{args.set}: {len(excerpts)} excerpts from {len(files)} files -> {base} ({', '.join(f's{s}' for s in sizes)})")


if __name__ == "__main__":
  main()
