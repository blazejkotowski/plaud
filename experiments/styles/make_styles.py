"""Style labels for a dataset, for the style-embedding experiments (data.styles in a synth config).

The dataset's audio is cut into passages of --window seconds, within each file (a file's last
passage takes the remainder if it is at least half a window, else it joins the one before). Each
passage gets a style:
  oracle   its own label (one embedding per passage: the upper limit of what conditioning can do)
  cluster  its k-means cluster (--k) on log-mel band statistics (mean and std over time of 64
           bands, standardised)
Each 2 s training chunk takes the style of the passage its first sample falls in.

Writes <out>.npy (one label per chunk index, as AudioFeatureDataset numbers them: files in glob
order, concatenated) and <out>.json (the passages, their sample ranges in that concatenation and
labels, so an evaluation can label any position).

  python experiments/styles/make_styles.py --dataset /mnt/.../percussion/audio --mode cluster --k 12 \
      --out experiments/styles/labels/percussion_k12
"""

import argparse
import glob
import json
import math
import os

import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dataset", required=True)
  ap.add_argument("--mode", choices=["oracle", "cluster"], required=True)
  ap.add_argument("--k", type=int, default=12)
  ap.add_argument("--window", type=float, default=16.0)
  ap.add_argument("--chunk", type=float, default=2.0)
  ap.add_argument("--fs", type=int, default=48000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  files = glob.glob(os.path.join(args.dataset, "**", "*.wav"), recursive=True)   # AudioFeatureDataset's order
  win = int(args.window * args.fs)
  passages, offset, file_offset = [], 0, {}
  for f in files:
    n = sf.info(f).frames
    file_offset[f] = offset
    starts = list(range(0, n, win))
    if len(starts) > 1 and n - starts[-1] < win // 2:
      starts.pop()
    for k, s in enumerate(starts):
      e = starts[k + 1] if k + 1 < len(starts) else n
      passages.append(dict(file=f, start=offset + s, end=offset + e))
    offset += n
  total = offset

  if args.mode == "oracle":
    labels = np.arange(len(passages))
  else:
    feats = []
    for p in passages:
      x, fs = sf.read(p["file"], start=p["start"] - file_offset[p["file"]], frames=p["end"] - p["start"], dtype="float32", always_2d=True)
      m = librosa.power_to_db(librosa.feature.melspectrogram(y=x.mean(1), sr=fs, n_fft=2048, hop_length=1024, n_mels=64, fmin=30))
      feats.append(np.concatenate([m.mean(1), m.std(1)]))
    F = np.array(feats)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    labels = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit_predict(F)
  for p, l in zip(passages, labels):
    p["label"] = int(l)

  n_chunk = int(args.chunk * args.fs)
  n_chunks = math.ceil(total / n_chunk)
  bounds = np.array([p["start"] for p in passages])
  chunk_labels = np.array([passages[int(np.searchsorted(bounds, i * n_chunk, side="right") - 1)]["label"] for i in range(n_chunks)])

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  np.save(args.out + ".npy", chunk_labels)
  json.dump(dict(dataset=args.dataset, mode=args.mode, k=args.k if args.mode == "cluster" else len(passages),
                 window_s=args.window, fs=args.fs, total=total, passages=passages), open(args.out + ".json", "w"), indent=1)
  sizes = np.bincount(labels)
  print(f"{args.out}: {len(passages)} passages, {len(sizes)} styles (passages per style {sizes.min()}-{sizes.max()}), "
        f"{n_chunks} chunks")


if __name__ == "__main__":
  main()
