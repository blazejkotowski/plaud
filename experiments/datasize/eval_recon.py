"""Score a PLAUD synth checkpoint's reconstruction of the data-size experiment's core excerpt.

The core (excerpts.json: excerpts[0]) is in every subset and in the full set, so all models are scored
on the same 16 s. It is located inside the model's OWN dataset (features are normalised over the whole
training set, 5th-95th percentile, so the core must go through the features the model was trained with),
cut into 2 s chunks as in training and reconstructed (encoder latents + features -> synth). Per chunk:

  mrstft     the training loss: spectral convergence + log-magnitude L1 over the config's FFT sizes
  logmel_db  mean |log-mel difference| in dB (128 mels, 2048 / 512)

Also the mean over 32 random chunks of the model's own training set (`train`), and the core's
original and reconstruction as WAVs.

  python experiments/datasize/eval_recon.py --config configs/research_datasize/harbour_s16.yaml \
      --excerpts /mnt/mariadata/datasets/research/datasize/harbour/excerpts.json --ckpt last --out results/x.json
"""

import argparse
import glob
import json
import math
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from ddsp import DDSP, AudioFeatureDataset
from ddsp.interfaces import build_control_space
from ddsp.utils import find_checkpoint


def core_offset(data: AudioFeatureDataset, dataset_path: str, ex: dict, s16_dir: str) -> int:
  """Sample offset of the core excerpt in the dataset's concatenated audio (the files in glob order)."""
  files = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
  lengths = data.file_lengths()
  core = ex["excerpts"][0]
  core_file = sorted(glob.glob(os.path.join(s16_dir, "audio", "000_*.wav")))[0]
  for k, f in enumerate(files):
    if os.path.realpath(f) == os.path.realpath(core["file"]):       # the full set: the core's source file
      return sum(lengths[:k]) + int(core["start"])
    if os.path.basename(f) == os.path.basename(core_file):          # a subset: the core is its own file
      return sum(lengths[:k])
  raise RuntimeError(f"core not found in {dataset_path}")


def mrstft(x: torch.Tensor, y: torch.Tensor, ffts: list) -> torch.Tensor:
  """Per item [B]: mean over FFT sizes of spectral convergence + log-magnitude L1 (x, y: [B, T])."""
  out = 0.0
  for n in ffts:
    w = torch.hann_window(n, device=x.device)
    X = torch.stft(x, n, n // 4, window=w, return_complex=True).abs()
    Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
    sc = torch.linalg.norm((X - Y).flatten(1), dim=1) / (torch.linalg.norm(X.flatten(1), dim=1) + 1e-7)
    lm = (torch.log(X + 1e-7) - torch.log(Y + 1e-7)).abs().flatten(1).mean(1)
    out = out + sc + lm
  return out / len(ffts)


def logmel_db(x: torch.Tensor, y: torch.Tensor, fs: int) -> torch.Tensor:
  mel = torch.tensor(librosa.filters.mel(sr=fs, n_fft=2048, n_mels=128), device=x.device, dtype=torch.float32)
  w = torch.hann_window(2048, device=x.device)
  spec = lambda s: mel @ torch.stft(s, 2048, 512, window=w, return_complex=True).abs().pow(2)
  return (10 * torch.log10(spec(x) + 1e-8) - 10 * torch.log10(spec(y) + 1e-8)).abs().flatten(1).mean(1)


def chunks(data: AudioFeatureDataset, start: int, n_chunks: int, n: int, rf: int):
  """[B, C, n] audio and [B, T_ctl, D] features, downsampled as in AudioFeatureDataset.__getitem__."""
  a = torch.stack([data._audio[:, start + k * n: start + (k + 1) * n] for k in range(n_chunks)])
  f = torch.stack([data._features[start + k * n: start + (k + 1) * n] for k in range(n_chunks)])
  f = torch.nn.functional.interpolate(f.transpose(1, 2), size=math.ceil(n / rf), mode="linear",
                                      align_corners=False).transpose(1, 2)
  return a, f


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--config", required=True)
  ap.add_argument("--excerpts", required=True)
  ap.add_argument("--ckpt", default="last", help="last, best, or a checkpoint path")
  ap.add_argument("--out", required=True, help="results .json (the WAVs go next to it)")
  ap.add_argument("--train-chunks", type=int, default=32)
  args = ap.parse_args()

  cfg = OmegaConf.load(args.config)
  fs, rf = int(cfg.audio.fs), int(cfg.model.resampling_factor)
  n = int(fs * float(cfg.audio.chunk_duration_s))
  cs = build_control_space(cfg.model.control_space)
  if args.ckpt in ("last", "best"):
    ckpt = find_checkpoint(os.path.join(cfg.experiment.training_dir, "synth", cfg.experiment.name), typ=args.ckpt)
  else:
    ckpt = args.ckpt
  model = DDSP.load_from_checkpoint(ckpt, control_space=cs, map_location="cuda", weights_only=False).cuda().eval()
  epoch = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", -1)) + 1
  data = AudioFeatureDataset(dataset_path=cfg.data.dataset_path, n_signal=n, sampling_rate=fs, resampling_factor=rf,
                             control_space=cs, n_channels=int(cfg.audio.n_channels))
  ex = json.load(open(args.excerpts))
  ffts = list(next(l for l in cfg.losses if l.type == "MRSTFT").params.fft_sizes)

  # the core, checked against the excerpt on disk
  off = core_offset(data, cfg.data.dataset_path, ex, os.path.join(os.path.dirname(args.excerpts), f"s{int(ex['excerpt_s'])}"))
  n_core = int(ex["excerpt_s"] * fs) // n
  ref = sf.read(sorted(glob.glob(os.path.join(os.path.dirname(args.excerpts), f"s{int(ex['excerpt_s'])}", "audio", "000_*.wav")))[0],
                dtype="float32", always_2d=True)[0].T
  got = data._audio[:, off:off + ref.shape[1]].cpu().numpy()
  err = float(np.abs(got.mean(0) - ref.mean(0)).max())
  assert err < 1e-3, f"core mismatch in {cfg.data.dataset_path}: max diff {err}"

  with torch.no_grad():
    a, f = chunks(data, off, n_core, n, rf)
    rec, _ = model._autoencode(a, f)
    x, y = a.sum(1), rec.reshape(a.shape).sum(1)                   # mono sums, [B, T]
    core = dict(mrstft=mrstft(x, y, ffts).tolist(), logmel_db=logmel_db(x, y, fs).tolist())
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(data), generator=g)[: args.train_chunks].tolist()
    tr = {"mrstft": [], "logmel_db": []}
    for i in idx:
      ta, tf = chunks(data, i * n, 1, n, rf)
      tr_rec, _ = model._autoencode(ta, tf)
      tx, ty = ta.sum(1), tr_rec.reshape(ta.shape).sum(1)
      tr["mrstft"].append(float(mrstft(tx, ty, ffts)))
      tr["logmel_db"].append(float(logmel_db(tx, ty, fs)))

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  stem = os.path.splitext(args.out)[0]
  sf.write(stem + "_orig.wav", a.permute(1, 0, 2).reshape(a.shape[1], -1).T.cpu().numpy(), fs)
  sf.write(stem + "_recon.wav", rec.reshape(a.shape).permute(1, 0, 2).reshape(a.shape[1], -1).T.cpu().numpy(), fs)
  res = dict(config=args.config, name=cfg.experiment.name, ckpt=ckpt, epoch=epoch, dataset=cfg.data.dataset_path,
             dataset_s=round(len(data) * n / fs, 1), core=core,
             core_mean={k: float(np.mean(v)) for k, v in core.items()},
             train_mean={k: float(np.mean(v)) for k, v in tr.items()})
  json.dump(res, open(args.out, "w"), indent=1)
  print(f"{cfg.experiment.name} e{epoch}: core mrstft {res['core_mean']['mrstft']:.4f} logmel {res['core_mean']['logmel_db']:.2f} dB"
        f" | train mrstft {res['train_mean']['mrstft']:.4f} logmel {res['train_mean']['logmel_db']:.2f} dB")


if __name__ == "__main__":
  main()
