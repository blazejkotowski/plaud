"""Reconstruct the first --n passages of a data-size excerpt list with a synth checkpoint trained on
the full set, for the style experiments (scored by score_passages.py in the percept env).

Each passage (16 s) is reconstructed in one pass with the encoder's mean latents, as the plugin
hears it. A style-conditioned model (data.styles in its config) gets each frame's style from the
labels' passage list (<styles>.json).

  python experiments/styles/eval_passages.py --config configs/research_styles/percussion_oracle.yaml \
      --excerpts /mnt/.../datasize/percussion/excerpts.json --ckpt training/.../e39.ckpt --out results/x/e40
writes <out>/pNN_orig.wav and <out>/pNN_recon.wav.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from ddsp import DDSP, AudioFeatureDataset
from ddsp.interfaces import build_control_space
from experiments.datasize.eval_recon import chunks


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--config", required=True)
  ap.add_argument("--excerpts", required=True)
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=8)
  ap.add_argument("--noise-bits", type=int, default=0, help="re-bake the synth's noise bands at 16 bits (same noise)")
  ap.add_argument("--interp", default="", help="'linear': ramp the band amplitudes between control frames")
  args = ap.parse_args()

  cfg = OmegaConf.load(args.config)
  fs, rf = int(cfg.audio.fs), int(cfg.model.resampling_factor)
  cs = build_control_space(cfg.model.control_space)
  model = DDSP.load_from_checkpoint(args.ckpt, control_space=cs, map_location="cuda", weights_only=False).cuda().eval()
  if args.noise_bits or args.interp:
    from ddsp.filterbank import FilterBank
    from ddsp.synths.synths import NoiseBandSynth
    for syn in model.synths:
      nbs = getattr(syn, "_noiseband_synth", None)
      if nbs is None:
        continue
      if args.noise_bits:
        old = nbs._filterbank
        new = FilterBank(n_filters=old._n_filters + 2, fs=old._fs, device="cuda", noise_bits=args.noise_bits)
        nbs._filterbank.noisebands, nbs._filterbank.noisebands_scale = new.noisebands.cuda(), new.noisebands_scale.cuda()
      if args.interp == "linear":
        def fwd(amplitudes, *a, _nbs=nbs, **k):
          rf = int(_nbs._resampling_factor); T = amplitudes.shape[-1] * rf
          bands = _nbs._noisebands.to(torch.float32) * _nbs._filterbank.noisebands_scale[:, None]
          idx = torch.arange(T, device=amplitudes.device) % bands.shape[-1]
          out = torch.zeros(amplitudes.shape[0], 1, T, device=amplitudes.device)
          for b0 in range(0, bands.shape[0], 50):                                 # chunked: [B, 50, T] at a time
            up = torch.nn.functional.interpolate(amplitudes[:, b0:b0 + 50], size=T, mode="linear", align_corners=False)
            out[:, 0] += (up * bands[b0:b0 + 50][:, idx]).sum(1)
          return out
        nbs.forward = fwd
  if model.encoder is not None:
    model.encoder.reparametrize = lambda mean, scale: (mean, torch.tensor(0.0, device=mean.device))
  data = AudioFeatureDataset(dataset_path=cfg.data.dataset_path, n_signal=int(fs * float(cfg.audio.chunk_duration_s)),
                             sampling_rate=fs, resampling_factor=rf, control_space=cs, n_channels=int(cfg.audio.n_channels))
  files = glob.glob(os.path.join(cfg.data.dataset_path, "**", "*.wav"), recursive=True)
  starts = np.cumsum([0] + list(data.file_lengths()))
  file_start = {os.path.realpath(f): int(s) for f, s in zip(files, starts)}
  styles = None
  if getattr(cfg.data, "styles", None):
    meta = json.load(open(os.path.splitext(cfg.data.styles)[0] + ".json"))
    bounds = np.array([p["start"] for p in meta["passages"]])
    labels = np.array([p["label"] for p in meta["passages"]])
    styles = lambda pos: labels[np.searchsorted(bounds, pos, side="right") - 1]

  ex = json.load(open(args.excerpts))
  n = int(ex["excerpt_s"] * fs)
  os.makedirs(args.out, exist_ok=True)
  for i, e in enumerate(ex["excerpts"][: args.n]):
    off = file_start[os.path.realpath(e["file"])] + int(e["start"])
    ref, _ = sf.read(e["file"], start=int(e["start"]), frames=n, dtype="float32", always_2d=True)
    got = data._audio[:, off:off + n].cpu().numpy()
    assert np.abs(got.mean(0) - ref.T.mean(0)).max() < 1e-3, f"passage {i} not at its offset"
    a, f = chunks(data, off, 1, n, rf)
    style = None
    if styles is not None:
      t = math.ceil(n / rf)
      style = torch.as_tensor(styles(off + np.arange(t) * rf), device="cuda")[None]
    with torch.no_grad():
      rec, _ = model._autoencode(a, f, style)
    rec = rec.reshape(a.shape[1], -1)[:, :n]
    sf.write(os.path.join(args.out, f"p{i:02d}_orig.wav"), a[0].T.cpu().numpy(), fs)
    sf.write(os.path.join(args.out, f"p{i:02d}_recon.wav"), rec.T.cpu().numpy(), fs)
  print(f"{args.ckpt}: {args.n} passages -> {args.out}")


if __name__ == "__main__":
  main()
