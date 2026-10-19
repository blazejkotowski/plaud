"""Quick listening check of a synth checkpoint (while it trains, or after).

Takes contiguous stretches of the training set (consecutive cached chunks), and writes for each:
  <n>_orig.wav   the input
  <n>_recon.wav  the reconstruction (encoder latents + features)
  <n>_fade.wav   the same, with the loudness feature faded to 0 over the second half
plus preview.png (spectrograms) and stats.json (output level at loudness 0, per clip).

  python -m cli.preview_synth --config configs/factory1/01_jukebox_jazz.yaml --out /tmp/preview_jazz
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from ddsp import DDSP, AudioFeatureDataset
from ddsp.interfaces import build_control_space
from ddsp.utils import find_checkpoint


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--config", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--clips", type=int, default=4)
  ap.add_argument("--chunks", type=int, default=5, help="consecutive 2 s chunks per clip")
  ap.add_argument("--checkpoint", default="last", choices=["last", "best"])
  ap.add_argument("--seed", type=int, default=1)
  args = ap.parse_args()

  cfg = OmegaConf.load(args.config)
  fs = int(cfg.audio.fs)
  n_signal = int(fs * float(cfg.audio.chunk_duration_s))
  ckpt_dir = os.path.join(cfg.experiment.training_dir, "synth", cfg.experiment.name)
  ckpt = find_checkpoint(ckpt_dir, typ=args.checkpoint)
  model = DDSP.load_from_checkpoint(ckpt, control_space=build_control_space(cfg.model.control_space),
                                    map_location="cuda", weights_only=False).cuda().eval()
  data = AudioFeatureDataset(dataset_path=cfg.data.dataset_path, n_signal=n_signal, sampling_rate=fs,
                             resampling_factor=int(cfg.model.resampling_factor),
                             control_space=build_control_space(cfg.model.control_space),
                             n_channels=int(cfg.audio.n_channels))
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(args.seed)
  li = model._loudness_index
  stats, specs = [], []
  for n in range(args.clips):
    start = int(rng.integers(0, len(data) - args.chunks))
    parts = [data[start + k] for k in range(args.chunks)]
    audio = torch.cat([p[0] for p in parts], dim=-1).unsqueeze(0).cuda()
    feats = torch.cat([p[1] for p in parts], dim=0).unsqueeze(0).cuda()
    with torch.no_grad():
      recon, _ = model._autoencode(audio, feats)
      z, f = model._last_z, model._last_features
      fade = f.clone()
      T = fade.shape[1]
      ramp = torch.ones(T, device=fade.device)
      ramp[T // 2:] = torch.linspace(1.0, 0.0, T - T // 2, device=fade.device)
      fade[..., li] = fade[..., li] * ramp
      faded = model._synthesize(model.decoder(fade, z))
      silent_db = model._silent_output_db(z, f)
    outs = {"orig": audio, "recon": recon, "fade": faded}
    for k, x in outs.items():
      y = x.reshape(-1, x.shape[-1]).T.cpu().numpy()
      sf.write(os.path.join(args.out, f"{n}_{k}.wav"), y, fs)
    tail = faded.reshape(-1)[-fs // 2:]
    stats.append(dict(clip=n, start_chunk=start, loudness0_db=round(float(silent_db.mean()), 1),
                      fade_tail_db=round(float(10 * torch.log10(tail.pow(2).mean() + 1e-12)), 1),
                      recon_db=round(float(10 * torch.log10(recon.pow(2).mean() + 1e-12)), 1),
                      orig_db=round(float(10 * torch.log10(audio.pow(2).mean() + 1e-12)), 1)))
    specs.append({k: v.reshape(-1, v.shape[-1])[0].cpu().numpy() for k, v in outs.items()})
  epoch = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", -1)) + 1
  with open(os.path.join(args.out, "stats.json"), "w") as f:
    json.dump(dict(checkpoint=ckpt, epoch=epoch, clips=stats), f, indent=1)

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(len(specs), 3, figsize=(15, 2.6 * len(specs)), squeeze=False)
  for r, s in enumerate(specs):
    for c, k in enumerate(["orig", "recon", "fade"]):
      ax = axes[r][c]
      ax.specgram(s[k], NFFT=1024, Fs=fs, noverlap=768, cmap="magma", vmin=-140, vmax=-20)
      ax.set_ylim(0, 16000)
      ax.set_title(f"clip {r} {k}", fontsize=9)
      ax.tick_params(labelsize=7)
  plt.tight_layout()
  plt.savefig(os.path.join(args.out, "preview.png"), dpi=70)
  print(json.dumps(stats))


if __name__ == "__main__":
  main()
