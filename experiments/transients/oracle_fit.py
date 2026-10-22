"""What can the noise-band synth reproduce with ideal controls? Fits its band amplitudes directly to original passages
(no encoder, no decoder) and scores the fit with the aligned measures, next to the trained model's reconstruction.

  fixed   the same noise every step and at the end: the synth's own ceiling
  random  a fresh random noise offset every step (as in training), scored on another one: the ceiling for a
          decoder that can't know which noise it will get

Loss: the model's own loss items (MRSTFT) plus EnvelopeLoss at --env-weight.
Synth variants: --n-filters N builds a noise-band synth with N bands instead of the model's; --flatten replaces
each noise band by its constant-envelope version (cos of its phase, same RMS), so only the amplitude control sets
the band's level over time.

  python experiments/transients/oracle_fit.py --config configs/factory2/02_percussion_nosilence.yaml \
      --ckpt training/synth/percussion_nosilence/last.ckpt --results experiments/styles/results/percussion/base/e160 \
      --out experiments/transients/fit --passages 0 1 2 3
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from ddsp import DDSP
from ddsp.interfaces import build_control_space
from ddsp.losses import EnvelopeLoss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics"))
from aligned import scores  # noqa: E402


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--config", required=True)
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--results", required=True, help="dir with pNN_orig.wav / pNN_recon.wav of the trained model")
  ap.add_argument("--out", required=True)
  ap.add_argument("--passages", type=int, nargs="+", default=[0, 1, 2, 3])
  ap.add_argument("--steps", type=int, default=600)
  ap.add_argument("--lr", type=float, default=0.05)
  ap.add_argument("--init", default="analysis", help="'analysis' (each band's amplitude from the original's level in that "
                  "band, per control frame) or a constant log amplitude")
  ap.add_argument("--env-weight", type=float, default=0.1)
  ap.add_argument("--env-smooth", type=float, nargs="+", default=[20.0, 100.0], help="EnvelopeLoss smoothing, ms")
  ap.add_argument("--mrstft-weight", type=float, default=1.0, help="scales the model's own loss items")
  ap.add_argument("--modes", nargs="+", default=["fixed", "random"])
  ap.add_argument("--n-filters", type=int, default=0)
  ap.add_argument("--flatten", action="store_true")
  ap.add_argument("--fb", nargs="*", default=[], help="filterbank options key=value (flatten_noise=1 flatten_iters=8 "
                  "clean_stopband=1 noise_bits=16); builds a new synth with them")
  ap.add_argument("--no-model", action="store_true", help="skip scoring the trained model's reconstruction")
  args = ap.parse_args()

  cfg = OmegaConf.load(args.config)
  fs_cfg = int(cfg.audio.fs)
  cs = build_control_space(cfg.model.control_space)
  model = DDSP.load_from_checkpoint(args.ckpt, control_space=cs, map_location="cuda", weights_only=False).cuda().eval()
  nb = model.synths[0]._noiseband_synth
  rf = int(cfg.model.resampling_factor)
  if args.n_filters or args.fb:
    from ddsp.synths.synths import NoiseBandSynth
    kw = {k: int(v) for k, v in (o.split("=") for o in args.fb)}
    nb = NoiseBandSynth(n_filters=args.n_filters or nb.n_params, fs=int(cfg.audio.fs), resampling_factor=rf, device="cuda",
                        **kw).cuda().eval()
  if args.flatten:
    from scipy.signal import hilbert
    fb = nb._filterbank
    bands = (fb.noisebands.float() * fb.noisebands_scale[:, None]).cpu().numpy()
    a = hilbert(bands, axis=1)
    flat = np.real(a / np.maximum(np.abs(a), 1e-12))
    flat *= np.sqrt((bands ** 2).mean(1, keepdims=True) / np.maximum((flat ** 2).mean(1, keepdims=True), 1e-20))
    flat = torch.tensor(flat, dtype=torch.float32)
    scale = (flat.abs().amax(1) / 127.0).clamp_min(1e-12)
    fb.noisebands = torch.round(flat / scale[:, None]).clamp(-128, 127).to(torch.int8).cuda()
    fb.noisebands_scale = scale.cuda()
  band_len = nb._noisebands.shape[-1]
  env = EnvelopeLoss(fs=int(cfg.audio.fs), smooth_ms=args.env_smooth).cuda()
  os.makedirs(args.out, exist_ok=True)
  report = {}

  # band edges of the synth (lowpass, bandpasses, highpass) and the power of each unit-amplitude band
  fb = nb._filterbank
  edges = np.concatenate(([0.0], np.asarray(fb._boundaries), [fs_cfg / 2 + 1]))
  band_pow = (fb.noisebands.float() * fb.noisebands_scale[:, None]).pow(2).mean(1).cuda()

  def analysis_init(y):
    n_fft = 4096
    S = torch.stft(y[0, 0], n_fft, rf, window=torch.hann_window(n_fft, device="cuda"), return_complex=True).abs() ** 2
    S = S[:, : y.shape[-1] // rf] * 2 / (torch.hann_window(n_fft).pow(2).sum().item())      # power per bin per frame
    f = torch.fft.rfftfreq(n_fft, 1 / fs_cfg).cuda()
    M = ((f[None] >= torch.tensor(edges[:-1], device="cuda")[:, None]) & (f[None] < torch.tensor(edges[1:], device="cuda")[:, None])).float()
    P = M @ S / n_fft                                                                       # [bands, frames]
    return torch.log(torch.sqrt(P / band_pow[:, None]) + 1e-6)[None]

  def synth(log_amp, shift):
    nb._noisebands_shift = shift
    return nb(torch.exp(log_amp), limit_components=0.0)          # [1, 1, n]

  for p in args.passages:
    orig = os.path.join(args.results, f"p{p:02d}_orig.wav")
    y, fs = sf.read(orig, dtype="float32", always_2d=True)
    y = torch.tensor(y.mean(1), device="cuda")
    n_ctrl = len(y) // rf
    y = y[: n_ctrl * rf][None, None]
    rep = {} if args.no_model else {"model": scores(orig, os.path.join(args.results, f"p{p:02d}_recon.wav"))}
    for mode in args.modes:
      torch.manual_seed(p)
      if args.init == "analysis":
        log_amp = analysis_init(y).detach().clone().requires_grad_(True)
      else:
        log_amp = torch.full((1, nb.n_params, n_ctrl), float(args.init), device="cuda", requires_grad=True)
      opt = torch.optim.Adam([log_amp], lr=args.lr)
      for step in range(args.steps):
        shift = 0 if mode == "fixed" else int(torch.randint(0, band_len // rf, (1,))) * rf
        x = synth(log_amp, shift)
        loss = args.mrstft_weight * sum(w * fn(x, y) for fn, w in model._loss_items) + args.env_weight * env(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
      with torch.no_grad():
        x = synth(log_amp, 0 if mode == "fixed" else (band_len // rf // 2) * rf)
      path = os.path.join(args.out, f"p{p:02d}_{mode}.wav")
      sf.write(path, x[0, 0].cpu().numpy(), fs)
      rep[mode] = scores(orig, path)
      print(f"p{p:02d} {mode}: loss {loss.item() if args.steps else float('nan'):.3f}", {k: round(v, 2) for k, v in rep[mode].items()}, flush=True)
    if "model" in rep:
      print(f"p{p:02d} model:", {k: round(v, 2) for k, v in rep["model"].items()}, flush=True)
    report[f"p{p:02d}"] = rep
  json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
