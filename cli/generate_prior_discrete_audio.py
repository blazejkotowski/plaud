"""
Offline generation: discrete prior -> latent compressor -> DDSP synth -> WAV.

Mirrors the inference path used by the nn~ export so you can audition a trained
codec prior outside Max/MSP. Supports mono and multichannel (e.g. stereo) models.
"""
import argparse
import math
import os
import sys
import tarfile
from typing import Optional, Tuple

# Ensure THIS repo's `ddsp` is imported (not a globally pip-installed one) when
# run as `python cli/generate_prior_discrete_audio.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa

from ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace, build_control_space
from ddsp.latent_compressor import LatentCompressor
from ddsp.prior import PriorDiscrete
from ddsp.registry import FEATURE_EXTRACTORS


def _build_control_space_from_ddsp_ckpt(ddsp_ckpt_path: str) -> Tuple[ControlSpace, int, int]:
  ckpt = torch.load(ddsp_ckpt_path, map_location='cpu')
  h = ckpt.get('hyper_parameters', {})
  feature_dim = int(h.get('feature_dim', 0) or 0)
  latent_size = int(h.get('latent_size', 0) or 0)

  fields = []
  if feature_dim > 0:
    fields.append(ControlField(name='features', dim=feature_dim, source='feature', extractor=None))
  if latent_size > 0:
    fields.append(ControlField(name='latents', dim=latent_size, source='latent', extractor=None))
  return ControlSpace(tuple(fields)), feature_dim, latent_size


def _extract_features_audio_rate(x_audio: torch.Tensor, fs: int, control_space: ControlSpace) -> torch.Tensor:
  """Compute audio-rate feature matrix [T, D_feat] following ControlSpace spec.

  x_audio is mono [T].
  """
  feats = []
  for field in control_space.fields:
    if field.source != 'feature':
      continue
    if not field.extractor:
      raise RuntimeError(f"ControlField '{field.name}' missing extractor; pass a ControlSpace built from config")

    extractor = FEATURE_EXTRACTORS.create(field.extractor, **(field.params or {}))
    feat = extractor(x_audio, fs)
    if feat.ndim == 1:
      feat = feat.unsqueeze(-1)

    norm = dict(field.normalization) if field.normalization is not None else {}
    if 'mean' in norm and 'std' in norm:
      mean = torch.as_tensor(norm['mean'], device=feat.device, dtype=feat.dtype)
      std = torch.as_tensor(norm['std'], device=feat.device, dtype=feat.dtype)
      feat = (feat - mean) / (std + 1e-8)
    elif 'min' in norm and 'max' in norm:
      minv = torch.as_tensor(norm['min'], device=feat.device, dtype=feat.dtype)
      maxv = torch.as_tensor(norm['max'], device=feat.device, dtype=feat.dtype)
      feat = (feat - minv) / (maxv - minv + 1e-8)

    feats.append(feat)

  if not feats:
    return x_audio.new_zeros((x_audio.shape[0], 0))
  return torch.cat(feats, dim=-1)


def _load_audio_multichannel(wav_path: str, fs: int, n_channels: int, seconds: float, offset_s: float,
                             device: str) -> Tuple[torch.Tensor, torch.Tensor]:
  """Load audio coerced to n_channels. Returns (multi [n_ch, T], mono [T])."""
  x_np, _ = librosa.load(wav_path, sr=int(fs), mono=False, offset=float(offset_s), duration=float(seconds))
  x = torch.from_numpy(np.atleast_2d(x_np)).to(device=device, dtype=torch.float32)  # [c, T]

  need = int(round(float(seconds) * float(fs)))
  if x.shape[-1] < need:
    x = F.pad(x, (0, need - x.shape[-1]))

  c = x.shape[0]
  if c == n_channels:
    multi = x
  elif c == 1:
    multi = x.repeat(n_channels, 1)
  elif c > n_channels:
    multi = x[:n_channels]
  else:  # 1 < c < n_channels
    multi = torch.cat([x, x[-1:].repeat(n_channels - c, 1)], dim=0)

  mono = multi.mean(dim=0)  # [T]
  return multi, mono


@torch.no_grad()
def _prime_tokens_from_wav(
  *,
  ddsp: DDSP,
  compressor: LatentCompressor,
  control_space: ControlSpace,
  feature_dim: int,
  latent_size: int,
  wav_path: str,
  seconds: float,
  offset_s: float,
  device: str,
) -> torch.Tensor:
  n_channels = int(getattr(ddsp, 'n_channels', 1))
  x_multi, x_mono = _load_audio_multichannel(wav_path, int(ddsp.fs), n_channels, seconds, offset_s, device)

  # features: audio-rate (from mono) -> control-rate
  x_feat_audio = _extract_features_audio_rate(x_mono, int(ddsp.fs), control_space)  # [T_audio, D_feat]
  T_ctl = int(math.ceil(x_mono.shape[0] / float(ddsp.resampling_factor)))
  if x_feat_audio.shape[-1] != feature_dim:
    raise RuntimeError(f'feature_dim mismatch: expected {feature_dim}, got {x_feat_audio.shape[-1]}')

  x_feat_ctl = F.interpolate(
    x_feat_audio.to(device).T.unsqueeze(0),
    size=T_ctl,
    mode='linear',
    align_corners=False,
  ).squeeze(0).T  # [T_ctl, D_feat]

  if ddsp.encoder is None:
    raise RuntimeError('Cannot prime: DDSP model has no encoder')

  # Encoder downmixes the [1, n_channels, T] input internally.
  mu, scale = ddsp.encoder(x_multi.unsqueeze(0))
  z, _ = ddsp.encoder.reparametrize(mu, scale)
  z = ddsp._smooth_latents(z)

  T_min = min(int(x_feat_ctl.shape[0]), int(z.shape[1]))
  x_feat_ctl = x_feat_ctl[:T_min].unsqueeze(0)
  z = z[:, :T_min, :]

  if z.shape[-1] != latent_size:
    raise RuntimeError(f'latent_size mismatch: expected {latent_size}, got {z.shape[-1]}')

  controls = torch.cat([x_feat_ctl, z], dim=-1)[:, :, :(feature_dim + latent_size)]  # [1, T, D]
  tokens = compressor.encode_codes(controls)
  if tokens.dim() == 2:
    tokens = tokens.unsqueeze(-1)
  return tokens


@torch.no_grad()
def _sample_tokens(
  prior: PriorDiscrete,
  n_tokens: int,
  primer_len: int,
  temperature: float,
  sampling: str,
  device: str,
  primer_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  codebook_size = int(prior.codebook_size)
  num_codebooks = int(prior.num_codebooks)
  max_len = int(getattr(prior, '_max_len', 256))
  start_id = int(getattr(prior, 'start_token_id', codebook_size))

  if n_tokens < 1:
    raise ValueError('n_tokens must be >= 1')

  sampling = str(sampling).lower()
  if sampling not in ('multinomial', 'argmax'):
    raise ValueError("sampling must be 'multinomial' or 'argmax'")

  # Position 0 is the learned START token (matches the nn~ wrapper cold start).
  # A real-audio primer (if given) follows START; generation fills the rest.
  buf = torch.full((1, n_tokens + 1, num_codebooks), start_id, dtype=torch.long, device=device)

  if primer_tokens is None:
    gen_start = 1
  else:
    if primer_tokens.dtype != torch.long:
      primer_tokens = primer_tokens.long()
    if primer_tokens.dim() != 3 or primer_tokens.shape[-1] != num_codebooks:
      raise ValueError('primer_tokens must have shape [1, T, num_codebooks]')
    L = min(int(primer_len), int(primer_tokens.shape[1]), int(n_tokens))
    buf[:, 1:1 + L, :] = primer_tokens[:, :L, :].to(device)
    gen_start = 1 + L

  for t in range(gen_start, n_tokens + 1):
    start = max(0, t - max_len)
    ctx = buf[:, start:t, :]
    logits = prior(ctx)  # [1, S, N, K]
    next_logits = logits[:, -1, :, :]  # [1, N, K]

    if sampling == 'argmax':
      buf[:, t, :] = torch.argmax(next_logits, dim=-1)
      continue

    temp = max(1e-4, float(temperature))
    probs = torch.softmax(next_logits / temp, dim=-1)
    samp = torch.multinomial(probs.reshape(-1, codebook_size), 1).reshape(1, num_codebooks)
    buf[:, t, :] = samp

  # Drop the START token; return the n_tokens real tokens.
  return buf[:, 1:, :]


def main():
  ap = argparse.ArgumentParser(description='Generate audio from a discrete prior -> latent compressor -> DDSP synth.')
  ap.add_argument('--prior_ckpt', type=str, required=True, help='Path to PriorDiscrete checkpoint (.ckpt).')
  ap.add_argument('--ddsp_ckpt', type=str, required=True, help='Path to DDSP checkpoint (.ckpt).')
  ap.add_argument('--ddsp_config', type=str, default='configs/experiment_hybrid.yaml',
                  help='DDSP YAML config used to build ControlSpace (required for priming feature extraction).')
  ap.add_argument('--compressor_ckpt', type=str, required=True, help='Path to LatentCompressor checkpoint (.ckpt).')

  ap.add_argument('--seconds', type=float, default=30.0, help='How many seconds to generate (approx).')
  ap.add_argument('--seed', type=int, default=0, help='Random seed for sampling.')
  ap.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (used for multinomial sampling).')
  ap.add_argument('--sampling', type=str, default='multinomial', choices=['multinomial', 'argmax'],
                  help="Sampling strategy: 'multinomial' (stochastic) or 'argmax' (greedy).")
  ap.add_argument('--primer_frac', type=float, default=0.25, help='Primer length as a fraction of prior max_len.')

  ap.add_argument('--prime_wav', type=str, default='',
                  help='Optional: path to a WAV file used to prime the prior (real tokens for the primer window).')
  ap.add_argument('--prime_seconds', type=float, default=4.0, help='How many seconds of prime_wav to use.')
  ap.add_argument('--prime_offset_s', type=float, default=0.0, help='Offset (seconds) into prime_wav.')

  ap.add_argument('--target_fs', type=int, default=0, help='Output WAV sample rate (0 = keep DDSP fs).')
  ap.add_argument('--out_dir', type=str, default='outputs/generated_prior_discrete', help='Output directory.')
  ap.add_argument('--prefix', type=str, default='', help='Optional filename prefix. If empty, derived from settings.')
  ap.add_argument('--device', type=str, default='', help="'cuda' or 'cpu'. Default: auto.")

  ap.add_argument('--no_plots', action='store_true', help='Skip the diagnostic plots.')
  ap.add_argument('--make_archive', action='store_true', help='Create a .tgz archive of the artifacts.')
  args = ap.parse_args()

  device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(int(args.seed))
  np.random.seed(int(args.seed))

  os.makedirs(args.out_dir, exist_ok=True)

  # Build ControlSpace. For priming we need real feature extractors, so prefer config.
  if args.ddsp_config and os.path.exists(args.ddsp_config):
    with open(args.ddsp_config, 'r') as f:
      ddsp_cfg = yaml.safe_load(f)
    control_space = build_control_space(ddsp_cfg['model']['control_space'])
    _, feature_dim, latent_size = _build_control_space_from_ddsp_ckpt(args.ddsp_ckpt)
  else:
    control_space, feature_dim, latent_size = _build_control_space_from_ddsp_ckpt(args.ddsp_ckpt)

  ddsp = DDSP.load_from_checkpoint(
    args.ddsp_ckpt,
    strict=False,
    streaming=False,
    device=device,
    control_space=control_space,
  ).to(device)
  ddsp.eval()

  compressor = LatentCompressor.load_from_checkpoint(args.compressor_ckpt, strict=False).to(device)
  compressor.eval()

  prior = PriorDiscrete.load_from_checkpoint(args.prior_ckpt, strict=False).to(device)
  prior.eval()

  compression_ratio = int(getattr(compressor, 'compression_ratio', 32))
  max_len = int(getattr(prior, '_max_len', 256))
  n_channels = int(getattr(ddsp, 'n_channels', 1))

  control_rate = float(ddsp.fs) / float(ddsp.resampling_factor)
  tokens_per_sec = control_rate / float(compression_ratio)
  n_tokens = int(math.ceil(float(args.seconds) * tokens_per_sec))

  primer_len = max(1, int(round(float(args.primer_frac) * max_len)))
  primer_len = min(primer_len, max_len)

  if not args.prefix:
    args.prefix = f'prior_discrete_seed{args.seed}_{int(args.seconds)}s_{args.sampling}_t{args.temperature}'

  print('device:', device)
  print('ddsp.fs:', ddsp.fs, 'resampling_factor:', ddsp.resampling_factor, 'control_rate:', control_rate)
  print('n_channels:', n_channels, 'feature_dim:', feature_dim, 'latent_size:', latent_size)
  print('compression_ratio:', compression_ratio, 'tokens_per_sec:', round(tokens_per_sec, 3))
  print('n_tokens:', n_tokens, 'prior_max_len:', max_len, 'primer_len:', primer_len)

  primer_tokens = None
  if args.prime_wav:
    print('priming from wav:', args.prime_wav)
    primer_tokens = _prime_tokens_from_wav(
      ddsp=ddsp,
      compressor=compressor,
      control_space=control_space,
      feature_dim=feature_dim,
      latent_size=latent_size,
      wav_path=args.prime_wav,
      seconds=float(args.prime_seconds),
      offset_s=float(args.prime_offset_s),
      device=device,
    )
    print('prime_tokens_len:', int(primer_tokens.shape[1]))
    if primer_tokens.shape[1] < primer_len:
      print(f'WARNING: prime_tokens shorter than primer_len; shrinking primer_len to {primer_tokens.shape[1]}')
      primer_len = int(primer_tokens.shape[1])

  print('sampling tokens...')
  tokens = _sample_tokens(
    prior,
    n_tokens=n_tokens,
    primer_len=primer_len,
    temperature=args.temperature,
    sampling=args.sampling,
    device=device,
    primer_tokens=primer_tokens,
  )

  # Diagnostic: fraction of unique tokens (low -> collapsed / "loopy").
  uniq = int(torch.unique(tokens).numel())
  print(f'unique token ids used: {uniq} / {int(prior.codebook_size) * int(prior.num_codebooks)} (low => collapsed/repetitive)')

  print('decoding codes -> controls...')
  controls = compressor.decode_codes(tokens)  # [1, T, D]
  controls = controls[:, :, :(feature_dim + latent_size)]
  features = controls[:, :, :feature_dim]
  latents = controls[:, :, feature_dim:feature_dim + latent_size]

  print('DDSP decode + synth...')
  synth_params = ddsp.decoder(features, latents)
  audio = ddsp._synthesize(synth_params)  # [1, n_channels, T] (or [1, T])
  if audio.dim() == 2:
    audio = audio.unsqueeze(1)

  mx = audio.abs().max().clamp(min=1e-8)
  audio = (audio / mx).clamp(-1, 1)

  orig_fs = int(ddsp.fs)
  target_fs = int(args.target_fs) or orig_fs
  wav = audio[0].cpu()  # [n_channels, T]
  if target_fs != orig_fs:
    print(f'resampling {orig_fs} -> {target_fs} ...')
    wav = torchaudio.functional.resample(wav, orig_freq=orig_fs, new_freq=target_fs)

  wav_path = os.path.join(args.out_dir, f'{args.prefix}.wav')
  torchaudio.save(wav_path, wav, sample_rate=target_fs)
  print('wrote:', wav_path)

  npz_path = os.path.join(args.out_dir, f'{args.prefix}_arrays.npz')
  np.savez(
    npz_path,
    tokens=tokens.detach().cpu().numpy()[0],
    controls=controls.detach().float().cpu().numpy()[0],
    feature_dim=feature_dim,
    latent_size=latent_size,
    ddsp_fs=orig_fs,
    n_channels=n_channels,
    control_rate=control_rate,
    compression_ratio=compression_ratio,
    temperature=float(args.temperature),
    sampling=str(args.sampling),
    seconds=float(args.seconds),
  )
  print('wrote:', npz_path)

  artifacts = [wav_path, npz_path]
  if not args.no_plots:
    controls_np = controls.detach().float().cpu().numpy()[0]
    n = controls_np.shape[0]
    ds = max(1, n // 5000)
    t = (np.arange(n)[::ds] / control_rate)

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i in range(feature_dim):
      ax[0].plot(t, controls_np[::ds, i], label=f'feature[{i}]', linewidth=0.8)
    ax[0].set_title('Generated DDSP features (control-rate)')
    ax[0].legend(ncol=4, fontsize=8)
    ax[0].grid(True, alpha=0.3)
    for i in range(latent_size):
      ax[1].plot(t, controls_np[::ds, feature_dim + i], label=f'latent[{i}]', linewidth=0.8)
    ax[1].set_title('Generated DDSP latents (control-rate)')
    ax[1].legend(ncol=4, fontsize=8)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlabel('time (s)')
    controls_png = os.path.join(args.out_dir, f'{args.prefix}_controls.png')
    fig.tight_layout()
    fig.savefig(controls_png, dpi=150)
    plt.close(fig)
    print('wrote:', controls_png)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.hist(tokens.detach().cpu().numpy().reshape(-1), bins=50)
    ax.set_title('Token histogram (all codebooks)')
    ax.set_xlabel('token id')
    ax.set_ylabel('count')
    ax.grid(True, alpha=0.3)
    hist_png = os.path.join(args.out_dir, f'{args.prefix}_token_hist.png')
    fig.tight_layout()
    fig.savefig(hist_png, dpi=150)
    plt.close(fig)
    print('wrote:', hist_png)
    artifacts += [controls_png, hist_png]

  if args.make_archive:
    archive_path = os.path.join(args.out_dir, f'{args.prefix}_artifacts.tgz')
    with tarfile.open(archive_path, 'w:gz') as tf:
      for p in artifacts:
        tf.add(p, arcname=os.path.basename(p))
    print('wrote:', archive_path)


if __name__ == '__main__':
  main()
