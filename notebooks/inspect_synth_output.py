#!/usr/bin/env python
"""Directly synthesize from checkpoint params to verify silence hypothesis."""
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import torch
import math
import numpy as np
from ddsp.blocks import _scaled_sigmoid
from ddsp.synths.complex_sine_synth import ComplexSineSynth, SynthCoreFunction

# Load checkpoint
ckpt = torch.load('../training/synth/complex_sine_single_1024_frame/last-v1.ckpt', map_location='cpu')
state = ckpt['state_dict']

out_b = state['decoder.output_params.bias']

# Simulate a single frame with bias-only output (no encoder signal contribution)
ss_bias = _scaled_sigmoid(out_b)
n_sines = 32

# Shape the params as [B=1, n_params=128, T=1]
params = ss_bias.unsqueeze(0).unsqueeze(-1)  # [1, 128, 1]
print(f"Params shape: {params.shape}")

# Run through synth
synth = ComplexSineSynth(n_sines=32, fs=44100, resampling_factor=1024, device='cpu')
with torch.no_grad():
    signal = synth(params)

print(f"Signal shape: {signal.shape}")
print(f"Signal RMS: {signal.pow(2).mean().sqrt():.6f}")
print(f"Signal max: {signal.abs().max():.6f}")
print(f"Signal min: {signal.min():.6f}")

# What would MSE be against the target?
import librosa
audio, _ = librosa.load('/mnt/mariadata/datasets/surrogate/single_sound/processed/synthesized_7.wav', sr=44100, mono=True)
target = torch.from_numpy(audio[:1024]).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]
mse = torch.nn.functional.mse_loss(signal, target)
print(f"MSE vs target: {mse.item():.6f}")
print(f"MSE of silence: {target.pow(2).mean():.6f}")

# The real question: what are the frequencies and amplitudes?
omega = params[0, :n_sines, 0] * math.pi
amp_start = params[0, 2*n_sines:3*n_sines, 0]
amp_end = params[0, 3*n_sines:, 0]
amp_mean = (amp_start + amp_end) / 2

freq_hz = omega * 44100 / (2 * math.pi)
print(f"\n=== Per-sinusoid analysis ===")
print(f"{'Sine':>4} {'Freq Hz':>10} {'Amp':>8} {'Contribution RMS':>18}")
for i in range(n_sines):
    # Each sine's contribution to RMS
    t = torch.arange(1024) / 44100.0
    s = amp_mean[i].item() * torch.cos(2 * math.pi * freq_hz[i].item() * t)
    print(f"{i:>4d} {freq_hz[i].item():>10.1f} {amp_mean[i].item():>8.4f} {s.pow(2).mean().sqrt().item():>18.6f}")

# Sum of 32 sines with similar amps and phases → what happens?
print(f"\n32 sines with amp~0.4 each, total signal RMS: {signal.pow(2).mean().sqrt():.6f}")
print(f"Expected RMS if fully incoherent: {0.4 * math.sqrt(32/2):.4f}")
print(f"Expected RMS if fully coherent: {0.4 * 32:.4f}")

# NOW: load the full version_0 (9604 epoch) checkpoint and compare
print("\n=== Now checking V0 (38k+ steps) ===")
ckpt_v0 = torch.load('../training/synth/complex_sine_single_1024_frame/best.ckpt', map_location='cpu')
state_v0 = ckpt_v0['state_dict']
out_b_v0 = state_v0['decoder.output_params.bias']
# V0 had different n_params (192 = 6*32?)
print(f"V0 bias shape: {out_b_v0.shape}")
if out_b_v0.shape[0] == 128:
    ss_v0 = _scaled_sigmoid(out_b_v0)
    amp_start_v0 = ss_v0[2*n_sines:3*n_sines]
    amp_end_v0 = ss_v0[3*n_sines:]
    freq_v0 = ss_v0[:n_sines] * math.pi * 44100 / (2 * math.pi)
    print(f"V0 amp_start: mean={amp_start_v0.mean():.6f}")
    print(f"V0 freq (Hz): {freq_v0.detach().numpy().round(1)}")
elif out_b_v0.shape[0] == 192:
    print("V0 uses 6K parameterization (192 = 6*32)")
    # This was probably a re/im version
    print(f"V0 bias stats: mean={out_b_v0.mean():.4f}, std={out_b_v0.std():.4f}")
