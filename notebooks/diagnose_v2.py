#!/usr/bin/env python
"""
Diagnose what the trained ComplexSineSynth is actually producing.
Uses raw state_dict inspection — no model reconstruction needed.
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import torch
import torch.nn.functional as F
import math
import numpy as np
import librosa
from ddsp.blocks import _scaled_sigmoid
from scipy.signal import find_peaks

# === Load target audio ===
audio, _ = librosa.load('/mnt/mariadata/datasets/surrogate/single_sound/processed/synthesized_7.wav',
                        sr=44100, mono=True)
print(f"Target audio: {audio.shape}, RMS={np.sqrt(np.mean(audio**2)):.4f}")
print(f"MSE of silence: {np.mean(audio**2):.6f}")

# === Analyze target frequencies (FFT) ===
print("\n" + "="*70)
print("TARGET: Ground truth frequencies from FFT")
print("="*70)

fft = np.abs(np.fft.rfft(audio[:44100]))  # 1 second
freqs = np.fft.rfftfreq(44100, 1/44100)
peaks, _ = find_peaks(fft, height=np.max(fft) * 0.01, distance=10)
peak_freqs = freqs[peaks]
peak_amps = fft[peaks]
idx = np.argsort(-peak_amps)[:20]
print("Top 20 frequency peaks:")
for i, j in enumerate(idx):
    print(f"  {i+1:2d}. {peak_freqs[j]:8.1f} Hz  (mag: {peak_amps[j]:.1f})")

# === V2 checkpoint analysis ===
print("\n" + "="*70)
print("CHECKPOINT V2: grad norm ON, angle params, 500 epochs")
print("="*70)

ckpt = torch.load('../training/synth/complex_sine_single_1024_frame/last-v1.ckpt', map_location='cpu')
state = ckpt['state_dict']

# Decoder output layer
out_w = state['decoder.output_params.weight']
out_b = state['decoder.output_params.bias']
n_sines = 32
n_params = 4 * n_sines

print(f"\nDecoder output layer:")
print(f"  weight: shape={out_w.shape}, mean={out_w.mean():.4f}, std={out_w.std():.4f}")
print(f"  bias:   shape={out_b.shape}, mean={out_b.mean():.4f}, std={out_b.std():.4f}")

# Check pre-activation output range
print(f"\n  Raw bias values (pre-sigmoid) for each param group:")
print(f"    omega bias:     mean={out_b[:n_sines].mean():.4f}, range=[{out_b[:n_sines].min():.4f}, {out_b[:n_sines].max():.4f}]")
print(f"    phi bias:       mean={out_b[n_sines:2*n_sines].mean():.4f}, range=[{out_b[n_sines:2*n_sines].min():.4f}, {out_b[n_sines:2*n_sines].max():.4f}]")
print(f"    amp_start bias: mean={out_b[2*n_sines:3*n_sines].mean():.4f}, range=[{out_b[2*n_sines:3*n_sines].min():.4f}, {out_b[2*n_sines:3*n_sines].max():.4f}]")
print(f"    amp_end bias:   mean={out_b[3*n_sines:].mean():.4f}, range=[{out_b[3*n_sines:].min():.4f}, {out_b[3*n_sines:].max():.4f}]")

# Pass through _scaled_sigmoid
ss = _scaled_sigmoid(out_b)
omega_ss = ss[:n_sines]
phi_ss = ss[n_sines:2*n_sines]
amp_start_ss = ss[2*n_sines:3*n_sines]
amp_end_ss = ss[3*n_sines:]

freq_hz = omega_ss * math.pi * 44100 / (2 * math.pi)
amp_mean = (amp_start_ss + amp_end_ss) / 2

print(f"\nAfter _scaled_sigmoid (bias only, proxy for zero-input):")
print(f"  Frequencies (Hz): mean={freq_hz.mean():.1f}, range=[{freq_hz.min():.1f}, {freq_hz.max():.1f}]")
print(f"  Amplitudes: mean={amp_mean.mean():.6f}, range=[{amp_mean.min():.6f}, {amp_mean.max():.6f}]")

# Check if biases moved at all from initialization
# nn.Linear init: bias ~U(-1/sqrt(in), 1/sqrt(in)) where in=hidden_size
in_features = out_w.shape[1]
init_bound = 1.0 / math.sqrt(in_features)
print(f"\n  Expected init bound: ±{init_bound:.4f} (in_features={in_features})")
print(f"  Bias std: {out_b.std():.4f} (init: ~{init_bound/math.sqrt(3):.4f})")
print(f"  Weight std: {out_w.std():.4f}")

# === _scaled_sigmoid analysis ===
print("\n" + "="*70)
print("_scaled_sigmoid ANALYSIS")
print("="*70)

# Gradient and range analysis
x = torch.linspace(-10, 10, 10000, requires_grad=True)
y = _scaled_sigmoid(x)
y.sum().backward()
g = x.grad.detach()

# Critical points
print(f"  f(-10) = {_scaled_sigmoid(torch.tensor(-10.0)):.8f}")
print(f"  f(-5)  = {_scaled_sigmoid(torch.tensor(-5.0)):.8f}")
print(f"  f(-2)  = {_scaled_sigmoid(torch.tensor(-2.0)):.6f}")
print(f"  f(0)   = {_scaled_sigmoid(torch.tensor(0.0)):.6f}")
print(f"  f(2)   = {_scaled_sigmoid(torch.tensor(2.0)):.6f}")
print(f"  f(5)   = {_scaled_sigmoid(torch.tensor(5.0)):.6f}")
print(f"  f(10)  = {_scaled_sigmoid(torch.tensor(10.0)):.6f}")

# Gradient at key points
for xi in [-5, -2, -1, 0, 1, 2, 5]:
    idx = int((xi + 10) / 20.0 * 9999)
    idx = max(0, min(9999, idx))
    print(f"  f'({xi:>3d}) = {g[idx]:.6f}")

# What x values give useful output ranges?
print(f"\n  Mapping: _scaled_sigmoid(x) * pi → frequency (Hz):")
for target_freq in [50, 100, 440, 1000, 5000, 10000, 20000]:
    omega_needed = target_freq * 2 * math.pi / 44100
    ss_needed = omega_needed / math.pi
    x_search = torch.linspace(-20, 20, 100000)
    ss_search = _scaled_sigmoid(x_search)
    closest_idx = (ss_search - ss_needed).abs().argmin()
    x_needed = x_search[closest_idx].item()
    grad_at = g[max(0, min(9999, int((x_needed + 10) / 20.0 * 9999)))] if abs(x_needed) <= 10 else 0
    print(f"    {target_freq:>5d} Hz → ss={ss_needed:.6f} → x≈{x_needed:.2f} (grad≈{grad_at:.6f})")

# Amplitude: what x values give reasonable amplitudes (0.01 to 0.5)?
print(f"\n  Mapping: _scaled_sigmoid(x) → amplitude:")
for target_amp in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    x_search = torch.linspace(-20, 20, 100000)
    ss_search = _scaled_sigmoid(x_search)
    closest_idx = (ss_search - target_amp).abs().argmin()
    x_needed = x_search[closest_idx].item()
    grad_at = g[max(0, min(9999, int((x_needed + 10) / 20.0 * 9999)))] if abs(x_needed) <= 10 else 0
    print(f"    amp={target_amp:.3f} → x≈{x_needed:.2f} (grad≈{grad_at:.6f})")

# === The fundamental problem ===
print("\n" + "="*70)
print("DIAGNOSIS: The amplitude collapse mechanism")
print("="*70)

# At init, all 32 sines have amp ~0.4 and random-ish frequencies
# When 32 sines are summed, the signal has RMS ~0.4*sqrt(32/2) ≈ 1.6
# But target has RMS ~0.21
# So the gradient wants to REDUCE total energy
# Easiest path: push all amps toward 0 (strong gradient, _scaled_sigmoid gradient is decent there)
# Hard path: find correct frequencies (gradient through cos/sin is weak)
#
# Once amps → 0, there's no gradient signal for frequencies anymore
# ∂MSE/∂omega = ∂MSE/∂signal * ∂signal/∂omega
# When amp → 0, ∂signal/∂omega → 0 regardless of the frequency
#
# This is the classic amplitude-frequency disentanglement problem

init_amp = 0.4
summed_rms = init_amp * math.sqrt(n_sines / 2)
target_rms = 0.21
print(f"  Init amplitude per sine: {init_amp:.2f}")
print(f"  Init summed RMS (32 sines): {summed_rms:.2f}")
print(f"  Target RMS: {target_rms:.2f}")
print(f"  Ratio (init/target): {summed_rms/target_rms:.1f}x too loud!")
print()
print("  → At init, gradient strongly pushes amplitudes DOWN")
print("  → Once amps ≈ 0, frequency gradients ∂loss/∂freq ∝ amp → vanish")
print("  → Network trapped at silence")
print()
print("  Esteban's solution (single frame, direct optimization):")
print("    - No _scaled_sigmoid → frequencies and amps are unconstrained")
print("    - Raw complex params → 2 DOF gradient for frequency")
print("    - Gradient normalization → balanced learning speed")
print("    - Init with reasonable values → never starts too loud")
