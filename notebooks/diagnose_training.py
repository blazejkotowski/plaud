#!/usr/bin/env python
"""
Diagnose ComplexSineSynth training: verify amplitude collapse hypothesis
and measure actual frequency detection quality.
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import torch
import torch.nn.functional as F
import math
import numpy as np
import librosa
from ddsp.ddsp import DDSP
from ddsp.blocks import _scaled_sigmoid
from ddsp.synths.complex_sine_synth import ComplexSineSynth
from omegaconf import OmegaConf

device = torch.device('cpu')

# === Load audio ===
audio, _ = librosa.load('/mnt/mariadata/datasets/surrogate/single_sound/processed/synthesized_7.wav',
                        sr=44100, mono=True)
chunk_len = int(2.0 * 44100)
chunk = audio[:chunk_len]
x_audio = torch.from_numpy(chunk).float().unsqueeze(0)  # [1, T]

print(f"Target audio: shape={x_audio.shape}, RMS={x_audio.pow(2).mean().sqrt():.4f}")
print(f"MSE of silence: {x_audio.pow(2).mean():.6f}")

# === Analyze V2 checkpoint (gradient norm ON, angle params, 500 epochs) ===
print("\n" + "="*70)
print("V2 checkpoint (grad norm ON, angle params, 500 epochs)")
print("="*70)

ckpt = torch.load('../training/synth/complex_sine_single_1024_frame/last-v1.ckpt', map_location='cpu')
state = ckpt['state_dict']

# Build model manually — bypass the OmegaConf issues
# Just use the encoder + decoder + synth directly
from ddsp.blocks import Decoder, WaveformEncoder

n_sines = 32
n_params = 4 * n_sines  # 128
resampling_factor = 1024
latent_size = 512
fs = 44100

# Create encoder
encoder = WaveformEncoder(
    hidden_channels=[16, 32, 64],
    strides=[16, 8, 8],
    latent_size=latent_size,
    resampling_factor=resampling_factor,
)

# Create decoder
decoder = Decoder(
    n_params=n_params,
    n_features=1,  # dummy features since we only use latent
    latent_size=latent_size,
    layer_sizes=[32, 64, 128],
    gru_layers=1
)

# Create synth
synth = ComplexSineSynth(n_sines=n_sines, fs=fs, resampling_factor=resampling_factor, device='cpu')

# Load weights
enc_state = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
dec_state = {k.replace('decoder.', ''): v for k, v in state.items() if k.startswith('decoder.')}

encoder.load_state_dict(enc_state)
decoder.load_state_dict(dec_state)
encoder.eval()
decoder.eval()

# Run forward pass
with torch.no_grad():
    mu, scale = encoder(x_audio)
    z, _ = encoder.reparametrize(mu, scale)

    T_ctl = z.shape[1]
    dummy_features = torch.zeros(1, T_ctl, 1)

    synth_params = decoder(dummy_features, z)  # [1, n_params, T_ctl]
    y_audio = synth(synth_params)  # [1, 1, signal_len]

print(f"\nEncoder output: mu shape={mu.shape}, z shape={z.shape}")
print(f"Decoder output (synth_params): shape={synth_params.shape}")
print(f"Synthesized audio: shape={y_audio.shape}")

# Align lengths
min_len = min(y_audio.shape[-1], x_audio.shape[-1])
y_trim = y_audio[0, 0, :min_len]
x_trim = x_audio[0, :min_len]
mse = F.mse_loss(y_trim, x_trim)
print(f"MSE: {mse.item():.6f}")
print(f"Output RMS: {y_trim.pow(2).mean().sqrt():.6f}")
print(f"Output max: {y_trim.abs().max():.6f}")

# Analyze synth params per frame
print(f"\n--- Synth params analysis (averaged over {synth_params.shape[2]} frames) ---")
omega_vals = synth_params[0, :n_sines, :]  # [32, T_ctl]
phi_vals = synth_params[0, n_sines:2*n_sines, :]
amp_start_vals = synth_params[0, 2*n_sines:3*n_sines, :]
amp_end_vals = synth_params[0, 3*n_sines:, :]

# These are already through _scaled_sigmoid, values in (0, ~2)
freq_hz = omega_vals * math.pi * fs / (2 * math.pi)  # Convert to Hz
amp_mean = (amp_start_vals + amp_end_vals) / 2

print(f"\nFrequency (Hz) stats:")
print(f"  Mean across all: {freq_hz.mean():.1f}")
print(f"  Std across sines: {freq_hz.mean(dim=1).std():.1f}")
print(f"  Range: [{freq_hz.min():.1f}, {freq_hz.max():.1f}]")
print(f"  Per-sine means: {freq_hz.mean(dim=1).detach().numpy().round(0).astype(int)}")

print(f"\nAmplitude stats:")
print(f"  Mean: {amp_mean.mean():.6f}")
print(f"  Std: {amp_mean.std():.6f}")
print(f"  Range: [{amp_mean.min():.6f}, {amp_mean.max():.6f}]")
print(f"  Per-sine means: {amp_mean.mean(dim=1).detach().numpy().round(6)}")

# Check if outputs vary across time
print(f"\nTime variation (std across frames):")
print(f"  Freq std over time: {freq_hz.std(dim=1).mean():.1f} Hz")
print(f"  Amp std over time: {amp_mean.std(dim=1).mean():.6f}")

# === What are the actual target frequencies? ===
print("\n" + "="*70)
print("Target signal analysis (ground truth frequencies)")
print("="*70)

# Use FFT to find the dominant frequencies
from scipy.signal import find_peaks
fft = np.abs(np.fft.rfft(chunk[:44100]))  # 1 second of audio
freqs = np.fft.rfftfreq(44100, 1/44100)
# Find peaks
peaks, props = find_peaks(fft, height=np.max(fft) * 0.01, distance=10)
peak_freqs = freqs[peaks]
peak_amps = fft[peaks]
# Sort by amplitude
idx = np.argsort(-peak_amps)[:20]
print("Top 20 frequency peaks:")
for i, j in enumerate(idx):
    print(f"  {i+1:2d}. {peak_freqs[j]:8.1f} Hz  (amplitude: {peak_amps[j]:.1f})")

# === _scaled_sigmoid analysis for gradient flow ===
print("\n" + "="*70)
print("_scaled_sigmoid gradient analysis")
print("="*70)

x_test = torch.linspace(-10, 10, 1000, requires_grad=True)
y_test = _scaled_sigmoid(x_test)
y_test.sum().backward()
grad = x_test.grad

print(f"At x=0: f(0)={_scaled_sigmoid(torch.tensor(0.0)):.4f}, f'(0)={grad[500]:.4f}")
print(f"At x=-5: f(-5)={_scaled_sigmoid(torch.tensor(-5.0)):.6f}, f'(-5)={grad[250]:.6f}")
print(f"At x=5: f(5)={_scaled_sigmoid(torch.tensor(5.0)):.4f}, f'(5)={grad[750]:.6f}")
print(f"Output range: [{y_test.min():.6f}, {y_test.max():.4f}]")

# What decoder output values map to what frequencies?
print("\n_scaled_sigmoid(x) * pi mapped to frequency:")
for target_freq in [100, 440, 1000, 5000, 10000, 20000]:
    omega_needed = target_freq * 2 * math.pi / fs  # radians/sample
    ss_needed = omega_needed / math.pi  # _scaled_sigmoid output needed
    # Inverse of _scaled_sigmoid is hard analytically, do it numerically
    x_search = torch.linspace(-20, 20, 10000)
    ss_search = _scaled_sigmoid(x_search)
    closest_idx = (ss_search - ss_needed).abs().argmin()
    x_needed = x_search[closest_idx].item()
    print(f"  {target_freq:>5d} Hz → ss_output={ss_needed:.6f} → x≈{x_needed:.2f}")
