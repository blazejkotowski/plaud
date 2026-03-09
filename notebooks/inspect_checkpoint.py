#!/usr/bin/env python
"""Inspect what the trained ComplexSineSynth is actually producing."""
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import torch
import math
import numpy as np
from ddsp.blocks import _scaled_sigmoid

# Load checkpoint
ckpt = torch.load('../training/synth/complex_sine_single_1024_frame/last-v1.ckpt', map_location='cpu')
state = ckpt['state_dict']

# Check output layer
out_w = state['decoder.output_params.weight']
out_b = state['decoder.output_params.bias']
print(f"Output layer: weight {out_w.shape}, bias {out_b.shape}")
print(f"  weight: mean={out_w.mean():.4f}, std={out_w.std():.4f}")
print(f"  bias:   mean={out_b.mean():.4f}, std={out_b.std():.4f}")

# _scaled_sigmoid on bias (proxy for zero-input behavior)
print("\n=== _scaled_sigmoid(bias) — network default output ===")
ss_bias = _scaled_sigmoid(out_b)
n_sines = 32
omega_raw = ss_bias[:n_sines]
phi_raw = ss_bias[n_sines:2*n_sines]
amp_start_raw = ss_bias[2*n_sines:3*n_sines]
amp_end_raw = ss_bias[3*n_sines:]

print(f"omega * pi (freq angles):  mean={omega_raw.mean()*math.pi:.4f}, min={omega_raw.min()*math.pi:.4f}, max={omega_raw.max()*math.pi:.4f}")
print(f"phi * pi (phase angles):   mean={phi_raw.mean()*math.pi:.4f}, min={phi_raw.min()*math.pi:.4f}, max={phi_raw.max()*math.pi:.4f}")
print(f"amp_start:  mean={amp_start_raw.mean():.6f}, min={amp_start_raw.min():.6f}, max={amp_start_raw.max():.6f}")
print(f"amp_end:    mean={amp_end_raw.mean():.6f}, min={amp_end_raw.min():.6f}, max={amp_end_raw.max():.6f}")

freq_hz = omega_raw * math.pi * 44100 / (2 * math.pi)
print(f"\nFrequencies (Hz): {freq_hz.detach().numpy().round(1)}")
print(f"Amp starts: {amp_start_raw.detach().numpy().round(6)}")
print(f"Amp ends:   {amp_end_raw.detach().numpy().round(6)}")

# Now actually run inference
print("\n=== Full Forward Pass Inference ===")
import librosa
from ddsp.ddsp import DDSP
from omegaconf import OmegaConf

cfg = OmegaConf.load('../configs/complex_sine/experiment_complex_sine_single_1024_frame.yaml')
model = DDSP(cfg)
model.load_state_dict(state)
model.eval()
model = model.cpu()

# Load audio
audio, _ = librosa.load('/mnt/mariadata/datasets/surrogate/single_sound/processed/synthesized_7.wav', sr=44100, mono=True)
chunk_len = int(2.0 * 44100)
chunk = audio[:chunk_len]
x = torch.from_numpy(chunk).float().unsqueeze(0)  # [1, T]
print(f"Input audio: shape={x.shape}, RMS={x.pow(2).mean().sqrt():.4f}")

with torch.no_grad():
    # Run through the model
    result = model(x)

# Check what the synth produced
y_audio = result['y_audio']
print(f"Output audio: shape={y_audio.shape}, RMS={y_audio.pow(2).mean().sqrt():.6f}, max={y_audio.abs().max():.6f}")
mse = torch.nn.functional.mse_loss(y_audio.squeeze(), x)
print(f"MSE loss: {mse.item():.6f}")

# Get the synth params before and after _scaled_sigmoid
synth_params = result.get('synth_params', None)
if synth_params is not None:
    print(f"\nSynth params: {synth_params.shape}")
    for k_name, sl in [("omega", slice(0, n_sines)), ("phi", slice(n_sines, 2*n_sines)),
                        ("amp_start", slice(2*n_sines, 3*n_sines)), ("amp_end", slice(3*n_sines, None))]:
        p = synth_params[0, sl, :]
        print(f"  {k_name}: mean={p.mean():.6f}, min={p.min():.6f}, max={p.max():.6f}")
