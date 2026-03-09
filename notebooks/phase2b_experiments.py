#!/usr/bin/env python
"""
Phase 2b: Fix MLP training with proper magnitude clamping.

Root cause: z_freq^n with n=1024 explodes when |z_freq| deviates from 1.
MLP outputs unconstrained values → |z_freq| >> 1 → z_freq^1024 = Inf → NaN.

Solutions:
1. Normalize z_freq to unit circle: z_hat = z / |z|
2. Clamp |z_freq| to [1-eps, 1+eps]
3. Use angle output + unit circle construction (but without cos/sin issues)

Key: z/|z| normalization gives gradient-friendly projection to unit circle.
Unlike cos(theta)/sin(theta) which has zero-derivative dead zones.
"""

import sys, os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, "/home/btadeusz/code/multi-sinusoidal-losses")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import json
import random

from ddsp.synths.complex_sine_synth import SynthCoreFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

N_SINES = 16
FS = 44100
N_TIME = 1024
N_STEPS = 10000

rng = np.random.RandomState(SEED)
target_freqs = [2 ** rng.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
target_phases = [rng.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
target_amps = [rng.uniform(0.1, 1.0) for _ in range(N_SINES)]

t = torch.arange(N_TIME, device=device, dtype=torch.float32) / FS
target_signal = torch.zeros(1, N_TIME, device=device)
for freq, phase, amp in zip(target_freqs, target_phases, target_amps):
    target_signal += amp * torch.cos(2 * math.pi * freq * t + phase).unsqueeze(0)

mse_loss = nn.MSELoss()


# SynthCore variants
class SynthCoreGradNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_freq, z_phase, amp_start, amp_end, signal_length):
        return SynthCoreFunction.forward(ctx, z_freq, z_phase, amp_start, amp_end, signal_length)
    @staticmethod
    def backward(ctx, grad_output):
        z_freq, z_phase, amp_start, amp_end = ctx.saved_tensors
        signal_length = ctx.signal_length
        with torch.enable_grad():
            z_freq_ = z_freq.clone().detach().requires_grad_(True)
            z_phase_ = z_phase.clone().detach().requires_grad_(True)
            amp_start_ = None if amp_start is None else amp_start.clone().detach().requires_grad_(True)
            amp_end_ = None if amp_end is None else amp_end.clone().detach().requires_grad_(True)
            signal = SynthCoreFunction.forward(ctx, z_freq_, z_phase_, amp_start_, amp_end_, signal_length)
            grads = torch.autograd.grad(signal, (z_freq_, z_phase_, amp_start_, amp_end_),
                                         grad_output, retain_graph=False, allow_unused=True)
        gf, gp, ga_s, ga_e = grads
        if gf is not None:
            gf = gf / (gf.abs() + 1e-8)
        if gp is not None:
            gp = gp / (gp.abs() + 1e-8)
        return gf, gp, ga_s, ga_e, None


def unit_normalize(z):
    """Project complex tensor to unit circle: z -> z / |z|"""
    return z / (z.abs() + 1e-8)


def soft_clamp_magnitude(z, max_mag=1.001, min_mag=0.999):
    """Clamp complex magnitude to [min_mag, max_mag] range."""
    mag = z.abs().clamp(min=1e-8)
    clamped_mag = mag.clamp(min=min_mag, max=max_mag)
    return z * (clamped_mag / mag)


class TinyDecoder(nn.Module):
    def __init__(self, input_dim, n_sines, hidden=256):
        super().__init__()
        n_params = 6 * n_sines
        self.n_sines = n_sines
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, n_params),
        )

    def forward(self, x):
        out = self.net(x)
        return out.unsqueeze(-1)


def test_variant(name, synth_core_cls, param_transform_fn, n_steps=N_STEPS, lr=1e-3, grad_clip=1.0):
    K = N_SINES
    torch.manual_seed(SEED)
    model = TinyDecoder(input_dim=64, n_sines=K, hidden=256).to(device)
    fixed_input = torch.randn(1, 64, device=device, generator=torch.Generator(device).manual_seed(SEED))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        params = model(fixed_input)

        freq_re = params[:, :K]
        freq_im = params[:, K:2*K]
        phase_re = params[:, 2*K:3*K]
        phase_im = params[:, 3*K:4*K]
        amp_start = torch.abs(params[:, 4*K:5*K]) + 1e-6
        amp_end = torch.abs(params[:, 5*K:6*K]) + 1e-6

        z_freq_raw = torch.complex(freq_re, freq_im)
        z_phase_raw = torch.complex(phase_re, phase_im)

        # Apply parameter transform (normalization, clamping, etc.)
        z_freq, z_phase = param_transform_fn(z_freq_raw, z_phase_raw)

        signal = synth_core_cls.apply(z_freq, z_phase, amp_start, amp_end, N_TIME)
        loss = mse_loss(signal, target_signal)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())

        if step % 2000 == 0:
            z_freq_mag = z_freq.abs().mean().item()
            total_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            print(f"  [{name}] Step {step:>5d}: loss={loss.item():.6f} | |z_freq|={z_freq_mag:.6f} | grad={total_norm:.4f}")

    print(f"  [{name}] Final: {losses[-1]:.6f}")
    return losses


# ══════════════════════════════════════════════════════════════════════
# Transform functions
# ══════════════════════════════════════════════════════════════════════

# V1: Unit normalization (z/|z|) + grad norm in synth
def transform_unit_norm(z_freq, z_phase):
    return unit_normalize(z_freq), unit_normalize(z_phase)

# V2: Unit normalization + NO grad norm
def transform_unit_norm_v2(z_freq, z_phase):
    return unit_normalize(z_freq), unit_normalize(z_phase)

# V3: Soft clamp magnitude
def transform_soft_clamp(z_freq, z_phase):
    return soft_clamp_magnitude(z_freq), soft_clamp_magnitude(z_phase)

# V4: Unit norm for freq only, phase can be free (amplitude in phase)
def transform_freq_norm_only(z_freq, z_phase):
    return unit_normalize(z_freq), z_phase

# V5: No transform (baseline - should NaN)
def transform_none(z_freq, z_phase):
    return z_freq, z_phase


# SynthCore without gradient normalization (standard autograd)
class SynthCoreStandard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_freq, z_phase, amp_start, amp_end, signal_length):
        return SynthCoreFunction.forward(ctx, z_freq, z_phase, amp_start, amp_end, signal_length)
    @staticmethod
    def backward(ctx, grad_output):
        z_freq, z_phase, amp_start, amp_end = ctx.saved_tensors
        signal_length = ctx.signal_length
        with torch.enable_grad():
            z_freq_ = z_freq.clone().detach().requires_grad_(True)
            z_phase_ = z_phase.clone().detach().requires_grad_(True)
            amp_start_ = None if amp_start is None else amp_start.clone().detach().requires_grad_(True)
            amp_end_ = None if amp_end is None else amp_end.clone().detach().requires_grad_(True)
            signal = SynthCoreFunction.forward(ctx, z_freq_, z_phase_, amp_start_, amp_end_, signal_length)
            grads = torch.autograd.grad(signal, (z_freq_, z_phase_, amp_start_, amp_end_),
                                        grad_output, retain_graph=False, allow_unused=True)
        return grads[0], grads[1], grads[2], grads[3], None


# ══════════════════════════════════════════════════════════════════════
# Run all MLP tests
# ══════════════════════════════════════════════════════════════════════

results = {}

tests = [
    # name, synth_core, transform, lr, grad_clip
    ("M1: unit_norm+gradnorm, lr=1e-3", SynthCoreGradNorm, transform_unit_norm, 1e-3, 1.0),
    ("M2: unit_norm+no_gradnorm, lr=1e-3", SynthCoreStandard, transform_unit_norm_v2, 1e-3, 1.0),
    ("M3: unit_norm+gradnorm, lr=1e-4", SynthCoreGradNorm, transform_unit_norm, 1e-4, 1.0),
    ("M4: unit_norm+no_gradnorm, lr=1e-4", SynthCoreStandard, transform_unit_norm_v2, 1e-4, 1.0),
    ("M5: soft_clamp+gradnorm, lr=1e-3", SynthCoreGradNorm, transform_soft_clamp, 1e-3, 1.0),
    ("M6: freq_only_norm+gradnorm, lr=1e-3", SynthCoreGradNorm, transform_freq_norm_only, 1e-3, 1.0),
    ("M7: unit_norm+gradnorm, lr=1e-3, clip=0.5", SynthCoreGradNorm, transform_unit_norm, 1e-3, 0.5),
    ("M8: no_transform+gradnorm (NaN baseline)", SynthCoreGradNorm, transform_none, 1e-3, 1.0),
]

for name, synth_cls, transform, lr, clip in tests:
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    results[name] = test_variant(name, synth_cls, transform, lr=lr, grad_clip=clip)

# Save
with open("phase2b_results.json", "w") as f:
    json.dump(results, f)

print("\n" + "="*70)
print("PHASE 2b SUMMARY")
print("="*70)
for name, losses in results.items():
    final = losses[-1]
    mn = min(losses)
    status = "NaN!" if math.isnan(final) else ("GOOD" if final < 1.0 else ("ok" if final < 2.0 else "bad"))
    print(f"  {status:5s} {name:55s} | final={final:.6f} | min={mn:.6f}")
