#!/usr/bin/env python
"""
Phase 2: Fix gradient flow through neural networks.

Finding: Per-element grad norm in SynthCore makes each sinusoid's gradient
magnitude 1, causing total gradient ~K which blows up network weights.

Solutions to test:
1. Global gradient norm (normalize total gradient, not per-element)
2. No synth grad norm + aggressive network gradient clipping
3. Scale down per-element grad norm by 1/sqrt(K)
4. Clamp grad magnitude instead of normalizing
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

from complex_synthesizer import freq_phase_to_complex_params
from ddsp.synths.complex_sine_synth import SynthCoreFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

N_SINES = 16
FS = 44100
N_TIME = 1024
N_STEPS_NET = 10000

rng = np.random.RandomState(SEED)
target_freqs = [2 ** rng.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
target_phases = [rng.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
target_amps = [rng.uniform(0.1, 1.0) for _ in range(N_SINES)]

t = torch.arange(N_TIME, device=device, dtype=torch.float32) / FS
target_signal = torch.zeros(1, N_TIME, device=device)
for freq, phase, amp in zip(target_freqs, target_phases, target_amps):
    target_signal += amp * torch.cos(2 * math.pi * freq * t + phase).unsqueeze(0)

mse_loss = nn.MSELoss()


# ── Variant 1: Global gradient normalization ──
class SynthCoreGlobalNorm(torch.autograd.Function):
    """Normalize total gradient L2 norm (not per-element)."""
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

        # Global L2 normalization: normalize entire gradient tensor to unit norm
        if gf is not None:
            norm_f = gf.abs().pow(2).sum().sqrt() + 1e-8
            gf = gf / norm_f
        if gp is not None:
            norm_p = gp.abs().pow(2).sum().sqrt() + 1e-8
            gp = gp / norm_p

        return gf, gp, ga_s, ga_e, None


# ── Variant 2: No grad norm in synth ──
class SynthCoreNoNorm(torch.autograd.Function):
    """Straight-through gradients (no normalization)."""
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


# ── Variant 3: Per-element norm scaled by 1/sqrt(K) ──
class SynthCoreScaledNorm(torch.autograd.Function):
    """Per-element but scaled down by sqrt(n_sinusoids)."""
    @staticmethod
    def forward(ctx, z_freq, z_phase, amp_start, amp_end, signal_length):
        return SynthCoreFunction.forward(ctx, z_freq, z_phase, amp_start, amp_end, signal_length)

    @staticmethod
    def backward(ctx, grad_output):
        z_freq, z_phase, amp_start, amp_end = ctx.saved_tensors
        signal_length = ctx.signal_length
        n_sines = z_freq.shape[1]
        scale = 1.0 / math.sqrt(n_sines)

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
            gf = scale * gf / (gf.abs() + 1e-8)
        if gp is not None:
            gp = scale * gp / (gp.abs() + 1e-8)
        return gf, gp, ga_s, ga_e, None


# ── Variant 4: Clamp max gradient magnitude ──
class SynthCoreClampNorm(torch.autograd.Function):
    """Clamp per-element gradient magnitude to max 1, but preserve smaller values."""
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
            mag = gf.abs().clamp(min=1e-8)
            gf = gf * torch.clamp(1.0 / mag, max=1.0)  # only scale down, never up
        if gp is not None:
            mag = gp.abs().clamp(min=1e-8)
            gp = gp * torch.clamp(1.0 / mag, max=1.0)
        return gf, gp, ga_s, ga_e, None


# ── MLP test harness ──
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
        return out.unsqueeze(-1)  # [B, 6K, 1]


def test_mlp(name, synth_core_cls, n_steps=N_STEPS_NET, lr=1e-3, grad_clip=1.0):
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

        z_freq = torch.complex(freq_re, freq_im)
        z_phase = torch.complex(phase_re, phase_im)

        signal = synth_core_cls.apply(z_freq, z_phase, amp_start, amp_end, N_TIME)
        loss = mse_loss(signal, target_signal)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        losses.append(loss.item())

        if step % 2000 == 0:
            # Check grad norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"  [{name}] Step {step:>5d}: loss = {loss.item():.6f} | grad_norm = {total_norm:.4f}")

    print(f"  [{name}] Final: {losses[-1]:.6f}")
    return losses


# ══════════════════════════════════════════════════════════════════════
# Run all variants
# ══════════════════════════════════════════════════════════════════════

results = {}

variants = [
    ("V1: Global norm (lr=1e-3, clip=1.0)", SynthCoreGlobalNorm, 1e-3, 1.0),
    ("V2: No norm (lr=1e-3, clip=1.0)", SynthCoreNoNorm, 1e-3, 1.0),
    ("V2b: No norm (lr=1e-3, clip=0.1)", SynthCoreNoNorm, 1e-3, 0.1),
    ("V2c: No norm (lr=1e-4, clip=0.1)", SynthCoreNoNorm, 1e-4, 0.1),
    ("V3: Scaled norm (lr=1e-3, clip=1.0)", SynthCoreScaledNorm, 1e-3, 1.0),
    ("V3b: Scaled norm (lr=1e-4, clip=1.0)", SynthCoreScaledNorm, 1e-4, 1.0),
    ("V4: Clamp norm (lr=1e-3, clip=1.0)", SynthCoreClampNorm, 1e-3, 1.0),
    ("V4b: Clamp norm (lr=1e-4, clip=1.0)", SynthCoreClampNorm, 1e-4, 1.0),
    ("V1b: Global norm (lr=1e-4, clip=1.0)", SynthCoreGlobalNorm, 1e-4, 1.0),
]

for name, cls, lr, clip in variants:
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    results[name] = test_mlp(name, cls, lr=lr, grad_clip=clip)

# Save
with open("phase2_results.json", "w") as f:
    json.dump(results, f)

print("\n" + "="*70)
print("PHASE 2 SUMMARY")
print("="*70)
for name, losses in results.items():
    final = losses[-1]
    mn = min(losses)
    status = "NaN!" if math.isnan(final) else ("OK" if final < 1.0 else "stuck")
    print(f"  {status:5s} {name:50s} | final={final:.6f} | min={mn:.6f}")
