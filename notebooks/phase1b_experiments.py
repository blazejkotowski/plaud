#!/usr/bin/env python
"""
Phase 1b: Test angle parameterization WITH gradient normalization enabled.

Phase 1 showed grad norm is critical. Now test if it fixes angle-based params too.
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

from complex_synthesizer import (
    ComplexFreqPhaseSynthesizer,
    freq_phase_to_complex_params,
)
from ddsp.synths.complex_sine_synth import ComplexSineSynth, SynthCoreFunction
from ddsp.blocks import _scaled_sigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

N_SINES = 16
FS = 44100
N_TIME = 1024
N_STEPS = 20000
LR = 0.001

# Same target as Phase 1
rng = np.random.RandomState(SEED)
target_freqs = [2 ** rng.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
target_phases = [rng.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
target_amps = [rng.uniform(0.1, 1.0) for _ in range(N_SINES)]

t = torch.arange(N_TIME, device=device, dtype=torch.float32) / FS
target_signal = torch.zeros(1, N_TIME, device=device)
for freq, phase, amp in zip(target_freqs, target_phases, target_amps):
    target_signal += amp * torch.cos(2 * math.pi * freq * t + phase).unsqueeze(0)

mse_loss = nn.MSELoss()

# ── Patched SynthCore with grad norm ON ──
class SynthCoreGradNorm(torch.autograd.Function):
    """Our SynthCoreFunction with gradient normalization RE-ENABLED."""
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

        grad_z_freq, grad_z_phase, grad_amp_start, grad_amp_end = grads

        # Re-enabled gradient normalization
        if grad_z_freq is not None:
            grad_z_freq = grad_z_freq / (grad_z_freq.abs() + 1e-8)
        if grad_z_phase is not None:
            grad_z_phase = grad_z_phase / (grad_z_phase.abs() + 1e-8)

        return grad_z_freq, grad_z_phase, grad_amp_start, grad_amp_end, None


def run_test(name, setup_fn, synth_fn, n_steps=N_STEPS, lr=LR, optimizer_cls=optim.SGD):
    params, extra = setup_fn()
    optimizer = optimizer_cls(params, lr=lr)
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        output = synth_fn(params, extra)
        loss = mse_loss(output, target_signal)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 5000 == 0:
            print(f"  [{name}] Step {step:>5d}: loss = {loss.item():.6f}")
    print(f"  [{name}] Final: {losses[-1]:.6f}")
    return losses


# ══════════════════════════════════════════════════════════════════════
# TEST C2: Angle params + grad norm ON in SynthCore
# ══════════════════════════════════════════════════════════════════════
print("="*70)
print("TEST C2: Angle params + grad norm ON (SGD)")
print("="*70)

def setup_c2():
    rng_init = np.random.RandomState(123)
    init_freqs = [2 ** rng_init.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
    init_phases = [rng_init.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
    init_amps = [rng_init.uniform(0.1, 1.0) for _ in range(N_SINES)]

    omega_init = torch.tensor([2 * math.pi * f / FS for f in init_freqs], device=device)
    phi_init = torch.tensor(init_phases, device=device, dtype=torch.float32)

    omega_p = nn.Parameter(omega_init.unsqueeze(-1))
    phi_p = nn.Parameter(phi_init.unsqueeze(-1))
    amp_start_p = nn.Parameter(torch.tensor(init_amps, device=device).unsqueeze(-1))
    amp_end_p = nn.Parameter(torch.tensor(init_amps, device=device).unsqueeze(-1))
    return [omega_p, phi_p, amp_start_p, amp_end_p], None

def synth_c2(params, extra):
    omega_p, phi_p, amp_start_p, amp_end_p = params
    z_freq = torch.complex(torch.cos(omega_p), torch.sin(omega_p))
    z_phase = torch.complex(torch.cos(phi_p), torch.sin(phi_p))
    return SynthCoreGradNorm.apply(
        z_freq.unsqueeze(0), z_phase.unsqueeze(0),
        amp_start_p.unsqueeze(0), amp_end_p.unsqueeze(0), N_TIME)

losses_c2 = run_test("C2 (angle+gradnorm)", setup_c2, synth_c2)


# ══════════════════════════════════════════════════════════════════════
# TEST D2: _scaled_sigmoid + grad norm ON
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST D2: _scaled_sigmoid + grad norm ON (SGD)")
print("="*70)

def setup_d2():
    pre_omega = nn.Parameter(torch.randn(N_SINES, 1, device=device) * 2.0)
    pre_phi = nn.Parameter(torch.randn(N_SINES, 1, device=device) * 2.0)
    pre_amp_start = nn.Parameter(torch.randn(N_SINES, 1, device=device))
    pre_amp_end = nn.Parameter(torch.randn(N_SINES, 1, device=device))
    return [pre_omega, pre_phi, pre_amp_start, pre_amp_end], None

def synth_d2(params, extra):
    pre_omega, pre_phi, pre_amp_start, pre_amp_end = params
    omega = _scaled_sigmoid(pre_omega) * math.pi
    phi = _scaled_sigmoid(pre_phi) * math.pi
    amp_start = _scaled_sigmoid(pre_amp_start)
    amp_end = _scaled_sigmoid(pre_amp_end)
    z_freq = torch.complex(torch.cos(omega), torch.sin(omega))
    z_phase = torch.complex(torch.cos(phi), torch.sin(phi))
    return SynthCoreGradNorm.apply(
        z_freq.unsqueeze(0), z_phase.unsqueeze(0),
        amp_start.unsqueeze(0), amp_end.unsqueeze(0), N_TIME)

losses_d2 = run_test("D2 (scaled_sig+gradnorm)", setup_d2, synth_d2)


# ══════════════════════════════════════════════════════════════════════
# TEST E_RAW2: Raw angle (no activation) + grad norm ON
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST E_RAW2: Raw (no activation) + grad norm ON (SGD)")
print("="*70)

def setup_eraw2():
    pre_omega = nn.Parameter(torch.randn(N_SINES, 1, device=device) * 2.0)
    pre_phi = nn.Parameter(torch.randn(N_SINES, 1, device=device) * 2.0)
    pre_amp_start = nn.Parameter(torch.randn(N_SINES, 1, device=device))
    pre_amp_end = nn.Parameter(torch.randn(N_SINES, 1, device=device))
    return [pre_omega, pre_phi, pre_amp_start, pre_amp_end], None

def synth_eraw2(params, extra):
    pre_omega, pre_phi, pre_amp_start, pre_amp_end = params
    z_freq = torch.complex(torch.cos(pre_omega), torch.sin(pre_omega))
    z_phase = torch.complex(torch.cos(pre_phi), torch.sin(pre_phi))
    return SynthCoreGradNorm.apply(
        z_freq.unsqueeze(0), z_phase.unsqueeze(0),
        pre_amp_start.unsqueeze(0), pre_amp_end.unsqueeze(0), N_TIME)

losses_eraw2 = run_test("E_RAW2 (raw+gradnorm)", setup_eraw2, synth_eraw2)


# ══════════════════════════════════════════════════════════════════════
# TEST: Adam optimizer variants (matching full network training)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST ADAM: Raw complex + grad norm + Adam lr=1e-4")
print("="*70)

def setup_adam():
    rng_init = np.random.RandomState(123)
    init_freqs = [2 ** rng_init.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
    init_phases = [rng_init.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
    init_amps = [rng_init.uniform(0.1, 1.0) for _ in range(N_SINES)]
    z_freq, z_phase = freq_phase_to_complex_params(init_freqs, init_phases, FS)
    z_freq, z_phase = z_freq.to(device), z_phase.to(device)
    z_freq_p = nn.Parameter(z_freq.squeeze(0).unsqueeze(-1))
    z_phase_p = nn.Parameter(z_phase.squeeze(0).unsqueeze(-1))
    amp_start_p = nn.Parameter(torch.tensor(init_amps, device=device).unsqueeze(-1))
    amp_end_p = nn.Parameter(torch.tensor(init_amps, device=device).unsqueeze(-1))
    return [z_freq_p, z_phase_p, amp_start_p, amp_end_p], None

def synth_adam(params, extra):
    z_freq_p, z_phase_p, amp_start_p, amp_end_p = params
    return SynthCoreGradNorm.apply(
        z_freq_p.unsqueeze(0), z_phase_p.unsqueeze(0),
        amp_start_p.unsqueeze(0), amp_end_p.unsqueeze(0), N_TIME)

losses_adam = run_test("ADAM (raw+gradnorm)", setup_adam, synth_adam, lr=1e-4, optimizer_cls=optim.Adam)

print("\n" + "="*70)
print("TEST ADAM2: Angle + grad norm + Adam lr=1e-4")
print("="*70)

losses_adam_angle = run_test("ADAM2 (angle+gradnorm)", setup_c2, synth_c2, lr=1e-4, optimizer_cls=optim.Adam)

print("\n" + "="*70)
print("TEST ADAM3: _scaled_sigmoid + grad norm + Adam lr=1e-4")
print("="*70)

losses_adam_ss = run_test("ADAM3 (ss+gradnorm)", setup_d2, synth_d2, lr=1e-4, optimizer_cls=optim.Adam)


# Save results
results = {
    "C2_angle_gradnorm": losses_c2,
    "D2_scaledsig_gradnorm": losses_d2,
    "ERAW2_raw_gradnorm": losses_eraw2,
    "ADAM_raw_gradnorm": losses_adam,
    "ADAM2_angle_gradnorm": losses_adam_angle,
    "ADAM3_ss_gradnorm": losses_adam_ss,
}
with open("phase1b_results.json", "w") as f:
    json.dump(results, f)

print("\n" + "="*70)
print("PHASE 1b SUMMARY")
print("="*70)
for name, losses in results.items():
    print(f"  {name:40s} | final={losses[-1]:.6f} | min={min(losses):.6f}")
