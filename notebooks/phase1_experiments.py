#!/usr/bin/env python
"""
Phase 1: Single-Frame Direct Optimization — Synth Comparison
============================================================

Tests A-E comparing Esteban's ComplexFreqPhaseSynthesizer vs our ComplexSineSynth
on a single-frame direct parameter optimization task.

Target: 16 sinusoids, 1024 samples, fs=44100, MSE loss only.
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

# Esteban's code
from complex_synthesizer import (
    ComplexFreqPhaseSynthesizer,
    freq_phase_to_complex_params,
    SynthCoreFunction as EstebanSynthCore,
)

# Our code
from ddsp.synths.complex_sine_synth import ComplexSineSynth, SynthCoreFunction as OurSynthCore
from ddsp.blocks import _scaled_sigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Reproducible setup ──
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Target signal parameters ──
N_SINES = 16
FS = 44100
N_TIME = 1024
N_STEPS = 20000
LR = 0.001  # SGD lr matching Esteban's notebook

# Generate target sinusoids with known frequencies, phases, amplitudes
rng = np.random.RandomState(SEED)
target_freqs = [2 ** rng.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
target_phases = [rng.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
target_amps = [rng.uniform(0.1, 1.0) for _ in range(N_SINES)]

print(f"\nTarget: {N_SINES} sinusoids, {N_TIME} samples, fs={FS}")
print(f"Frequencies: {[f'{f:.1f}' for f in sorted(target_freqs)]}")

# Generate target signal analytically
t = torch.arange(N_TIME, device=device, dtype=torch.float32) / FS
target_signal = torch.zeros(1, N_TIME, device=device)
for freq, phase, amp in zip(target_freqs, target_phases, target_amps):
    target_signal += amp * torch.cos(2 * math.pi * freq * t + phase).unsqueeze(0)

print(f"Target signal: min={target_signal.min():.4f}, max={target_signal.max():.4f}, rms={target_signal.pow(2).mean().sqrt():.4f}")

mse_loss = nn.MSELoss()

def run_optimization(name, setup_fn, synth_fn, n_steps=N_STEPS, lr=LR, optimizer_cls=optim.SGD):
    """Generic optimization loop. Returns loss history."""
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

    print(f"  [{name}] Final loss: {losses[-1]:.6f}")
    return losses


# ══════════════════════════════════════════════════════════════════════
# TEST A: Esteban's ComplexFreqPhaseSynthesizer (baseline — should work)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST A: Esteban's ComplexFreqPhaseSynthesizer (raw complex params, SGD)")
print("="*70)

esteban_synth = ComplexFreqPhaseSynthesizer(N_TIME)

def setup_a():
    # Random starting frequencies and phases
    rng_init = np.random.RandomState(123)
    init_freqs = [2 ** rng_init.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
    init_phases = [rng_init.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
    init_amps = [rng_init.uniform(0.1, 1.0) for _ in range(N_SINES)]

    z_freq, z_phase = freq_phase_to_complex_params(init_freqs, init_phases, FS)
    z_freq, z_phase = z_freq.to(device), z_phase.to(device)

    z_freq_p = nn.Parameter(z_freq.squeeze(0))
    z_phase_p = nn.Parameter(z_phase.squeeze(0))
    amp_start_p = nn.Parameter(torch.tensor(init_amps, device=device, dtype=torch.float32))
    amp_end_p = nn.Parameter(torch.tensor(init_amps, device=device, dtype=torch.float32))

    return [z_freq_p, z_phase_p, amp_start_p, amp_end_p], None

def synth_a(params, extra):
    z_freq_p, z_phase_p, amp_start_p, amp_end_p = params
    return esteban_synth(
        z_freq_p.unsqueeze(0), z_phase_p.unsqueeze(0),
        amp_start_p.unsqueeze(0), amp_end_p.unsqueeze(0)
    )

losses_a = run_optimization("Test A (Esteban)", setup_a, synth_a)


# ══════════════════════════════════════════════════════════════════════
# TEST B: Our SynthCoreFunction with raw complex params, single frame
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST B: Our SynthCoreFunction (raw complex params, 1 frame, SGD)")
print("="*70)

def setup_b():
    rng_init = np.random.RandomState(123)
    init_freqs = [2 ** rng_init.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
    init_phases = [rng_init.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
    init_amps = [rng_init.uniform(0.1, 1.0) for _ in range(N_SINES)]

    z_freq, z_phase = freq_phase_to_complex_params(init_freqs, init_phases, FS)
    z_freq, z_phase = z_freq.to(device), z_phase.to(device)

    # Shape for our SynthCore: [batch, n_sines, n_frames=1]
    z_freq_p = nn.Parameter(z_freq.squeeze(0).unsqueeze(-1))   # [K, 1]
    z_phase_p = nn.Parameter(z_phase.squeeze(0).unsqueeze(-1))  # [K, 1]
    amp_start_p = nn.Parameter(torch.tensor(init_amps, device=device).unsqueeze(-1))  # [K, 1]
    amp_end_p = nn.Parameter(torch.tensor(init_amps, device=device).unsqueeze(-1))    # [K, 1]

    return [z_freq_p, z_phase_p, amp_start_p, amp_end_p], None

def synth_b(params, extra):
    z_freq_p, z_phase_p, amp_start_p, amp_end_p = params
    # [1, K, 1] tensors
    signal = OurSynthCore.apply(
        z_freq_p.unsqueeze(0), z_phase_p.unsqueeze(0),
        amp_start_p.unsqueeze(0), amp_end_p.unsqueeze(0),
        N_TIME  # signal_length = 1024 (1 frame × 1024 samples)
    )
    return signal  # [1, N_TIME]

losses_b = run_optimization("Test B (Ours, raw complex)", setup_b, synth_b)


# ══════════════════════════════════════════════════════════════════════
# TEST C: Our synth with angle parameterization (no _scaled_sigmoid)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST C: Our SynthCoreFunction (angle params, no _scaled_sigmoid, SGD)")
print("="*70)

def setup_c():
    rng_init = np.random.RandomState(123)
    init_freqs = [2 ** rng_init.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
    init_phases = [rng_init.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
    init_amps = [rng_init.uniform(0.1, 1.0) for _ in range(N_SINES)]

    # Convert freqs to omega (rad/sample)
    omega_init = torch.tensor([2 * math.pi * f / FS for f in init_freqs], device=device)
    phi_init = torch.tensor(init_phases, device=device, dtype=torch.float32)

    omega_p = nn.Parameter(omega_init.unsqueeze(-1))  # [K, 1]
    phi_p = nn.Parameter(phi_init.unsqueeze(-1))      # [K, 1]
    amp_start_p = nn.Parameter(torch.tensor(init_amps, device=device).unsqueeze(-1))
    amp_end_p = nn.Parameter(torch.tensor(init_amps, device=device).unsqueeze(-1))

    return [omega_p, phi_p, amp_start_p, amp_end_p], None

def synth_c(params, extra):
    omega_p, phi_p, amp_start_p, amp_end_p = params
    # Construct complex from angles
    z_freq = torch.complex(torch.cos(omega_p), torch.sin(omega_p))
    z_phase = torch.complex(torch.cos(phi_p), torch.sin(phi_p))
    signal = OurSynthCore.apply(
        z_freq.unsqueeze(0), z_phase.unsqueeze(0),
        amp_start_p.unsqueeze(0), amp_end_p.unsqueeze(0),
        N_TIME
    )
    return signal

losses_c = run_optimization("Test C (angles, no scaled_sigmoid)", setup_c, synth_c)


# ══════════════════════════════════════════════════════════════════════
# TEST D: Our synth with _scaled_sigmoid activation (matches decoder)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST D: Our SynthCoreFunction (_scaled_sigmoid × π, SGD)")
print("="*70)

def setup_d():
    # Initialize in the pre-sigmoid domain
    # _scaled_sigmoid(x) ≈ 2*sigmoid(x)^2.303
    # We want omega ≈ random freq angles, so we need to invert
    rng_init = np.random.RandomState(123)

    # For now, just use random initialization in pre-sigmoid space
    # _scaled_sigmoid(0) ≈ 0.4, so omega = 0.4*π ≈ 1.26 → freq ≈ 8.8 kHz
    pre_omega = nn.Parameter(torch.randn(N_SINES, 1, device=device) * 2.0)  # broader init
    pre_phi = nn.Parameter(torch.randn(N_SINES, 1, device=device) * 2.0)
    pre_amp_start = nn.Parameter(torch.randn(N_SINES, 1, device=device))
    pre_amp_end = nn.Parameter(torch.randn(N_SINES, 1, device=device))

    return [pre_omega, pre_phi, pre_amp_start, pre_amp_end], None

def synth_d(params, extra):
    pre_omega, pre_phi, pre_amp_start, pre_amp_end = params
    # Apply _scaled_sigmoid to get [0, ~2] range
    omega = _scaled_sigmoid(pre_omega) * math.pi
    phi = _scaled_sigmoid(pre_phi) * math.pi
    amp_start = _scaled_sigmoid(pre_amp_start)
    amp_end = _scaled_sigmoid(pre_amp_end)

    z_freq = torch.complex(torch.cos(omega), torch.sin(omega))
    z_phase = torch.complex(torch.cos(phi), torch.sin(phi))

    signal = OurSynthCore.apply(
        z_freq.unsqueeze(0), z_phase.unsqueeze(0),
        amp_start.unsqueeze(0), amp_end.unsqueeze(0),
        N_TIME
    )
    return signal

losses_d = run_optimization("Test D (_scaled_sigmoid)", setup_d, synth_d)


# ══════════════════════════════════════════════════════════════════════
# TEST E: Alternative activations
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST E: Alternative activations")
print("="*70)

def make_activation_test(name, freq_act, phase_act, amp_act):
    """Create setup/synth functions for a given activation."""
    def setup():
        pre_omega = nn.Parameter(torch.randn(N_SINES, 1, device=device) * 2.0)
        pre_phi = nn.Parameter(torch.randn(N_SINES, 1, device=device) * 2.0)
        pre_amp_start = nn.Parameter(torch.randn(N_SINES, 1, device=device))
        pre_amp_end = nn.Parameter(torch.randn(N_SINES, 1, device=device))
        return [pre_omega, pre_phi, pre_amp_start, pre_amp_end], None

    def synth(params, extra):
        pre_omega, pre_phi, pre_amp_start, pre_amp_end = params
        omega = freq_act(pre_omega)
        phi = phase_act(pre_phi)
        amp_start = amp_act(pre_amp_start)
        amp_end = amp_act(pre_amp_end)

        z_freq = torch.complex(torch.cos(omega), torch.sin(omega))
        z_phase = torch.complex(torch.cos(phi), torch.sin(phi))

        signal = OurSynthCore.apply(
            z_freq.unsqueeze(0), z_phase.unsqueeze(0),
            amp_start.unsqueeze(0), amp_end.unsqueeze(0),
            N_TIME
        )
        return signal

    return setup, synth

activations = {
    "E1: sigmoid×2π": (
        lambda x: torch.sigmoid(x) * 2 * math.pi,
        lambda x: torch.sigmoid(x) * 2 * math.pi,
        lambda x: torch.sigmoid(x) * 2.0,
    ),
    "E2: tanh×π+π": (
        lambda x: torch.tanh(x) * math.pi + math.pi,  # [0, 2π]
        lambda x: torch.tanh(x) * math.pi + math.pi,
        lambda x: torch.tanh(x) + 1.0,  # [0, 2]
    ),
    "E3: softplus": (
        lambda x: torch.nn.functional.softplus(x),  # [0, ∞), will clamp later
        lambda x: torch.nn.functional.softplus(x),
        lambda x: torch.nn.functional.softplus(x),
    ),
    "E4: raw (no activation)": (
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ),
}

losses_e = {}
for act_name, (f_act, p_act, a_act) in activations.items():
    print(f"\n--- {act_name} ---")
    setup, synth = make_activation_test(act_name, f_act, p_act, a_act)
    losses_e[act_name] = run_optimization(act_name, setup, synth)


# ══════════════════════════════════════════════════════════════════════
# TEST F: Grad normalization sweep on raw complex (Test B variant)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST F: Gradient normalization variants")
print("="*70)

# Test B with grad normalization re-enabled
def setup_f():
    return setup_b()  # same init as Test B

def synth_f_with_grad_norm(params, extra):
    """Same as synth_b but we manually normalize grads after backward."""
    return synth_b(params, extra)

# We need a custom training loop for this
def run_with_grad_norm(name, n_steps=N_STEPS, lr=LR):
    params, _ = setup_b()
    optimizer = optim.SGD(params, lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        output = synth_b(params, None)
        loss = mse_loss(output, target_signal)
        loss.backward()

        # Normalize gradients on complex params (freq, phase)
        for p in params[:2]:  # z_freq, z_phase
            if p.grad is not None:
                p.grad = p.grad / (p.grad.abs() + 1e-8)

        optimizer.step()
        losses.append(loss.item())

        if step % 5000 == 0:
            print(f"  [{name}] Step {step:>5d}: loss = {loss.item():.6f}")

    print(f"  [{name}] Final loss: {losses[-1]:.6f}")
    return losses

losses_f_no_norm = losses_b  # already computed
print("\n--- With grad normalization ---")
losses_f_with_norm = run_with_grad_norm("Test F (grad norm ON)")


# ══════════════════════════════════════════════════════════════════════
# Save all results
# ══════════════════════════════════════════════════════════════════════
results = {
    "A_esteban": losses_a,
    "B_ours_raw_complex": losses_b,
    "C_angle_no_act": losses_c,
    "D_scaled_sigmoid": losses_d,
    "F_grad_norm_on": losses_f_with_norm,
    "F_grad_norm_off": losses_f_no_norm,
}
for k, v in losses_e.items():
    results[k] = v

# Save to JSON
output_path = os.path.join(os.path.dirname(__file__), "phase1_results.json")
with open(output_path, "w") as f:
    json.dump(results, f)
print(f"\nResults saved to: {output_path}")

# Print summary
print("\n" + "="*70)
print("PHASE 1 SUMMARY")
print("="*70)
for name, losses in results.items():
    print(f"  {name:40s} | final_loss={losses[-1]:.6f} | min_loss={min(losses):.6f}")
