#!/usr/bin/env python
"""
Phase 1c: Validate re/im pair parameterization (network-compatible approach).

Key insight from Phase 1a/b:
- Raw complex params work → they ARE (re, im) pairs
- Angle params fail → cos/sin kills gradient information (1 DOF on unit circle)
- Solution: decoder outputs (re, im) pairs → synth constructs complex numbers

We also test through a small MLP (Phase 2 preview) and multi-frame (Phase 3 preview).
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
N_STEPS = 20000

# Same target
rng = np.random.RandomState(SEED)
target_freqs = [2 ** rng.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
target_phases = [rng.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
target_amps = [rng.uniform(0.1, 1.0) for _ in range(N_SINES)]

t = torch.arange(N_TIME, device=device, dtype=torch.float32) / FS
target_signal = torch.zeros(1, N_TIME, device=device)
for freq, phase, amp in zip(target_freqs, target_phases, target_amps):
    target_signal += amp * torch.cos(2 * math.pi * freq * t + phase).unsqueeze(0)

mse_loss = nn.MSELoss()


class SynthCoreGradNorm(torch.autograd.Function):
    """SynthCoreFunction with gradient normalization ON."""
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


# ══════════════════════════════════════════════════════════════════════
# TEST G: Real/Imag pairs (like decoder would output)
# n_params = 6K: freq_re(K) + freq_im(K) + phase_re(K) + phase_im(K) + amp_s(K) + amp_e(K)
# ══════════════════════════════════════════════════════════════════════

def test_reim_direct(name, n_steps, lr, opt_cls=optim.SGD):
    """Direct optimization of re/im pairs."""
    rng_init = np.random.RandomState(123)
    init_freqs = [2 ** rng_init.uniform(np.log2(50), np.log2(16000)) for _ in range(N_SINES)]
    init_phases = [rng_init.uniform(0, 2 * np.pi) for _ in range(N_SINES)]
    init_amps = [rng_init.uniform(0.1, 1.0) for _ in range(N_SINES)]

    # Convert to complex, then split into re/im
    z_freq, z_phase = freq_phase_to_complex_params(init_freqs, init_phases, FS)
    z_freq, z_phase = z_freq.squeeze(0).to(device), z_phase.squeeze(0).to(device)  # [K]

    # Create 6K parameter tensor [6K, 1] for single frame
    params_init = torch.cat([
        z_freq.real.unsqueeze(-1),   # freq_re [K, 1]
        z_freq.imag.unsqueeze(-1),   # freq_im [K, 1]
        z_phase.real.unsqueeze(-1),  # phase_re [K, 1]
        z_phase.imag.unsqueeze(-1),  # phase_im [K, 1]
        torch.tensor(init_amps, device=device).unsqueeze(-1),  # amp_start [K, 1]
        torch.tensor(init_amps, device=device).unsqueeze(-1),  # amp_end [K, 1]
    ], dim=0)  # [6K, 1]

    params_p = nn.Parameter(params_init)
    optimizer = opt_cls([params_p], lr=lr)
    K = N_SINES
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        freq_re = params_p[:K]        # [K, 1]
        freq_im = params_p[K:2*K]
        phase_re = params_p[2*K:3*K]
        phase_im = params_p[3*K:4*K]
        amp_start = torch.abs(params_p[4*K:5*K])  # Ensure positive
        amp_end = torch.abs(params_p[5*K:6*K])

        z_freq = torch.complex(freq_re, freq_im)
        z_phase = torch.complex(phase_re, phase_im)

        signal = SynthCoreGradNorm.apply(
            z_freq.unsqueeze(0), z_phase.unsqueeze(0),
            amp_start.unsqueeze(0), amp_end.unsqueeze(0), N_TIME)

        loss = mse_loss(signal, target_signal)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 5000 == 0:
            print(f"  [{name}] Step {step:>5d}: loss = {loss.item():.6f}")

    print(f"  [{name}] Final: {losses[-1]:.6f}")
    return losses

print("="*70)
print("TEST G: Re/Im pairs direct optimization")
print("="*70)

print("\n--- SGD lr=0.001 ---")
losses_g_sgd = test_reim_direct("G (re/im, SGD)", N_STEPS, 0.001, optim.SGD)

print("\n--- Adam lr=1e-4 ---")
losses_g_adam = test_reim_direct("G (re/im, Adam)", N_STEPS, 1e-4, optim.Adam)

print("\n--- Adam lr=1e-3 ---")
losses_g_adam2 = test_reim_direct("G (re/im, Adam 1e-3)", N_STEPS, 1e-3, optim.Adam)


# ══════════════════════════════════════════════════════════════════════
# TEST H: Re/Im pairs through a small MLP (Phase 2 preview)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST H: MLP → Re/Im pairs → synth (network test)")
print("="*70)

class TinyDecoder(nn.Module):
    """Minimal MLP that outputs synth params as re/im pairs."""
    def __init__(self, input_dim, n_sines, hidden=256):
        super().__init__()
        n_params = 6 * n_sines  # re/im for freq and phase + amp_start + amp_end
        self.n_sines = n_sines
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, n_params),
        )

    def forward(self, x):
        """x: [B, input_dim] → params: [B, 6K, 1]"""
        out = self.net(x)  # [B, 6K]
        return out.unsqueeze(-1)  # [B, 6K, 1]


def test_mlp_reim(name, n_steps=10000, lr=1e-3):
    K = N_SINES
    model = TinyDecoder(input_dim=64, n_sines=K, hidden=256).to(device)

    # Fixed input (we're testing if network can learn to produce synth params)
    fixed_input = torch.randn(1, 64, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        params = model(fixed_input)  # [1, 6K, 1]

        freq_re = params[:, :K]
        freq_im = params[:, K:2*K]
        phase_re = params[:, 2*K:3*K]
        phase_im = params[:, 3*K:4*K]
        amp_start = torch.abs(params[:, 4*K:5*K]) + 1e-6
        amp_end = torch.abs(params[:, 5*K:6*K]) + 1e-6

        z_freq = torch.complex(freq_re, freq_im)
        z_phase = torch.complex(phase_re, phase_im)

        signal = SynthCoreGradNorm.apply(z_freq, z_phase, amp_start, amp_end, N_TIME)

        loss = mse_loss(signal, target_signal)
        loss.backward()

        # Gradient clipping (matching full network)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        losses.append(loss.item())

        if step % 2000 == 0:
            print(f"  [{name}] Step {step:>5d}: loss = {loss.item():.6f}")

    print(f"  [{name}] Final: {losses[-1]:.6f}")
    return losses

losses_h_1e3 = test_mlp_reim("H (MLP, Adam 1e-3)", n_steps=10000, lr=1e-3)
losses_h_1e4 = test_mlp_reim("H (MLP, Adam 1e-4)", n_steps=10000, lr=1e-4)


# ══════════════════════════════════════════════════════════════════════
# TEST I: Multi-frame direct optimization (Phase 3 preview)
# Generate 8 frames × 1024 samples, same sinusoids but continuous
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST I: Multi-frame direct optimization (8 frames × 1024)")
print("="*70)

N_FRAMES = 8
SIGNAL_LEN = N_FRAMES * N_TIME  # 8192 samples

# Target: same sinusoids across all frames (continuous signal)
t_long = torch.arange(SIGNAL_LEN, device=device, dtype=torch.float32) / FS
target_multi = torch.zeros(1, SIGNAL_LEN, device=device)
for freq, phase, amp in zip(target_freqs, target_phases, target_amps):
    target_multi += amp * torch.cos(2 * math.pi * freq * t_long + phase).unsqueeze(0)

def test_multiframe_reim(name, n_steps=20000, lr=1e-3, opt_cls=optim.Adam):
    K = N_SINES
    rng_init = np.random.RandomState(123)
    init_freqs = [2 ** rng_init.uniform(np.log2(50), np.log2(16000)) for _ in range(K)]
    init_phases = [rng_init.uniform(0, 2 * np.pi) for _ in range(K)]
    init_amps = [rng_init.uniform(0.1, 1.0) for _ in range(K)]

    z_freq, z_phase = freq_phase_to_complex_params(init_freqs, init_phases, FS)
    z_freq, z_phase = z_freq.squeeze(0).to(device), z_phase.squeeze(0).to(device)

    # n_frames copies (each frame gets its own params)
    params_init = torch.cat([
        z_freq.real.unsqueeze(-1).expand(-1, N_FRAMES),   # [K, F]
        z_freq.imag.unsqueeze(-1).expand(-1, N_FRAMES),
        z_phase.real.unsqueeze(-1).expand(-1, N_FRAMES),
        z_phase.imag.unsqueeze(-1).expand(-1, N_FRAMES),
        torch.tensor(init_amps, device=device).unsqueeze(-1).expand(-1, N_FRAMES),
        torch.tensor(init_amps, device=device).unsqueeze(-1).expand(-1, N_FRAMES),
    ], dim=0).clone()  # [6K, F]

    params_p = nn.Parameter(params_init)
    optimizer = opt_cls([params_p], lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        freq_re = params_p[:K]
        freq_im = params_p[K:2*K]
        phase_re = params_p[2*K:3*K]
        phase_im = params_p[3*K:4*K]
        amp_start = torch.abs(params_p[4*K:5*K]) + 1e-6
        amp_end = torch.abs(params_p[5*K:6*K]) + 1e-6

        z_freq = torch.complex(freq_re, freq_im)
        z_phase = torch.complex(phase_re, phase_im)

        signal = SynthCoreGradNorm.apply(
            z_freq.unsqueeze(0), z_phase.unsqueeze(0),
            amp_start.unsqueeze(0), amp_end.unsqueeze(0),
            SIGNAL_LEN)

        loss = mse_loss(signal, target_multi)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 5000 == 0:
            print(f"  [{name}] Step {step:>5d}: loss = {loss.item():.6f}")

    print(f"  [{name}] Final: {losses[-1]:.6f}")
    return losses

losses_i_adam = test_multiframe_reim("I (multi-frame, Adam)", n_steps=20000, lr=1e-3)
losses_i_sgd = test_multiframe_reim("I (multi-frame, SGD)", n_steps=20000, lr=1e-3, opt_cls=optim.SGD)


# ══════════════════════════════════════════════════════════════════════
# TEST J: Multi-frame through MLP
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST J: Multi-frame MLP (8 frames × 1024)")
print("="*70)

class TinyTemporalDecoder(nn.Module):
    """MLP + GRU that outputs per-frame synth params as re/im pairs."""
    def __init__(self, input_dim, n_sines, n_frames, hidden=256):
        super().__init__()
        n_params = 6 * n_sines
        self.n_sines = n_sines
        self.n_frames = n_frames
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_params)

    def forward(self, x):
        """x: [B, n_frames, input_dim] → params: [B, 6K, n_frames]"""
        h = self.net(x)            # [B, F, hidden]
        h, _ = self.gru(h)         # [B, F, hidden]
        out = self.output(h)       # [B, F, 6K]
        return out.permute(0, 2, 1)  # [B, 6K, F]


def test_mlp_multiframe(name, n_steps=10000, lr=1e-3):
    K = N_SINES
    model = TinyTemporalDecoder(input_dim=64, n_sines=K, n_frames=N_FRAMES, hidden=256).to(device)
    fixed_input = torch.randn(1, N_FRAMES, 64, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        params = model(fixed_input)  # [1, 6K, F]

        freq_re = params[:, :K]
        freq_im = params[:, K:2*K]
        phase_re = params[:, 2*K:3*K]
        phase_im = params[:, 3*K:4*K]
        amp_start = torch.abs(params[:, 4*K:5*K]) + 1e-6
        amp_end = torch.abs(params[:, 5*K:6*K]) + 1e-6

        z_freq = torch.complex(freq_re, freq_im)
        z_phase = torch.complex(phase_re, phase_im)

        signal = SynthCoreGradNorm.apply(z_freq, z_phase, amp_start, amp_end, SIGNAL_LEN)
        loss = mse_loss(signal, target_multi)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 2000 == 0:
            print(f"  [{name}] Step {step:>5d}: loss = {loss.item():.6f}")

    print(f"  [{name}] Final: {losses[-1]:.6f}")
    return losses

losses_j = test_mlp_multiframe("J (MLP multi-frame)", n_steps=10000, lr=1e-3)
losses_j2 = test_mlp_multiframe("J2 (MLP multi-frame, 1e-4)", n_steps=10000, lr=1e-4)


# ══════════════════════════════════════════════════════════════════════
# Save all results
# ══════════════════════════════════════════════════════════════════════
results = {
    "G_reim_SGD": losses_g_sgd,
    "G_reim_Adam_1e4": losses_g_adam,
    "G_reim_Adam_1e3": losses_g_adam2,
    "H_mlp_Adam_1e3": losses_h_1e3,
    "H_mlp_Adam_1e4": losses_h_1e4,
    "I_multiframe_Adam": losses_i_adam,
    "I_multiframe_SGD": losses_i_sgd,
    "J_mlp_multiframe_1e3": losses_j,
    "J_mlp_multiframe_1e4": losses_j2,
}
with open("phase1c_results.json", "w") as f:
    json.dump(results, f)

print("\n" + "="*70)
print("PHASE 1c SUMMARY")
print("="*70)
for name, losses in results.items():
    final = losses[-1]
    mn = min(losses)
    print(f"  {name:40s} | final={final:.6f} | min={mn:.6f}")
