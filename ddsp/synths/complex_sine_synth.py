"""
ComplexSineSynth – PLAUD-compatible synthesizer using Esteban's complex-surrogate
frequency estimation via exponentiation on the unit circle.

Core synthesis identity (MUST NOT CHANGE):
    z_combined = z_phase * z_freq ** n
    sinusoids  = Re(z_combined)

Each control frame is synthesised as an independent mini-segment, exactly
matching Esteban's original SynthCoreFunction:
  - z_freq  is constant within the frame (unit-circle phasor encoding frequency)
  - z_phase is constant within the frame (unit-circle phasor encoding start phase)
  - amplitude is linearly interpolated from the current to the next frame
  - n = 0, 1, ..., R-1  where R = resampling_factor

The decoder learns to predict phase-continuous z_phase values across frames —
that is the key insight of the complex-surrogate approach. No cumulative
products or manual phase accumulation are needed.

Parameter layout (per control-rate time-step, per sinusoid):
    0 : z_freq real part  }  → normalized to unit circle inside forward
    1 : z_freq imag part  }
    2 : z_phase real part }  → normalized to unit circle inside forward
    3 : z_phase imag part }
    4 : amplitude (real, ≥0 after scaled sigmoid applied by decoder)

Total n_params = 5 * n_sines
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .synths import BaseSynth, register_synth


# ---------------------------------------------------------------------------
# Pure synthesis function – used by both training (via autograd wrapper) and
# inference/export (called directly, TorchScript-compatible).
# ---------------------------------------------------------------------------

def _synth_forward(
    z_freq: torch.Tensor,
    z_phase: torch.Tensor,
    amplitudes: torch.Tensor,
    resampling_factor: int,
) -> torch.Tensor:
    """Complex-exponential oscillator bank (pure function, no custom grad).

    Each control frame is an independent mini-segment:
        sinusoid[b, k, f, n] = amp_env[b,k,f,n] · Re( z_phase[b,k,f] · z_freq[b,k,f]^n )
    where n ∈ [0, R) and amp_env linearly interpolates amplitude across frames.

    Args:
        z_freq:      [B, K, T_ctl] complex – unit-magnitude frequency phasors
        z_phase:     [B, K, T_ctl] complex – unit-magnitude phase phasors
        amplitudes:  [B, K, T_ctl] float   – per-sinusoid amplitude per frame
        resampling_factor: int – audio samples per control frame (R)
    Returns:
        signal: [B, 1, T_audio]  where T_audio = T_ctl * R
    """
    device = z_freq.device
    B, K, T_ctl = z_freq.shape
    R = resampling_factor

    # Local sample indices within each frame: n = 0, 1, ..., R-1
    n = torch.arange(R, device=device, dtype=torch.float32)       # [R]

    # --- Core identity (DO NOT CHANGE) ---
    # z_combined = z_phase * z_freq^n
    # sinusoids  = Re(z_combined)
    z_freq_expanded  = z_freq.unsqueeze(-1)                       # [B, K, T_ctl, 1]
    z_phase_expanded = z_phase.unsqueeze(-1)                      # [B, K, T_ctl, 1]
    n_expanded       = n.view(1, 1, 1, R)                         # [1, 1, 1, R]

    z_combined = z_phase_expanded * (z_freq_expanded ** n_expanded)  # [B, K, T_ctl, R]
    sinusoids  = torch.real(z_combined)                              # [B, K, T_ctl, R]

    # --- Amplitude envelope (linear interp, matching Esteban's amp_start/amp_end) ---
    t = torch.linspace(0.0, 1.0, R, device=device).view(1, 1, 1, R)  # [1,1,1,R]

    amp_start = amplitudes.unsqueeze(-1)                          # [B, K, T_ctl, 1]
    amp_end = torch.cat(
        [amplitudes[:, :, 1:], amplitudes[:, :, -1:]],
        dim=-1
    ).unsqueeze(-1)                                               # [B, K, T_ctl, 1]
    amp_env = amp_start * (1.0 - t) + amp_end * t                # [B, K, T_ctl, R]

    sinusoids = sinusoids * amp_env                               # [B, K, T_ctl, R]

    # Reshape to audio rate: [B, K, T_ctl * R]
    sinusoids = sinusoids.reshape(B, K, T_ctl * R)

    signal = sinusoids.sum(dim=1, keepdim=True)                   # [B, 1, T_audio]
    return signal


# ---------------------------------------------------------------------------
# Custom autograd Function – gradient normalization for training only
# (not used during export/inference)
# ---------------------------------------------------------------------------

class _ComplexSynthCore(torch.autograd.Function):
    """Wraps _synth_forward with Esteban's gradient normalization on the
    complex z_freq / z_phase parameters.  Only used during training.
    """

    @staticmethod
    def forward(ctx, z_freq, z_phase, amplitudes, resampling_factor):
        ctx.save_for_backward(z_freq, z_phase, amplitudes)
        ctx.resampling_factor = resampling_factor
        return _synth_forward(z_freq, z_phase, amplitudes, resampling_factor)

    @staticmethod
    def backward(ctx, grad_output):
        z_freq, z_phase, amplitudes = ctx.saved_tensors
        R = ctx.resampling_factor

        # Re-run forward under autograd to get true local gradients,
        # then normalize the complex ones (Esteban's trick).
        with torch.enable_grad():
            zf  = z_freq.clone().detach().requires_grad_(True)
            zp  = z_phase.clone().detach().requires_grad_(True)
            amp = amplitudes.clone().detach().requires_grad_(True)

            signal = _synth_forward(zf, zp, amp, R)

            grads = torch.autograd.grad(
                signal, (zf, zp, amp),
                grad_output,
                retain_graph=False,
                allow_unused=True,
            )

        grad_z_freq, grad_z_phase, grad_amp = grads

        # 🔥 Normalize gradients for complex parameters (Esteban's trick)
        if grad_z_freq is not None:
            grad_z_freq = grad_z_freq / (grad_z_freq.abs() + 1e-8)
        if grad_z_phase is not None:
            grad_z_phase = grad_z_phase / (grad_z_phase.abs() + 1e-8)
        # Amplitude gradients flow un-normalised

        return grad_z_freq, grad_z_phase, grad_amp, None


# ---------------------------------------------------------------------------
# Public synth module
# ---------------------------------------------------------------------------

@register_synth
class ComplexSineSynth(BaseSynth):
    """
    PLAUD-compatible sinusoidal synthesizer using the complex-surrogate
    exponential form for frequency estimation.

    The decoder predicts 5 real values per sinusoid per control-rate frame:
        z_freq_re, z_freq_im, z_phase_re, z_phase_im, amplitude

    Each control frame is an independent mini-segment where z_freq and z_phase
    are constant.  The core identity

        sinusoid = Re( z_phase · z_freq^n )    n = 0 .. R-1

    is applied exactly as in Esteban's SynthCoreFunction, with amplitude
    linearly interpolated between consecutive frames.

    The decoder/network is responsible for predicting z_phase values that
    produce phase-continuous audio across frame boundaries — this is what
    the complex-surrogate gradient normalization enables it to learn.

    Args:
        n_sines:           number of sinusoidal components
        fs:                sample rate (Hz)
        resampling_factor: control-rate → audio-rate upsampling factor
        streaming:         whether to maintain phase across calls (not yet impl.)
        device:            target device
    """

    def __init__(
        self,
        n_sines: int = 100,
        fs: int = 44100,
        resampling_factor: int = 32,
        streaming: bool = False,
        device: str = 'cuda',
    ):
        super().__init__(fs=fs, resampling_factor=resampling_factor)
        self._n_sines = n_sines
        self._resampling_factor = resampling_factor
        self.streaming = streaming
        self._device = device

    # ------------------------------------------------------------------
    # BaseSynth interface
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """5 real values per sinusoid: z_freq (re, im), z_phase (re, im), amp."""
        return 5 * self._n_sines

    @property
    def jit_name(self):
        return "ComplexSineSynth"

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        parameters: torch.Tensor,
        limit_components: float = 0.0,
        waveshaping_factor: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            parameters: [B, 5*n_sines, T_ctl]  – predicted by the decoder
                Layout per sinusoid group of 5 channels:
                    [z_freq_re, z_freq_im, z_phase_re, z_phase_im, amplitude]
            limit_components: fraction of sinusoids to mute (0 = use all)
            waveshaping_factor: unused, present for API compat with other synths

        Returns:
            signal: [B, 1, T_audio]  where T_audio = T_ctl * resampling_factor
        """
        B, P, T_ctl = parameters.shape
        K = self._n_sines
        assert P == self.n_params, f"Expected {self.n_params} params, got {P}"

        # ---- unpack --------------------------------------------------------
        # Reshape to [B, n_sines, 5, T_ctl] for easy slicing
        params = parameters.view(B, K, 5, T_ctl)

        z_freq_re  = params[:, :, 0, :]   # [B, K, T_ctl]
        z_freq_im  = params[:, :, 1, :]
        z_phase_re = params[:, :, 2, :]
        z_phase_im = params[:, :, 3, :]
        amp        = params[:, :, 4, :]    # already in [0, ~2] from scaled sigmoid

        # ---- build complex phasors & normalise to unit circle ---------------
        z_freq  = torch.complex(z_freq_re, z_freq_im)         # [B, K, T_ctl]
        z_phase = torch.complex(z_phase_re, z_phase_im)       # [B, K, T_ctl]

        # Project onto unit circle (crucial – exponentiation assumes |z|=1)
        z_freq  = z_freq  / (z_freq.abs()  + 1e-8)
        z_phase = z_phase / (z_phase.abs() + 1e-8)

        # ---- optional component limiting -----------------------------------
        limit_components = max(0.0, min(limit_components, 1.0))
        if limit_components > 0:
            max_sines = max(int(K * (1 - limit_components)), 1)
            _, indices = torch.topk(amp.mean(dim=-1), max_sines, dim=1)
            mask = torch.zeros(B, K, 1, device=amp.device)
            mask.scatter_(1, indices.unsqueeze(-1), 1.0)
            amp = amp * mask.expand_as(amp)

        # ---- synthesize ----------------------------------------------------
        # Training: use custom autograd Function for gradient normalization.
        # Inference/export: call the pure function directly (TorchScript-safe).
        #
        # NOTE: torch.jit.is_scripting() is evaluated at *compile* time by
        # TorchScript, so the _ComplexSynthCore branch is completely removed
        # from the scripted graph.  A plain `if self.training:` is NOT enough
        # because TorchScript still parses (and rejects) both branches.
        if torch.jit.is_scripting():
            signal = _synth_forward(z_freq, z_phase, amp, self._resampling_factor)
        elif self.training:
            signal = _ComplexSynthCore.apply(z_freq, z_phase, amp, self._resampling_factor)
        else:
            signal = _synth_forward(z_freq, z_phase, amp, self._resampling_factor)

        return signal
