"""
Low-level DSP utilities used by the harmonic + noise DDSP synthesizers.

References:
  - Engel et al., "DDSP: Differentiable Digital Signal Processing", ICLR 2020
  - github.com/acids-ircam/ddsp_pytorch (ACIDS implementation)
"""

import math
import torch
import torch.nn.functional as F


def scale_function(x: torch.Tensor) -> torch.Tensor:
    """Modified sigmoid that maps decoder outputs to positive amplitudes.

    Applies sigmoid then scales so the output starts near 0 for large negative
    inputs and grows roughly linearly for positive inputs:

        y = 2 * sigmoid(x) ** log(10) + 1e-7

    This avoids the hard saturation of plain sigmoid and gives a useful
    gradient landscape for amplitude prediction.
    """
    return 2.0 * torch.sigmoid(x) ** math.log(10) + 1e-7


def remove_above_nyquist(amplitudes: torch.Tensor,
                         pitch: torch.Tensor,
                         sampling_rate: int) -> torch.Tensor:
    """Zero out harmonic amplitudes whose frequencies exceed Nyquist.

    Args:
      amplitudes: [B, T, n_harmonics]  — per-harmonic amplitude envelopes
      pitch:      [B, T, 1]            — fundamental frequency in Hz
      sampling_rate: int

    Returns:
      amplitudes with above-Nyquist harmonics zeroed.
    """
    n_harmonics = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harmonics + 1, device=pitch.device, dtype=pitch.dtype)  # [B,T,H]
    mask = (pitches < sampling_rate / 2.0).to(amplitudes.dtype)
    return amplitudes * mask


def upsample(signal: torch.Tensor, factor: int) -> torch.Tensor:
    """Upsample a control-rate tensor along the time axis (dim=1).

    Args:
      signal: [B, T_ctl, C]
      factor: integer upsample ratio

    Returns:
      [B, T_ctl * factor, C]
    """
    # F.interpolate expects [B, C, T]
    signal = signal.permute(0, 2, 1)
    signal = F.interpolate(signal, scale_factor=float(factor), mode="linear")
    return signal.permute(0, 2, 1)


def amp_to_impulse_response(amp: torch.Tensor, target_size: int) -> torch.Tensor:
    """Convert frequency-domain amplitudes to a time-domain impulse response.

    Follows the ACIDS DDSP reference: irfft (no n=), roll, window, pad,
    roll back.

    Args:
      amp: [B, T, n_bands]  — one-sided magnitude spectrum (positive freqs)
      target_size: int — desired time-domain length

    Returns:
      ir: [B, T, target_size]
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], dim=-1)  # complex pair
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)  # [B, T, filter_size]  (filter_size = 2*(n_bands-1))

    filter_size = amp.shape[-1]

    # Center the impulse, apply Hann window, then pad to target_size
    amp = torch.roll(amp, filter_size // 2, dims=-1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)
    amp = amp * win

    amp = F.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -(filter_size // 2), dims=-1)

    return amp


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """FFT convolution between signal and kernel, following ACIDS DDSP reference.

    Both tensors are [B, T_frames, frame_size].

    Args:
      signal: [B, T, frame_size]
      kernel: [B, T, frame_size]

    Returns:
      [B, T, frame_size]  — convolved signal (same shape)
    """
    # Pad signal on the right, kernel on the left
    signal = F.pad(signal, (0, signal.shape[-1]))
    kernel = F.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output
