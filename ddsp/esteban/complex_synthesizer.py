import torch
import torch.nn as nn
import math

class SynthCoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_freq, z_phase, amp_start, amp_end, n_time):
        ctx.save_for_backward(z_freq, z_phase, amp_start, amp_end)
        ctx.n_time = n_time

        device = z_freq.device
        batch_size, num_sinusoids = z_freq.shape
        n = torch.arange(n_time, device=device, dtype=torch.float32)

        # Expand shapes for broadcasting
        z_freq_expanded = z_freq.unsqueeze(1)
        z_phase_expanded = z_phase.unsqueeze(1)
        n_expanded = n.view(1, n_time, 1)

        # Compute z_phase * z_freq^n
        z_combined = z_phase_expanded * (z_freq_expanded ** n_expanded)
        sinusoids = torch.real(z_combined)

        # Apply amplitude envelope if provided
        if amp_start is not None and amp_end is not None:
            t = torch.linspace(0, 1, n_time, device=device)
            amp_start_expanded = amp_start.unsqueeze(1)
            amp_end_expanded = amp_end.unsqueeze(1)
            t_expanded = t.view(1, n_time, 1)
            amp_env = amp_start_expanded * (1 - t_expanded) + amp_end_expanded * t_expanded
            sinusoids = sinusoids * amp_env

        signal = sinusoids.sum(dim=-1)  # [batch_size, n_time]
        return signal

    @staticmethod
    def backward(ctx, grad_output):
        z_freq, z_phase, amp_start, amp_end = ctx.saved_tensors
        n_time = ctx.n_time
        device = z_freq.device

        # We’ll let autograd compute true grads, then normalize them.
        with torch.enable_grad():
            z_freq_ = z_freq.clone().detach().requires_grad_(True)
            z_phase_ = z_phase.clone().detach().requires_grad_(True)

            amp_start_ = None if amp_start is None else amp_start.clone().detach().requires_grad_(True)
            amp_end_ = None if amp_end is None else amp_end.clone().detach().requires_grad_(True)

            signal = SynthCoreFunction.forward(ctx, z_freq_, z_phase_, amp_start_, amp_end_, n_time)
            grads = torch.autograd.grad(
                signal,
                (z_freq_, z_phase_, amp_start_, amp_end_),
                grad_output,
                retain_graph=False,
                allow_unused=True
            )

        grad_z_freq, grad_z_phase, grad_amp_start, grad_amp_end = grads

        # 🔥 Normalize the local gradients (only if they exist)
        if grad_z_freq is not None:
            grad_z_freq = grad_z_freq / (grad_z_freq.abs() + 1e-8)
        if grad_z_phase is not None:
            grad_z_phase = grad_z_phase / (grad_z_phase.abs() + 1e-8)

        return grad_z_freq, grad_z_phase, grad_amp_start, grad_amp_end, None

class ComplexFreqPhaseSynthesizer(nn.Module):
    def __init__(self, n_time):
        super().__init__()
        self.n_time = n_time

    def forward(self, z_freq, z_phase, amp_start=None, amp_end=None):
        # Use the custom autograd Function
        return SynthCoreFunction.apply(z_freq, z_phase, amp_start, amp_end, self.n_time)

def freq_phase_to_complex_params(frequencies, phases, sample_rate):
    """
    NEW: Convert frequencies and phases to separate z_freq and z_phase parameters.
    
    Args:
        frequencies: [batch_size, num_sinusoids] tensor (Hz)
        phases: [batch_size, num_sinusoids] tensor (radians)
        sample_rate: scalar, sampling rate in Hz
    
    Returns:
        z_freq: [batch_size, num_sinusoids] frequency control parameters
        z_phase: [batch_size, num_sinusoids] phase control parameters
    """
    # Ensure tensors and handle broadcasting
    frequencies = torch.as_tensor(frequencies, dtype=torch.float32)
    phases      = torch.as_tensor(phases, dtype=torch.float32)
    
    # Add batch dimension if needed
    if frequencies.dim() == 1:
        frequencies = frequencies.unsqueeze(0)
    if phases.dim() == 1:
        phases = phases.unsqueeze(0)
    
    # Convert frequency to radians per sample (for z_freq)
    omega = 2 * math.pi * frequencies / sample_rate
    z_freq = torch.complex(torch.cos(omega), torch.sin(omega))
    
    # Convert phase to complex phasor (for z_phase)
    z_phase = torch.complex(torch.cos(phases), torch.sin(phases))
    
    # Verify normalization
    freq_magnitudes  = z_freq.abs()
    phase_magnitudes = z_phase.abs()
    
    assert torch.allclose(freq_magnitudes, torch.ones_like(freq_magnitudes), atol=1e-6), \
        f"Frequency parameters not normalized! Max magnitude: {freq_magnitudes.max()}"
    assert torch.allclose(phase_magnitudes, torch.ones_like(phase_magnitudes), atol=1e-6), \
        f"Phase parameters not normalized! Max magnitude: {phase_magnitudes.max()}"
    
    return z_freq, z_phase

def complex_params_to_freq_phase(z_freq, z_phase, sample_rate):
    """
    NEW: Convert separate z_freq and z_phase back to frequency and phase.
    
    Args:
        z_freq: [batch_size, num_sinusoids] frequency control parameters
        z_phase: [batch_size, num_sinusoids] phase control parameters
        sample_rate: scalar, sampling rate in Hz
    
    Returns:
        frequencies: [batch_size, num_sinusoids] tensor (Hz)
        phases: [batch_size, num_sinusoids] tensor (radians)
    """
    # Extract frequency from z_freq
    omega = torch.angle(z_freq)
    frequencies = omega * sample_rate / (2 * math.pi)
    
    # Extract phase from z_phase
    phases = torch.angle(z_phase)
    
    return frequencies, phases

import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

def fir_lowpass_kernel(cutoff_freq, sample_rate, num_taps=101, window='hamming'):
    nyquist = sample_rate / 2
    fc = cutoff_freq / nyquist

    if num_taps % 2 == 0:
        num_taps += 1

    n = torch.arange(num_taps) - (num_taps - 1) / 2
    h = torch.sinc(2 * fc * n)

    if window.lower() == 'hamming':
        w = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(num_taps) / (num_taps - 1))
    elif window.lower() == 'hann':
        w = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(num_taps) / (num_taps - 1))
    else:
        w = torch.ones(num_taps)
    
    h *= w
    h /= h.sum()
    return h

def batched_fir_lowpass(signals, cutoff_freq=512, sample_rate=44100, num_taps=11):
    batch_size, signal_length = signals.shape
    kernel = fir_lowpass_kernel(cutoff_freq, sample_rate, num_taps).to(signals.device)

    signals = signals.unsqueeze(1)  # [batch, 1, length]
    kernel = kernel.view(1, 1, -1)
    pad = num_taps // 2
    filtered = F.conv1d(F.pad(signals, (pad, pad)), kernel)
    return filtered.squeeze(1)

def plot_filter_response(cutoff_freq, sample_rate, num_taps=101, filename='filter_response.png'):
    kernel = fir_lowpass_kernel(cutoff_freq, sample_rate, num_taps)
    # Compute frequency response via FFT
    freq_response = torch.fft.rfft(kernel, n=4096)
    freq = torch.fft.rfftfreq(4096, 1/sample_rate)
    magnitude = freq_response.abs()

    plt.figure(figsize=(8,4))
    plt.plot(freq, magnitude)
    plt.title(f'FIR Low-Pass Filter Response (cutoff={cutoff_freq} Hz)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename)
    plt.close()
    print(f"Filter response saved to {filename}")

# # Example usage
# if __name__ == "__main__":
#     batch_size = 4
#     signal_length = 1024
#     sample_rate = 44100
#     cutoff = 16  # Hz

#     # Random signals
#     x = torch.randn(batch_size, signal_length)

#     # Filter signals
#     y = batched_fir_lowpass(x, cutoff, sample_rate, num_taps=3)
#     print(y.shape)  # [4, 1024]

#     # Plot frequency response
#     plot_filter_response(cutoff, sample_rate, num_taps=3)
