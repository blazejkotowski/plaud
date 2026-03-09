import math
import torch
import torch.nn.functional as F

from .synths import BaseSynth, register_synth


@register_synth
class ComplexSineSynth(BaseSynth):
  """
  A synthesiser that generates sinusoids using complex exponentials.

  Uses z_phase * z_freq^n formulation where z_freq and z_phase are unit complex.

  Following Esteban's approach: the decoder outputs angles (in radians), and we
  construct unit complex numbers as exp(i*θ) = cos(θ) + i*sin(θ).
  This guarantees they're always on the unit circle.

  Parameters layout (all real-valued from decoder, interpreted as angles/amplitudes):
    - omega:      [0, n_sines)       - frequency as radians per sample
    - phi:        [n_sines, 2*n_sines)  - phase offset in radians
    - amp_start:  [2*n_sines, 3*n_sines) - amplitude at frame start
    - amp_end:    [3*n_sines, 4*n_sines) - amplitude at frame end

  Arguments:
    - n_sines: int, the number of sinusoids
    - fs: int, the sampling rate of the input signal
    - resampling_factor: int, the internal up / down sampling factor for the signal
    - device: str, the device to use
  """

  def __init__(self, n_sines: int = 32, fs: int = 44100, resampling_factor: int = 32, device: str = 'cuda'):
    super().__init__()
    self._fs = fs
    self._resampling_factor = resampling_factor
    self._device = device
    self.n_sines = n_sines

    # For overlap-add inference: store the "tail" from previous chunk
    # persistent=False prevents saving to checkpoint
    self.register_buffer("_ola_tail", torch.empty(0), persistent=False)
    self._ola_initialized = False


  @property
  def n_params(self):
    # omega (frequency angle), phi (phase angle), amp_start, amp_end
    return 4 * self.n_sines


  @property
  def jit_name(self):
    return "ComplexSineSynth"


  def reset_ola(self):
    """Reset the overlap-add tail buffer for starting a new audio generation."""
    self._ola_initialized = False
    self._ola_tail = torch.empty(0)


  def forward(self, parameters: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted parameters.
    Args:
      - parameters: torch.Tensor[batch_size, n_params, n_frames], real-valued parameters
    Returns:
      - signal: torch.Tensor[batch_size, 1, sig_length], the synthesized signal

    Uses overlap-add synthesis with Hann window in both training and inference
    for consistent behavior. Streaming tail buffer is only used during inference.
    """
    k = self.n_sines

    # Extract angles and amplitudes
    omega = parameters[:, :k, :] * math.pi  # frequency: radians per sample [0, ~2π]
    phi = parameters[:, k:2*k, :] * math.pi  # phase: radians [0, ~2π]
    amp_starts = parameters[:, 2*k:3*k, :]
    amp_ends = parameters[:, 3*k:, :]

    # Construct unit complex numbers from angles
    z_freqs = torch.complex(torch.cos(omega), torch.sin(omega))
    z_phases = torch.complex(torch.cos(phi), torch.sin(phi))

    # Use overlap-add synthesis in both modes
    signal = self._overlap_add_synthesis(z_freqs, z_phases, amp_starts, amp_ends, parameters.device)

    return signal.unsqueeze(1)


  def _overlap_add_synthesis(self, z_freq, z_phase, amp_start, amp_end, device):
    """
    Overlap-add synthesis: generate extended windowed frames and sum them.

    Each frame is synthesized with 2x length, Hann windowed, then overlap-added
    at 50% overlap (hop = samples_per_frame).

    Uses F.fold for vectorized, gradient-friendly overlap-add.
    """
    batch_size, num_sinusoids, n_frames = z_freq.shape
    samples_per_frame = self._resampling_factor
    window_size = 2 * samples_per_frame  # 50% overlap
    hop_size = samples_per_frame
    output_length = samples_per_frame * n_frames

    # Hann window (sqrt-Hann for COLA property: sum of squared windows = 1)
    window = torch.hann_window(window_size, device=device, dtype=amp_start.dtype)

    # Time indices within each window: [window_size]
    n = torch.arange(window_size, device=device, dtype=torch.float32)
    n_expanded = n.view(1, window_size, 1, 1)

    # Expand parameters for broadcasting
    z_freq_expanded = z_freq.unsqueeze(1)    # [batch, 1, n_sin, n_frames]
    z_phase_expanded = z_phase.unsqueeze(1)  # [batch, 1, n_sin, n_frames]

    # Compute z_phase * z_freq^n for extended window
    # Result: [batch, window_size, n_sin, n_frames]
    z_combined = z_phase_expanded * (z_freq_expanded ** n_expanded)
    sinusoids = torch.real(z_combined)

    # Amplitude envelope over the window (linear interpolation extended)
    t = torch.linspace(0, 1, window_size, device=device, dtype=amp_start.dtype)
    t_expanded = t.view(1, window_size, 1, 1)
    amp_start_expanded = amp_start.unsqueeze(1)  # [batch, 1, n_sin, n_frames]
    amp_end_expanded = amp_end.unsqueeze(1)
    amp_env = amp_start_expanded * (1 - t_expanded) + amp_end_expanded * t_expanded

    # Apply amplitude envelope
    sinusoids = sinusoids * amp_env

    # Sum over sinusoids: [batch, window_size, n_frames]
    frames_summed = sinusoids.sum(dim=2)

    # Apply Hann window: [batch, window_size, n_frames]
    frames_windowed = frames_summed * window.view(1, window_size, 1)

    # Reshape for F.fold: [batch, window_size, n_frames] -> [batch, window_size, n_frames]
    # F.fold expects [batch, C*kernel_size, L] where L = n_frames
    # We treat window_size as kernel_size and sum along it
    frames_for_fold = frames_windowed  # [batch, window_size, n_frames]

    # Output size for fold: we want output_length + samples_per_frame to capture all overlap
    fold_output_size = output_length + samples_per_frame

    # Use F.fold for vectorized overlap-add
    # fold expects input shape [batch, C * kernel_size, num_blocks]
    # Here C=1, kernel_size=window_size, num_blocks=n_frames
    output = F.fold(
      frames_for_fold,  # [batch, window_size, n_frames]
      output_size=(1, fold_output_size),
      kernel_size=(1, window_size),
      stride=(1, hop_size)
    )  # [batch, 1, 1, fold_output_size]

    output = output.squeeze(1).squeeze(1)  # [batch, fold_output_size]

    # Handle streaming (inference only)
    if not self.training:
      if self._ola_initialized and self._ola_tail.shape[0] == batch_size:
        output = output.clone()
        output[:, :samples_per_frame] = output[:, :samples_per_frame] + self._ola_tail

      # Save tail for next chunk
      self._ola_tail = output[:, output_length:output_length + samples_per_frame].clone().detach()
      self._ola_initialized = True

    # Return only the valid portion
    return output[:, :output_length]


  def get_frequencies_hz(self, parameters: torch.Tensor) -> torch.Tensor:
    """
    Extract frequencies in Hz from parameters.

    Args:
      - parameters: torch.Tensor[batch_size, n_params, n_frames]
    Returns:
      - frequencies: torch.Tensor[batch_size, n_sines, n_frames] in Hz
    """
    k = self.n_sines
    omega = parameters[:, :k, :] * math.pi  # radians per sample
    freq_hz = omega * self._fs / (2 * math.pi)
    return freq_hz


  def get_amplitudes(self, parameters: torch.Tensor) -> tuple:
    """
    Extract amplitude envelopes from parameters.

    Args:
      - parameters: torch.Tensor[batch_size, n_params, n_frames]
    Returns:
      - (amp_starts, amp_ends): tuple of [batch_size, n_sines, n_frames] tensors
    """
    k = self.n_sines
    amp_starts = parameters[:, 2*k:3*k, :]
    amp_ends = parameters[:, 3*k:, :]
    return amp_starts, amp_ends


  def extract_sinusoid_data(self, parameters: torch.Tensor) -> dict:
    """
    Extract all sinusoid data from decoder parameters: frequencies, amplitudes, and phases.

    Args:
      - parameters: torch.Tensor[batch_size, n_params, n_frames]
    Returns:
      - dict with keys:
        - 'frequencies_hz': torch.Tensor[batch_size, n_sines, n_frames] in Hz
        - 'amplitudes_mean': torch.Tensor[batch_size, n_sines, n_frames] mean of amp_start and amp_end
        - 'amplitudes_start': torch.Tensor[batch_size, n_sines, n_frames]
        - 'amplitudes_end': torch.Tensor[batch_size, n_sines, n_frames]
        - 'phases_rad': torch.Tensor[batch_size, n_sines, n_frames] phase in radians
    """
    k = self.n_sines
    frequencies_hz = self.get_frequencies_hz(parameters)
    amp_starts, amp_ends = self.get_amplitudes(parameters)
    amplitudes_mean = (amp_starts + amp_ends) / 2.0

    # Extract phase (phi) in radians
    phi = parameters[:, k:2*k, :] * math.pi

    return {
      'frequencies_hz': frequencies_hz,
      'amplitudes_mean': amplitudes_mean,
      'amplitudes_start': amp_starts,
      'amplitudes_end': amp_ends,
      'phases_rad': phi,
    }


class SynthCoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_freq, z_phase, amp_start, amp_end, signal_length):
        ctx.save_for_backward(z_freq, z_phase, amp_start, amp_end)
        ctx.signal_length = signal_length

        device = z_freq.device
        # z_freq: [batch_size, n_sinusoids, n_frames] (complex)
        batch_size, num_sinusoids, n_frames = z_freq.shape
        samples_per_frame = signal_length // n_frames

        # Time indices within each frame: [samples_per_frame]
        n = torch.arange(samples_per_frame, device=device, dtype=torch.float32)

        # Expand for broadcasting:
        # z_freq: [batch, n_sin, n_frames] -> [batch, 1, n_sin, n_frames]
        # n: [samples_per_frame] -> [1, samples_per_frame, 1, 1]
        z_freq_expanded = z_freq.unsqueeze(1)       # [batch, 1, n_sin, n_frames]
        z_phase_expanded = z_phase.unsqueeze(1)     # [batch, 1, n_sin, n_frames]
        n_expanded = n.view(1, samples_per_frame, 1, 1)

        # Compute z_phase * z_freq^n per frame
        # Result: [batch, samples_per_frame, n_sin, n_frames]
        z_combined = z_phase_expanded * (z_freq_expanded ** n_expanded)
        sinusoids = torch.real(z_combined)
        # print("sinusoids.shape", sinusoids.shape)

        # Apply amplitude envelope within each frame
        if amp_start is not None and amp_end is not None:
            t = torch.linspace(0, 1, samples_per_frame, device=device)
            # amp_start: [batch, n_sin, n_frames] -> [batch, 1, n_sin, n_frames]
            amp_start_expanded = amp_start.unsqueeze(1)
            amp_end_expanded = amp_end.unsqueeze(1)
            t_expanded = t.view(1, samples_per_frame, 1, 1)
            amp_env = amp_start_expanded * (1 - t_expanded) + amp_end_expanded * t_expanded
            sinusoids = sinusoids * amp_env

        # print("sinusoids after amp env shape", sinusoids.shape)

        # Sum over sinusoids: [batch, samples_per_frame, n_frames]
        sinusoids_summed = sinusoids.sum(dim=2)

        # print("sinusoids_summed shape", sinusoids_summed.shape)

        # Reshape to interleave frames into a continuous signal:
        # [batch, samples_per_frame, n_frames] -> [batch, n_frames, samples_per_frame] -> [batch, signal_length]
        signal = sinusoids_summed.permute(0, 2, 1).reshape(batch_size, signal_length)
        # print("signal shape:", signal.shape)

        return signal

    @staticmethod
    def backward(ctx, grad_output):
        z_freq, z_phase, amp_start, amp_end = ctx.saved_tensors
        signal_length = ctx.signal_length
        device = z_freq.device

        # We'll let autograd compute true grads, then normalize them.
        with torch.enable_grad():
            z_freq_ = z_freq.clone().detach().requires_grad_(True)
            z_phase_ = z_phase.clone().detach().requires_grad_(True)

            amp_start_ = None if amp_start is None else amp_start.clone().detach().requires_grad_(True)
            amp_end_ = None if amp_end is None else amp_end.clone().detach().requires_grad_(True)

            signal = SynthCoreFunction.forward(ctx, z_freq_, z_phase_, amp_start_, amp_end_, signal_length)
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
