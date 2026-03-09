import math
import torch

from .synths import BaseSynth, register_synth


@register_synth
class CumprodSineSynth(BaseSynth):
  """
  Sinusoidal synthesiser using cumulative product for continuous phase.

  Unlike ComplexSineSynth, this synth does NOT predict an independent phase
  per frame. Instead, phase accumulates continuously as the integral of
  instantaneous frequency across the entire signal. This makes the synth
  equivalent to a bank of real sinusoidal oscillators and ensures that the
  learned frequencies/amplitudes are directly usable for real-sinusoid
  resynthesis.

  Uses the complex surrogate (cumprod form) for differentiable optimisation:
    z[t] = cumprod(exp(i * omega[t]))
    signal = Re(z) * amplitude

  Parameters layout (from decoder, [0, ~2] range via scaled_sigmoid):
    - omega:      [0, n_sines)                 - instantaneous frequency (radians/sample after scaling)
    - amp_start:  [n_sines, 2*n_sines)         - amplitude at frame start
    - amp_end:    [2*n_sines, 3*n_sines)       - amplitude at frame end

  No phase parameter — phase is derived from frequency accumulation.
  """

  def __init__(self, n_sines: int = 32, fs: int = 44100, resampling_factor: int = 32, device: str = 'cuda'):
    super().__init__(fs=fs, resampling_factor=resampling_factor)
    self._device = device
    self.n_sines = n_sines

  @property
  def n_params(self):
    # omega, amp_start, amp_end (no phase)
    return 3 * self.n_sines

  @property
  def jit_name(self):
    return "CumprodSineSynth"

  def forward(self, parameters: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted parameters using cumulative product.

    Args:
      - parameters: torch.Tensor[batch_size, n_params, n_frames], real-valued parameters
    Returns:
      - signal: torch.Tensor[batch_size, 1, sig_length], the synthesized signal
    """
    k = self.n_sines
    samples_per_frame = self._resampling_factor
    n_frames = parameters.shape[-1]
    signal_length = samples_per_frame * n_frames
    batch_size = parameters.shape[0]

    # Extract parameters
    omega = parameters[:, :k, :] * math.pi      # [batch, n_sines, n_frames] radians/sample
    amp_starts = parameters[:, k:2*k, :]         # [batch, n_sines, n_frames]
    amp_ends = parameters[:, 2*k:3*k, :]         # [batch, n_sines, n_frames]

    # Upsample omega from frame-rate to sample-rate via linear interpolation
    # [batch, n_sines, n_frames] -> [batch, n_sines, signal_length]
    omega_upsampled = torch.nn.functional.interpolate(
      omega, size=signal_length, mode='linear', align_corners=True
    )  # [batch, n_sines, signal_length]

    # Build unit complex phasors from instantaneous frequency
    z_inst = torch.complex(torch.cos(omega_upsampled), torch.sin(omega_upsampled))

    # Cumulative product to accumulate phase continuously
    # Prepend a 1+0j to start at zero phase, then truncate last sample
    initial = torch.ones(batch_size, k, 1, dtype=z_inst.dtype, device=z_inst.device)
    z_cat = torch.cat([initial, z_inst[:, :, :-1]], dim=-1)  # [batch, n_sines, signal_length]
    z_accum = torch.cumprod(z_cat, dim=-1)  # [batch, n_sines, signal_length]

    # Take the real part -> cosine oscillators
    sinusoids = z_accum.real  # [batch, n_sines, signal_length]

    # Build amplitude envelope: linear interpolation within each frame
    # [batch, n_sines, n_frames] -> [batch, n_sines, signal_length]
    amp_starts_up = amp_starts.repeat_interleave(samples_per_frame, dim=-1)
    amp_ends_up = amp_ends.repeat_interleave(samples_per_frame, dim=-1)
    t_frame = torch.linspace(0, 1, samples_per_frame, device=parameters.device)
    t_frame = t_frame.repeat(n_frames)  # [signal_length]
    amp_env = amp_starts_up * (1 - t_frame) + amp_ends_up * t_frame  # [batch, n_sines, signal_length]

    # Apply amplitude and sum over sinusoids
    signal = (sinusoids * amp_env).sum(dim=1)  # [batch, signal_length]

    return signal.unsqueeze(1)

  def get_frequencies_hz(self, parameters: torch.Tensor) -> torch.Tensor:
    """Extract frequencies in Hz from parameters."""
    k = self.n_sines
    omega = parameters[:, :k, :] * math.pi  # radians per sample
    freq_hz = omega * self._fs / (2 * math.pi)
    return freq_hz

  def get_amplitudes(self, parameters: torch.Tensor) -> tuple:
    """Extract amplitude envelopes from parameters."""
    k = self.n_sines
    amp_starts = parameters[:, k:2*k, :]
    amp_ends = parameters[:, 2*k:3*k, :]
    return amp_starts, amp_ends

  def extract_sinusoid_data(self, parameters: torch.Tensor) -> dict:
    """Extract all sinusoid data from decoder parameters."""
    k = self.n_sines
    frequencies_hz = self.get_frequencies_hz(parameters)
    amp_starts, amp_ends = self.get_amplitudes(parameters)
    amplitudes_mean = (amp_starts + amp_ends) / 2.0

    return {
      'frequencies_hz': frequencies_hz,
      'amplitudes_mean': amplitudes_mean,
      'amplitudes_start': amp_starts,
      'amplitudes_end': amp_ends,
      'phases_rad': None,  # No explicit phase - it's derived from frequency
    }

  # def compute_frequency_stability_loss(self, parameters: torch.Tensor,
  #                                       freq_dev_offset: float = 20.0,
  #                                       freq_dev_slope: float = 0.01) -> torch.Tensor:
  #   """
  #   Compute a loss that penalizes frequency jumps between consecutive frames.
  #   Uses linear frequency-dependent threshold: offset + slope * freq_hz
  #   """
  #   k = self.n_sines
  #   omega = parameters[:, :k, :]
  #   freq_hz = omega * self._fs / 2

  #   freq_diff = torch.diff(freq_hz, dim=-1)
  #   freq_diff_abs = torch.abs(freq_diff)

  #   freq_at_boundaries = (freq_hz[:, :, :-1] + freq_hz[:, :, 1:]) / 2
  #   threshold_hz = freq_dev_offset + freq_dev_slope * freq_at_boundaries.abs()

  #   excess = torch.relu(freq_diff_abs - threshold_hz)
  #   return excess.mean()
