import math
import torch
import torch.nn.functional as F

from .synths import BaseSynth, register_synth


@register_synth
class SimpleSineSynth(BaseSynth):
  """
  Additive sinusoidal synthesizer using cumulative-sum phase and torch.sin.

  Per frame per sinusoid the decoder predicts (in raw/table format):
    - omega: [0, n_sines)           — instantaneous frequency in rad/sample
    - amp:   [n_sines, 2*n_sines)   — amplitude (positive after softplus)

  Using omega (rad/sample) as the native frequency format keeps the param
  table, bin selection, and distillation infrastructure identical to
  ComplexSineSynth — only the synthesis step differs.

  Synthesis:
    1. Upsample ω to audio rate via linear interpolation
    2. Compute instantaneous phase via cumulative sum: φ(n) = Σ ω(n)
    3. Generate audio: x(n) = Σ_k  a_k(n) · sin(φ_k(n))

  This is the standard differentiable additive synth (DDSP-style) without
  complex exponentials. Phase continuity is implicit in the cumsum.
  """

  def __init__(self, n_sines: int = 32, fs: int = 44100,
               resampling_factor: int = 32, device: str = 'cuda'):
    super().__init__()
    self._fs = fs
    self._resampling_factor = resampling_factor
    self._device = device
    self.n_sines = n_sines

  @property
  def n_params(self):
    return 2 * self.n_sines

  @property
  def jit_name(self):
    return "SimpleSineSynth"

  @property
  def raw_output(self) -> bool:
    return True

  @property
  def params_per_sine(self) -> int:
    return 2

  def forward(self, parameters: torch.Tensor, limit_components: float = 0.0) -> torch.Tensor:
    """
    Args:
      parameters: [B, n_params, T_ctl]  — raw values: omega (rad/sample), amp (raw)
      limit_components: fraction of sinusoids to mute (0=all active)
    Returns:
      signal: [B, 1, n_signal]
    """
    k = self.n_sines
    T = parameters.shape[-1]
    n_signal = self._resampling_factor * T

    # --- activations ---
    # Frequency: raw value IS omega (rad/sample), take abs for positive freq
    omega = parameters[:, :k, :].abs()                               # [B, k, T]
    # Amplitude: softplus / n_sines (same scaling as ComplexSineSynth)
    amp = F.softplus(parameters[:, k:, :]) / k                      # [B, k, T]

    # --- limit components ---
    if limit_components > 0:
      max_sines = max(int(k * (1 - limit_components)), 1)
      mean_amp = amp.mean(dim=-1)  # [B, k]
      _, top_idx = torch.topk(mean_amp, max_sines, dim=1)
      mask = torch.zeros_like(amp)
      mask.scatter_(1, top_idx.unsqueeze(-1).expand(-1, -1, T), 1)
      amp = amp * mask

    # --- upsample to audio rate ---
    # [B, k, T] → [B, k, n_signal]
    omega_up = F.interpolate(omega, size=n_signal, mode='linear', align_corners=True)
    amp_up = F.interpolate(amp, size=n_signal, mode='linear', align_corners=True)

    # --- synthesis ---
    phase = torch.cumsum(omega_up, dim=-1)             # cumulative phase
    signal = (amp_up * torch.sin(phase)).sum(dim=1)    # [B, n_signal]

    return signal.unsqueeze(1)  # [B, 1, n_signal]

  # ----- inspection helpers (match ComplexSineSynth API) -----

  def get_frequencies_hz(self, parameters: torch.Tensor) -> torch.Tensor:
    """[B, n_params, T] → [B, n_sines, T] in Hz."""
    omega = parameters[:, :self.n_sines, :].abs()
    return omega * self._fs / (2 * math.pi)

  def get_amplitudes(self, parameters: torch.Tensor) -> tuple:
    """Returns (amp, amp) tuple for API compat (no start/end distinction)."""
    amp = F.softplus(parameters[:, self.n_sines:, :]) / self.n_sines
    return amp, amp

  def extract_sinusoid_data(self, parameters: torch.Tensor) -> dict:
    k = self.n_sines
    freq_hz = self.get_frequencies_hz(parameters)
    amp = F.softplus(parameters[:, k:, :]) / k
    return {
      'frequencies_hz': freq_hz,
      'amplitudes_mean': amp,
      'amplitudes_start': amp,
      'amplitudes_end': amp,
      'phases_rad': None,  # phase is implicit (cumsum)
    }
