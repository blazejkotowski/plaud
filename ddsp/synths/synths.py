from torch import nn
import torch
import math
import torchaudio
import torch.nn.functional as F
import numpy as np

from ddsp.filterbank import FilterBank
# from ddsp.sgd.sinusoidal_gradient_descent.core import complex_oscillator


from typing import Type, Callable, Dict, Any, List, Tuple

_SYNTH_REGISTRY = {}

def register_synth(cls):
  _SYNTH_REGISTRY[cls.__name__] = cls

  return cls

class BaseSynth(nn.Module):
  """
  Base class for synthesizers.

  Arguments:
    - fs: int, the sampling rate of the input signal
    - resampling_factor: int, the internal up / down sampling factor for the signal
  """

  def __init__(self, fs: int = 44100, resampling_factor: int = 32):
    super().__init__()
    self._fs = fs
    self._resampling_factor = resampling_factor

  def forward(self, *args, **kwargs):
    raise NotImplementedError

  @property
  def n_params(self) -> int:
    """Returns the number of control parameters"""
    raise NotImplementedError

  @classmethod
  def builder(cls: Type['BaseSynth'], **kwargs) -> Callable[[], 'BaseSynth']:
    """
    Returns a builder function that instantiates the class with the given arguments.

    Usage:
      synth_builder = SineSynth.builder(n_sines=100)
      synth = synth_builder()  # creates a new instance of SineSynth
    """
    def _builder():
      return cls(**kwargs)
    return _builder

  @classmethod
  def to_config(cls, **kwargs) -> dict:
    """Returns a configuration dictionary for instantiating the synth."""
    return {
        "class": cls.__name__,
        "params": kwargs,
    }

  @staticmethod
  def from_config(config: dict) -> "BaseSynth":
    """Instantiates the synth from a config dictionary."""
    cls_name = config["class"]
    params = config["params"]

    cls = _SYNTH_REGISTRY[cls_name]
    return cls(**params)


@register_synth
class HarmonicSynth(BaseSynth):
  """
  Harmonic synthesizer that generates a signal as a sum of harmonically-related sine waves.

  Arguments:
    - n_harmonics: int, the maximum number of harmonics to synthesize
    - fs: int, sampling rate
    - resampling_factor: int, upsampling factor for control parameters
    - streaming: bool, whether to use continuous phase (not implemented here yet)
    - device: str, computation device
  """
  def __init__(self,
              n_harmonics: int = 100,
              fs: int = 44100,
              resampling_factor: int = 32,
              streaming: bool = False,
              device: str = 'cuda'):
    super().__init__(fs=fs, resampling_factor=resampling_factor)
    self._n_harmonics = n_harmonics
    self.streaming = streaming
    self._device = device

  @property
  def n_params(self):
    return self._n_harmonics + 1  # 1 fundamental + n_harmonics amplitudes

  @property
  def jit_name(self):
    return "HarmonicSynth"

  def forward(self, parameters: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a harmonic signal.

    Arguments:
      - parameters: [batch_size, 1+n_harmonics, time], fundamental frequency in Hz and amplitudes per harmonic
    Returns:
      - signal: [batch_size, 1, time], synthesized signal
    """
    pitch = parameters[:, :1, :]
    amplitudes = parameters[:, 1:, :]

    # Upsample inputs to match the target sample rate
    pitch = F.interpolate(pitch, scale_factor=float(self._resampling_factor), mode='linear')
    amplitudes = F.interpolate(amplitudes, scale_factor=float(self._resampling_factor), mode='linear')

    # Compute angular frequency (omega) by integrating pitch over time
    omega = torch.cumsum(2 * math.pi * pitch / self._fs, dim=-1)  # [B, 1, T]

    # Generate harmonic numbers [1, 2, ..., n]
    harmonics = torch.arange(1, self._n_harmonics + 1, device=omega.device).view(1, -1, 1)  # [1, n, 1]

    # Multiply each harmonic index with base omega
    omegas = omega * harmonics  # [B, n_harmonics, T]

    # Compute signal as weighted sum of harmonic sines
    signal = (torch.sin(omegas) * amplitudes).sum(dim=1, keepdim=True)  # [B, 1, T]

    return signal

@register_synth
class SineSynth(BaseSynth):
  """
  Mixture of sinweaves synthesiser.

  Arguments:
    - fs: int, the sampling rate of the input signal
    - n_sines: int, the number of sinewaves to synthesise
    - resampling_factor: int, the internal up / down sampling factor for the sinewaves
    - streaming: bool, whether to run the model in streaming mode
  """
  def __init__(self,
               fs: int = 44100,
               n_sines: int = 500,
               resampling_factor: int = 32,
               streaming: bool = False,
               device: str = 'cuda'):
    super().__init__()
    self._fs = fs
    self._n_sines = n_sines
    self._resampling_factor = resampling_factor
    # self._phases = None
    self.register_buffer("_phases", torch.empty(0))
    self._phases_initialized = False

    self.streaming = streaming
    self._device = device


  @property
  def n_params(self):
    return 2*self._n_sines

  @property
  def jit_name(self):
    return "SineSynth"


  def forward(self, parameters: torch.Tensor, limit_components: float = 0.0):
    """
    Generates a mixture of sinewaves with the given frequencies and amplitudes per sample.

    Arguments:
      - parameters: torch.Tensor[batch_size, n_params, n_samples], the parameters of the synthesizer
      - limit_components: float, the attenuation factor for the number of sinewaves
    """
    # limit_components = kwargs.get('limit_components', 0.0)

    frequencies = parameters[:, :self._n_sines, :] # n_sines frequencies
    amplitudes = parameters[:, self._n_sines:, :] # n_sines amplitudes
    # amplitudes = torch.ones_like(amplitudes) # workaround for now, amplitudes are not used

    batch_size = frequencies.shape[0]

    # We only need to initialise phases buffer if we are in streaming mode
    if self.streaming and (not self._phases_initialized or self._phases.shape[0] != batch_size):
      self._phases = torch.zeros(batch_size, self._n_sines)
      self._phases_initialized = True

    # Upsample from the internal sampling rate to the target sampling rate
    frequencies = F.interpolate(frequencies, scale_factor=float(self._resampling_factor), mode='linear')
    amplitudes = F.interpolate(amplitudes, scale_factor=float(self._resampling_factor), mode='linear')

    # round frequencies to the 2nd decimal place
    # frequencies = torch.round(frequencies * 100) / 100.0
    # Scale the frequencies to the Nyquist frequency
    frequencies = frequencies * self._fs / 4 # range [0, 2] to [0, fs/2]
    # frequencies = frequencies * self._fs / 2 # range [0, 1] to [0, fs/2]

    # print(f'est freq range in batch: {frequencies.min().item()} - {frequencies.max().item()}')

    # cancel the sines above nyquist frequency
    amplitudes *= (frequencies < self._fs / 2).float() + 1e-4

    # Calculate the phase increments
    omegas = frequencies * 2 * math.pi / self._fs

    # Calculate the phases at points, in place
    phases = omegas.cumsum(dim=-1)
    phases = phases % (2 * math.pi)

    if self.streaming:
      # Shift the phases by the last phase from last generation
      # breakpoint()
      phases = (phases.permute(2, 0, 1) + self._phases).permute(1, 2, 0)

      # Copy the last phases for next iteration
      self._phases.copy_(phases[: ,: , -1] % (2 * math.pi))


    # If the limit_components is higher than 0, we need to limit the number of sinewaves
    # by choosing the first max_sines, in relation to their amplitudes
    # clamp limit_components to [0, 1]
    limit_components = max(0.0, min(limit_components, 1.0))
    if limit_components > 0:
      # Calculate the number of sinewaves to keep (at least 1)
      max_sines = int(self._n_sines * (1 - limit_components))
      max_sines = max(max_sines, 1)

      # Get the indices of the max_sines largest amplitudes
      _, indices = torch.topk(amplitudes, max_sines, dim=1)
      # Create a mask of the same shape as the amplitudes
      mask = torch.zeros_like(amplitudes)
      # Set the values at the indices to 1
      mask.scatter_(1, indices, 1)

      # Apply the mask to the frequencies and amplitudes
      frequencies = frequencies * mask
      amplitudes = amplitudes * mask
      phases = phases * mask

    # Generate and sum the sinewaves
    signal = torch.sum(amplitudes * torch.sin(phases), dim=1, keepdim=True)
    return signal

@register_synth
class NoiseBandSynth(BaseSynth):
  """
  A synthesiser that generates a mixture noise bands from amplitudes.

  Arguments:
    - n_filters: int, the number of filters in the filterbank
    - fs: int, the sampling rate of the input signal
    - resampling_factor: int, the internal up / down sampling factor for the signal
    - device: str, the device to use
  """

  def __init__(self, n_filters: int = 2048, fs: int = 44100, resampling_factor: int = 32, device: str = 'cuda'):
    super().__init__()
    self._resampling_factor = resampling_factor
    self._device = device

    self._filterbank = FilterBank(
      n_filters=n_filters,
      fs=fs,
      device=self._device
    )

    # Shift of the noisebands between inferences, to maintain continuity
    self._noisebands_shift = 0

    # NOTE: the fast batch=1 path needs a [n_blocks, n_filters, rf] "blocked" view of
    # `noisebands`. That view is a pure reshape+permute (zero-copy) of the existing
    # `noisebands` buffer, so we derive it on the fly in forward() instead of storing a
    # second, .contiguous() resident copy here. Keeping that copy doubled the resident
    # noise-bank memory (mem exp1).


  @property
  def n_params(self):
    return len(self._filterbank.noisebands)


  @property
  def jit_name(self):
    return "NoiseBandSynth"


  def forward(self, amplitudes: torch.Tensor, limit_components: float = 0.0, waveshaping_factor: float = 0.0,
              limit_mode: int = 0, spectral_roll: float = 0.0, spectral_stretch: float = 0.0,
              spectral_warp: float = 0.0, harmonise_amount: float = 0.0, harmonise_notes: int = 0,
              harmonise_focus: float = 0.0, partial_merge: bool = True) -> torch.Tensor:
    # limit_mode / spectral_* / harmonise_* / partial_merge accepted for a uniform _synthesize call signature; only
    # BendableNoiseBandSynth acts on them.
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - amplitudes: torch.Tensor[batch_size, n_bands, n_ctrl], the predicted amplitudes of the noise bands
      - limit_components: float, the attenuation factor for the number of used bands
      - waveshaping_factor: float, does nothing, it is here for the JIT compatibility with other synth classes
    Returns:
      - signal: torch.Tensor[batch_size, 1, sig_length], the synthesized signal
    """
    limit_components = max(0.0, min(limit_components, 1.0))
    rf = int(self._resampling_factor)
    noisebands = self._noisebands  # int8 [n_filters, band_len] (mem exp3)
    nb_scale = self._filterbank.noisebands_scale  # [n_filters] per-band; dequant = int8_slice.float() * scale[:,None]
    band_len = noisebands.shape[-1]
    n_ctrl = amplitudes.shape[-1]
    total_len = n_ctrl * rf
    batch_size = amplitudes.shape[0]

    if self.training:
      # Roll noisebands randomly during training to avoid overfitting to specific noise values
      noisebands = torch.roll(noisebands, shifts=int(torch.randint(0, band_len, size=(1,), device=noisebands.device)), dims=-1)

    # Track shift for streaming continuity (start_pos is always a multiple of rf)
    start_pos = self._noisebands_shift % band_len
    self._noisebands_shift = (self._noisebands_shift + total_len) % band_len

    if limit_components > 0:
      upsampled_amplitudes = amplitudes.repeat_interleave(rf, dim=-1)
      max_bands = max(int(self.n_params * (1 - limit_components)), 1)
      mean_amps = upsampled_amplitudes.mean(dim=-1)
      _, indices = torch.topk(mean_amps, max_bands, dim=1)
      mask = torch.zeros_like(upsampled_amplitudes)
      mask.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, total_len), 1)
      upsampled_amplitudes = upsampled_amplitudes * mask
      signal = torch.zeros(batch_size, 1, total_len, dtype=upsampled_amplitudes.dtype, device=upsampled_amplitudes.device)
      out_pos, nb_pos = 0, start_pos
      while out_pos < total_len:
        avail = band_len - nb_pos
        take = min(avail, total_len - out_pos)
        nb_seg = noisebands[:, nb_pos:nb_pos+take].to(torch.float32) * nb_scale.unsqueeze(-1)
        signal[:, :, out_pos:out_pos+take] = (upsampled_amplitudes[:, :, out_pos:out_pos+take] * nb_seg).sum(1, keepdim=True)
        out_pos += take; nb_pos = (nb_pos + take) % band_len
      return signal

    # Fast inference path for batch_size=1 and no limit_components:
    # Periodic blocked BMM — avoids repeat_interleave and processes at control rate.
    # _nb_blocked: [n_blocks, n_filters, rf], each block = one control frame's audio.
    if not self.training and batch_size == 1:
      # Blocked view [n_blocks, n_filters, rf] derived as a zero-copy reshape+permute
      # of `noisebands` [n_filters, band_len]; no second resident copy (mem exp1).
      n_f = noisebands.shape[0]
      n_blocks = band_len // rf
      nb_blocked = noisebands.reshape(n_f, n_blocks, rf).permute(1, 0, 2)
      start_block = start_pos // rf
      # Fold the per-band scale into the amplitudes (cheap: [n_ctrl, n_filters]) instead of
      # scaling the large noise slices — out = Σ_f amp_f·(int8_f·scale_f) (mem exp3).
      amp_T = amplitudes.squeeze(0).T * nb_scale.unsqueeze(0)  # [n_ctrl, n_filters]
      out = torch.empty(n_ctrl, rf, dtype=amplitudes.dtype, device=amplitudes.device)
      t, b = 0, start_block
      while t < n_ctrl:
        avail = n_blocks - b
        take = min(avail, n_ctrl - t)
        # nb_blocked[b:b+take] is a (non-contiguous) int8 view; convert just this small
        # per-segment slice to float (never the full table), then bmm against scaled amps.
        nb_seg = nb_blocked[b:b+take].to(torch.float32)
        out[t:t+take] = torch.bmm(amp_T[t:t+take].unsqueeze(1), nb_seg).squeeze(1)
        t += take; b = (b + take) % n_blocks
      return out.reshape(1, 1, -1)

    # General path: training or batch_size > 1
    upsampled_amplitudes = amplitudes.repeat_interleave(rf, dim=-1)
    signal = torch.zeros(batch_size, 1, total_len, dtype=upsampled_amplitudes.dtype, device=upsampled_amplitudes.device)
    out_pos, nb_pos = 0, start_pos
    while out_pos < total_len:
      avail = band_len - nb_pos
      take = min(avail, total_len - out_pos)
      nb_chunk = noisebands[:, nb_pos:nb_pos+take].to(torch.float32) * nb_scale.unsqueeze(-1)
      amp_chunk = upsampled_amplitudes[:, :, out_pos:out_pos+take]
      signal[:, :, out_pos:out_pos+take] = (amp_chunk * nb_chunk).sum(dim=1, keepdim=True)
      out_pos += take; nb_pos = (nb_pos + take) % band_len
    return signal

  @property
  def _noisebands(self):
    """Delegate the noisebands to the filterbank object."""
    return self._filterbank.noisebands

@register_synth
class BendableNoiseBandSynth(BaseSynth):
  def __init__(self,
               n_filters: int = 2048,
               fs: int = 44100,
               resampling_factor: int = 32,
               device: str = 'cuda',
               streaming: bool = False):
    super().__init__()

    print(f"Initializing BendableNoiseBandSynth with resampling_factor {resampling_factor}...")

    self._streaming = streaming
    self._resampling_factor = resampling_factor

    self._noiseband_synth = NoiseBandSynth(
      n_filters=n_filters,
      fs=fs,
      resampling_factor=resampling_factor,
      device=device,
    )

    boundaries = self._noiseband_synth._filterbank._boundaries
    centers = (np.array(boundaries[:-1]) + np.array(boundaries[1:])) / 2
    # Add the lowpass and highpass frequencies
    centers = np.concatenate(([boundaries[0]/2], centers, [(boundaries[-1] + fs//2)/2]))
    frequencies = centers.tolist()

    self._sine_synth = SubbandSineSynth(
      fs=fs,
      n_sines=len(frequencies),
      resampling_factor=resampling_factor,
      streaming=self._streaming,
      frequencies=frequencies,
      device=device
    )

    # Band-amplitude vector length (== n_filters == sine bank size). Used by the control-rate
    # spectral transforms (bends + limiting) in _apply_spectral.
    self._n_bands = self._sine_synth._n_sines
    self.register_buffer('_band_idx', torch.arange(self._n_bands).float())

    # Harmonise and the sine path's partial merge work on band pitches (MIDI note numbers). Bands
    # outside 30 Hz - 12 kHz are never moved. All non-persistent: old checkpoints load unchanged.
    centres_t = torch.tensor(frequencies, dtype=torch.float32)
    self.register_buffer('_band_pitch', 69.0 + 12.0 * torch.log2(torch.clamp(centres_t, min=1.0) / 440.0), persistent=False)
    self.register_buffer('_band_snappable', (centres_t >= 30.0) & (centres_t <= 12000.0), persistent=False)
    # Harmonise remap for the current notes/focus, rebuilt only when they change: band i sends its
    # power to bands _harm_idx[i] with weights _harm_w[i]; _harm_freq is each band's sine frequency
    # with the target bands retuned to their exact note.
    self.register_buffer('_harm_idx', torch.zeros(0, 1, dtype=torch.long), persistent=False)
    self.register_buffer('_harm_w', torch.zeros(0, 1), persistent=False)
    self.register_buffer('_harm_freq', torch.zeros(0), persistent=False)
    self._harm_notes: int = -1
    self._harm_focus: float = -1.0
    # Partial merge picks peaks on a smoothed copy of the band levels (state carried across blocks).
    self.register_buffer('_merge_smooth', torch.zeros(0, 0), persistent=False)
    self._merge_alpha: float = 1.0 - math.exp(-1.0 / (0.04 * fs / resampling_factor))   # ~40 ms

  @property
  def n_params(self):
    return self._noiseband_synth.n_params

  @property
  def streaming(self):
    return self._streaming

  @property
  def jit_name(self):
    return "BendableNoiseBandSynth"

  @streaming.setter
  def streaming(self, value: bool):
    self._streaming = value
    self._sine_synth.streaming = value
    self._noiseband_synth.streaming = value

  def _gather_interp(self, x: torch.Tensor, src: torch.Tensor, wrap: bool) -> torch.Tensor:
    """
    Resample the band-amplitude vector: output band i takes the (linearly interpolated) value at
    source position src[i]. `wrap` = circular (roll) vs clamped (stretch/warp). x: [B, N, T].
    """
    N = x.shape[1]
    if wrap:
      s = src % float(N)
      i0 = torch.floor(s)
      frac = (s - i0).view(1, N, 1)
      i0l = i0.long() % N
      i1l = (i0l + 1) % N
    else:
      s = torch.clamp(src, 0.0, float(N - 1))
      i0 = torch.floor(s)
      frac = (s - i0).view(1, N, 1)
      i0l = i0.long()
      i1l = torch.clamp(i0l + 1, 0, N - 1)
    a0 = x.index_select(1, i0l)
    a1 = x.index_select(1, i1l)
    return a0 + (a1 - a0) * frac

  def _limit_mask(self, x: torch.Tensor, limit_components: float, limit_mode: int) -> torch.Tensor:
    """
    Control-rate keep-mask over bands. Amount -> keep count k; mode -> which bands survive. Returned
    mask multiplies the amplitudes and is upsampled downstream, so on/off becomes a smooth fade.
    """
    N = x.shape[1]
    k = int(N * (1.0 - limit_components) + 0.5)
    if k < 1:
      k = 1
    if k >= N:
      return torch.ones(1, 1, 1, device=x.device, dtype=x.dtype)

    if limit_mode == 1:      # density: evenly index-spaced (perceptually even across Bark)
      idx = torch.linspace(0.0, float(N - 1), k, device=x.device).round().long()
      m = torch.zeros(N, device=x.device, dtype=x.dtype).index_fill(0, idx, 1.0)
      return m.view(1, N, 1)
    elif limit_mode == 2:    # lower: spectral low-pass
      m = torch.zeros(1, N, 1, device=x.device, dtype=x.dtype)
      m[:, :k, :] = 1.0
      return m
    elif limit_mode == 3:    # higher: spectral high-pass
      m = torch.zeros(1, N, 1, device=x.device, dtype=x.dtype)
      m[:, N - k:, :] = 1.0
      return m
    elif limit_mode == 4:    # peaks: keep k most prominent local maxima
      a = x.abs()
      left = torch.zeros_like(a); right = torch.zeros_like(a)
      left[:, 1:, :] = a[:, :-1, :]
      right[:, :-1, :] = a[:, 1:, :]
      is_peak = (a >= left) & (a >= right)
      score = torch.where(is_peak, a, torch.full_like(a, -1.0))
      _, idx = torch.topk(score, k, dim=1)
      m = torch.zeros_like(a)
      m.scatter_(1, idx, 1.0)
      return m * is_peak.to(x.dtype)   # drop filler when fewer than k peaks exist
    elif limit_mode == 5:    # stochastic: random ~k bands per frame (grainy)
      keep_prob = float(k) / float(N)
      return (torch.rand(x.shape[0], N, x.shape[2], device=x.device) < keep_prob).to(x.dtype)
    else:                    # 0 loudest: top-k by magnitude
      _, idx = torch.topk(x.abs(), k, dim=1)
      m = torch.zeros_like(x)
      m.scatter_(1, idx, 1.0)
      return m

  def _apply_spectral(self, x: torch.Tensor, spectral_roll: float, spectral_stretch: float,
                      spectral_warp: float, limit_components: float, limit_mode: int) -> torch.Tensor:
    """
    Control-rate spectral transforms on the band-amplitude vector [B, N, T]: bends (reshape the
    spectrum) then limiting (thin it). All no-ops at their neutral values, so the default path is
    unchanged. Doing this at control rate lets the downstream linear upsampling smooth every change.
    """
    idx = self._band_idx
    N = x.shape[1]
    # stretch: i -> i/scale (scale>1 spreads energy toward highs); then warp: smooth energy-
    # conserving skew; then roll: circular shift with wrap. Each guarded to its neutral value.
    if spectral_stretch != 0.0:
      scale = 2.0 ** max(-2.0, min(spectral_stretch, 2.0))
      x = self._gather_interp(x, idx / scale, False)
    if spectral_warp != 0.0:
      gamma = 2.0 ** max(-2.0, min(spectral_warp, 2.0))
      x = self._gather_interp(x, float(N) * torch.pow(idx / float(N), 1.0 / gamma), False)
    if spectral_roll != 0.0:
      roll = max(0.0, min(spectral_roll, 1.0))
      x = self._gather_interp(x, idx - roll * float(N), True)
    if limit_components > 0.0:
      x = x * self._limit_mask(x, limit_components, limit_mode)
    return x

  def _build_harmonise(self, notes: int, focus: float):
    """
    Remap for a set of pitch classes (`notes`: bit k = pitch class k, C = 0) at a focus (0 soft,
    1 laser). Every snappable band sends its power to the band nearest its nearest allowed note
    (any octave); a soft focus spreads that arrival over a Gaussian of up to ±0.35 semitone.
    """
    N = self._n_bands
    dev = self._band_pitch.device
    bits = torch.tensor([(notes >> k) & 1 for k in range(12)], device=dev)
    midi = torch.arange(24, 128, device=dev)
    targets = midi[bits[midi % 12] == 1].float()                                 # [M] allowed MIDI notes
    pitch = self._band_pitch
    which = torch.argmin(torch.abs(pitch.unsqueeze(1) - targets.unsqueeze(0)), dim=1)    # [N] nearest note
    nearest_note = targets[which]
    note_band = torch.argmin(torch.abs(targets.unsqueeze(1) - pitch.unsqueeze(0)), dim=1)  # [M] band of each note
    dest = note_band[which]                                                               # [N]
    own = torch.arange(N, device=dev)
    keep = self._band_snappable
    dest = torch.where(keep, dest, own)

    width = 0.35 * (1.0 - max(0.0, min(focus, 1.0)))
    if width < 1.0e-3:
      idx = dest.unsqueeze(1)
      w = torch.ones(N, 1, device=dev)
    else:
      offsets = torch.arange(-16, 17, device=dev)
      idx = torch.clamp(dest.unsqueeze(1) + offsets.unsqueeze(0), 0, N - 1)                         # [N, 33]
      w = torch.exp(-0.5 * ((pitch[idx] - nearest_note.unsqueeze(1)) / width) ** 2)
      w[:, 16] = 1.0
      w = w / w.sum(dim=1, keepdim=True)
      stay = torch.zeros_like(w)
      stay[:, 16] = 1.0
      idx = torch.where(keep.unsqueeze(1), idx, own.unsqueeze(1).expand(N, 33))
      w = torch.where(keep.unsqueeze(1), w, stay)

    freq = self._sine_synth._base_freqs.clone().float()
    freq[note_band] = 440.0 * torch.pow(2.0, (targets - 69.0) / 12.0)
    self._harm_idx = idx
    self._harm_w = w
    self._harm_freq = freq
    self._harm_notes = notes
    self._harm_focus = focus

  def _harmonise(self, x: torch.Tensor, amount: float, notes: int, focus: float) -> torch.Tensor:
    """Moves `amount` of each band's power (bands are incoherent, so power adds) onto the held notes."""
    if notes != self._harm_notes or abs(focus - self._harm_focus) > 1.0e-4 or self._harm_idx.shape[0] != x.shape[1]:
      self._build_harmonise(notes, focus)
    P = x * x
    moved = torch.zeros_like(P)
    for k in range(self._harm_idx.shape[1]):
      moved.index_add_(1, self._harm_idx[:, k], P * self._harm_w[:, k].view(1, -1, 1))
    return torch.sqrt(torch.clamp((1.0 - amount) * P + amount * moved, min=0.0))

  def _merge_partials(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sine path only: each spectral peak becomes one sine at its interpolated frequency, and every
    other band's power joins its nearest peak (in pitch). Without this, a partial that falls between
    two band centres is played by both neighbouring sines, 12 Hz apart below 3 kHz, and beats.
    Peaks are picked on a ~40 ms smoothed copy so they don't flicker frame to frame.
    Returns (amplitudes, frequencies), both [B, N, T].
    """
    B, N, T = x.shape
    if self._merge_smooth.shape[0] != B or self._merge_smooth.shape[1] != N or not self._streaming:
      self._merge_smooth = x[:, :, 0].clone()
    cur = self._merge_smooth
    smooth = torch.empty_like(x)
    a = self._merge_alpha
    for t in range(T):
      cur = cur + a * (x[:, :, t] - cur)
      smooth[:, :, t] = cur
    if self._streaming:
      self._merge_smooth = cur.detach()

    left = torch.cat([smooth[:, :1, :] * 0.0, smooth[:, :-1, :]], dim=1)
    right = torch.cat([smooth[:, 1:, :], smooth[:, -1:, :] * 0.0], dim=1)
    floor = smooth.amax(dim=1, keepdim=True) * 1.0e-3
    is_peak = (smooth >= left) & (smooth > right) & (smooth > floor)

    idx = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, T)
    lp = torch.cummax(torch.where(is_peak, idx, torch.full_like(idx, -1)), dim=1).values
    rp = torch.flip(torch.cummin(torch.flip(torch.where(is_peak, idx, torch.full_like(idx, N)), [1]), dim=1).values, [1])
    pitch = self._band_pitch.view(1, N, 1)
    p_all = self._band_pitch
    inf = torch.full_like(x, 1.0e9)
    dl = torch.where(lp >= 0, pitch - p_all[torch.clamp(lp, 0, N - 1)], inf)
    dr = torch.where(rp < N, p_all[torch.clamp(rp, 0, N - 1)] - pitch, inf)
    target = torch.where(dl <= dr, lp, rp)
    target = torch.where((lp < 0) & (rp >= N), idx, target)
    amps = torch.sqrt(torch.zeros_like(x).scatter_add(1, target, x * x))

    # Parabolic interpolation of each peak on log levels -> fraction of a band, then Hz.
    eps = 1.0e-9
    l = torch.log(left + eps)
    c = torch.log(smooth + eps)
    r = torch.log(right + eps)
    den = l - 2.0 * c + r
    d = torch.where(den < -1.0e-9, 0.5 * (l - r) / den, torch.zeros_like(den))
    d = torch.clamp(d, -0.5, 0.5)
    base = self._sine_synth._base_freqs.float()
    up = torch.cat([base[1:] - base[:-1], (base[-1:] - base[-2:-1])]).view(1, N, 1)
    down = torch.cat([(base[1:2] - base[0:1]), base[1:] - base[:-1]]).view(1, N, 1)
    freqs = base.view(1, N, 1) + torch.where(d > 0, d * up, d * down)
    return amps, freqs

  def forward(self, parameters: torch.Tensor, limit_components: float = 0.0, waveshaping_factor: float = 0.0,
              limit_mode: int = 0, spectral_roll: float = 0.0, spectral_stretch: float = 0.0,
              spectral_warp: float = 0.0, harmonise_amount: float = 0.0, harmonise_notes: int = 0,
              harmonise_focus: float = 0.0, partial_merge: bool = True):
    """
    Waveshaping factor controls the amount of interpolation between noisebands and sinewaves when in the range [0, 0.5].
    On the other hand, in the rangee [0.5, 1], it controls the amount of waveshaping applied to the sinewaves.

    limit_mode selects the partial-limiting strategy (0 loudest, 1 density, 2 lower, 3 higher,
    4 peaks, 5 stochastic). spectral_roll/stretch/warp are spectral bend amounts (0 = neutral).

    harmonise_*: snap the bands to the pitch classes in `harmonise_notes` (12-bit mask, C = bit 0);
    amount 0 or no notes = off. partial_merge: the sine path plays each peak as one retuned sine.
    """
    # waveshaping_factor = kwargs.pop('waveshaping', 0.0)
    # Clamp waveshaping factor to [0, 1]
    waveshaping_factor = max(0.0, min(waveshaping_factor, 1.0))
    if waveshaping_factor < 0.5:
      interpolation = waveshaping_factor * 2
      waveshaping_factor = 0.0
    else:
      interpolation = 1.0
      waveshaping_factor = (waveshaping_factor - 0.5) * 2

    # Spectral bends + partial limiting, all at control rate on the band-amplitude vector, so the
    # synths' linear upsampling ramps any band on/off into a smooth fade (no clicks). Limiting is
    # therefore applied here and the sub-synths are called with limit_components=0.
    # Harmonise sits after the bends (so they can't pull the snapped bands off their notes) and before
    # the limiting (so Partials keeps the snapped bands).
    harmonise_amount = max(0.0, min(harmonise_amount, 1.0))
    harmonising = harmonise_amount > 0.0 and (harmonise_notes & 4095) != 0
    parameters = self._apply_spectral(parameters, spectral_roll, spectral_stretch, spectral_warp, 0.0, limit_mode)
    if harmonising:
      parameters = self._harmonise(parameters, harmonise_amount, harmonise_notes & 4095, harmonise_focus)
    if limit_components > 0.0:
      parameters = parameters * self._limit_mask(parameters, limit_components, limit_mode)

    # Interpolation = 0 -> only noisebands
    # Interpolation = 1 -> only sinewaves
    if interpolation == 0.0:
      noise_signal = self._noiseband_synth(parameters, limit_components=0.0)
      return noise_signal

    sine_signal = self._sines(parameters, waveshaping_factor, harmonise_amount if harmonising else 0.0, partial_merge)
    if interpolation == 1.0:
      return sine_signal
    noise_signal = self._noiseband_synth(parameters, limit_components=0.0)
    return (1-interpolation)*noise_signal + interpolation*sine_signal

  def _sines(self, x: torch.Tensor, waveshaping_factor: float, retune: float, partial_merge: bool) -> torch.Tensor:
    """The sine path: merged partials (or one sine per band), with harmonised bands pulled to their
    exact note by `retune` (the harmonise amount)."""
    base = self._sine_synth._base_freqs.float().view(1, -1, 1)
    if partial_merge:
      amps, freqs = self._merge_partials(x)
    else:
      amps, freqs = x, base.expand(x.shape[0], x.shape[1], x.shape[2])
    if retune > 0.0:
      note = self._harm_freq.view(1, -1, 1)
      is_target = note != base
      freqs = torch.where(is_target, freqs + retune * (note - freqs), freqs)
    return self._sine_synth.forward_frames(amps, freqs, waveshaping_factor=waveshaping_factor)


@register_synth
class SubbandSineSynth(BaseSynth):
  """
  Mixture of sinweaves synthesiser.

  Arguments:
    - fs: int, the sampling rate of the input signal
    - n_sines: int, the number of sinewaves to synthesise
    - resampling_factor: int, the internal up / down sampling factor for the sinewaves
    - streaming: bool, whether to run the model in streaming mode
  """
  def __init__(self,
               fs: int = 44100,
               n_sines: int = 500,
               resampling_factor: int = 32,
               streaming: bool = False,
               frequencies: List = None,
               device: str = 'cuda'):
    super().__init__()
    self._fs = fs
    self._n_sines = n_sines
    self._resampling_factor = resampling_factor
    # self._phases = None
    self.register_buffer("_phases", torch.empty(0))
    self._phases_initialized = False

    self.streaming = streaming
    self._device = device

    # self._base_freqs = torch.linspace(40, self._fs / 2, self._n_sines, device=self._device)
    # self._base_freqs =
    if frequencies is not None:
      assert len(frequencies) == n_sines, "Frequencies list must have the same length as n_sines"
      base_freqs = torch.tensor(frequencies, device=self._device)
      self.register_buffer('_base_freqs', base_freqs)
    else:
      self.register_buffer('_base_freqs', self._bark_freqs(self._fs, self._n_sines, device=self._device))

    # shift ranges are the maximum shift according to the difference between the base frequencies
    shift_ranges = (self._base_freqs[1:] - self._base_freqs[:-1]) / 2
    shift_ranges = torch.cat((shift_ranges, shift_ranges[-1].view(1)))
    self.register_buffer('_shift_ranges', shift_ranges)

    # --- Fixed-frequency fast-path caches (used by forward_bank) ---------------------------------
    # When there is no pitch-bend the per-band frequency is constant, so ω = 2πf/fs, the nyquist
    # gate and the phase ramp are all time-invariant and can be precomputed / cached.
    self.register_buffer('_omega', (self._base_freqs * 2 * math.pi / self._fs).view(1, -1, 1))
    self.register_buffer('_nyquist_mask', (self._base_freqs < self._fs / 2).float().view(1, -1, 1) + 1e-4)
    self.register_buffer('_phase_ramp', torch.empty(0))  # ω·t, lazily built per block length
    # Fixed, seeded random start phases. Starting every sine at 0 makes the equally spaced bank (12 Hz
    # apart below 3 kHz) line up every 83 ms into a 12 Hz pulse; random phases never line up.
    _g = torch.Generator().manual_seed(1234)
    self.register_buffer('_phase_init', torch.rand(self._n_sines, generator=_g) * 2 * math.pi, persistent=False)
    self._ramp_len: int = -1
    self._ws_factor: float = -1.0   # cached waveshaper knob + its analytic RMS scalar
    self._ws_scale: float = 1.0

  @property
  def n_params(self):
    return 2*self._n_sines

  @property
  def jit_name(self):
    return "SubbandSineSynth"


  def forward(self, parameters: torch.Tensor, limit_components: float = 0.0, waveshaping_factor: float = 0.0,
              limit_mode: int = 0, spectral_roll: float = 0.0, spectral_stretch: float = 0.0,
              spectral_warp: float = 0.0, harmonise_amount: float = 0.0, harmonise_notes: int = 0,
              harmonise_focus: float = 0.0, partial_merge: bool = True):
    # limit_mode / spectral_* / harmonise_* / partial_merge accepted for a uniform _synthesize call signature; unused here.
    """
    Generates a mixture of sinewaves with the given frequencies and amplitudes per sample.

    Arguments:
      - parameters: torch.Tensor[batch_size, n_params, n_samples], the parameters of the synthesizer
      - limit_components: float, the attenuation factor for the number of sinewaves
      - waveshaping_factor: float, the amount of waveshaping to apply to the sinewaves (0.0 = no waveshaping, 1.0 = full waveshaping)
    """
    # limit_components = kwargs.get('limit_components', 0.0)
    # waveshaping_factor = kwargs.get('waveshaping_factor', 0.0)

    shift_ratios = parameters[:, :self._n_sines, :] # n_sines shift ratios
    amplitudes = parameters[:, self._n_sines:, :] # n_sines amplitudes

    batch_size = shift_ratios.shape[0]

    shift_ratios = (shift_ratios - 1)*2 # shift from [0, 2] to [-2, 2] range

    # We only need to initialise phases buffer if we are in streaming mode
    # if self._streaming and (self._phases is None or self._phases.shape[0] != batch_size):
    #   # self._phases = torch.zeros(batch_size, self._n_sines, device=self._device)
    #   self._phases = torch.zeros(batch_size, self._n_sines)

    # We only need to initialise phases buffer if we are in streaming mode
    if self.streaming and (not self._phases_initialized or self._phases.shape[0] != batch_size):
      self._phases = torch.zeros(batch_size, self._n_sines)
      self._phases_initialized = True

    # Upsample from the internal sampling rate to the target sampling rate
    shift_ratios = F.interpolate(shift_ratios, scale_factor=float(self._resampling_factor), mode='linear')
    amplitudes = F.interpolate(amplitudes, scale_factor=float(self._resampling_factor), mode='linear')

    # Calculate the shifts from the ratios
    shifts = shift_ratios * self._shift_ranges.view(1, -1, 1)
    # Calculate the frequencies from the shifts
    frequencies = self._base_freqs.view(1, -1, 1) + shifts

    # cancel the sines above nyquist frequency
    amplitudes *= (frequencies < self._fs / 2).float() + 1e-4

    # Normalize the amplitudes
    # amplitudes /= amplitudes.sum(-1, keepdim=True)
    # amplitudes /= amplitudes.sum(1, keepdim=True)

    # Multiply the amplitudes by the general loudness
    # amplitudes *= general_amplitude

    # Calculate the phase increments
    omegas = frequencies * 2 * math.pi / self._fs

    # Calculate the phases at points, in place
    phases = omegas.cumsum(dim=-1)
    phases = phases % (2 * math.pi)

    if self.streaming:
      # Shift the phases by the last phase from last generation
      # breakpoint()
      phases = (phases.permute(2, 0, 1) + self._phases).permute(1, 2, 0)

      # Copy the last phases for next iteration
      self._phases.copy_(phases[: ,: , -1] % (2 * math.pi))


    # If the limit_components is higher than 0, we need to limit the number of sinewaves
    # by choosing the first max_sines, in relation to their amplitudes
    # clamp limit_components to [0, 1]
    limit_components = max(0.0, min(limit_components, 1.0))
    if limit_components > 0:
      # Calculate the number of sinewaves to keep (at least 1)
      max_sines = int(self._n_sines * (1 - limit_components))
      max_sines = max(max_sines, 1)

      # Get the indices of the max_sines largest amplitudes
      _, indices = torch.topk(amplitudes, max_sines, dim=1)
      # Create a mask of the same shape as the amplitudes
      mask = torch.zeros_like(amplitudes)
      # Set the values at the indices to 1
      mask.scatter_(1, indices, 1)

      # Apply the mask to the frequencies and amplitudes
      frequencies = frequencies * mask
      amplitudes = amplitudes * mask
      phases = phases * mask

    # Generate and sum the sinewaves
    components = torch.sin(phases)
    # clamp wavewshaping factor to [0, 1]
    waveshaping_factor = max(0.0, min(waveshaping_factor, 1.0))
    if waveshaping_factor > 0.0:
      components = self._waveshape_tanh(components, waveshaping_factor)

    signal = torch.sum(amplitudes * components, dim=1, keepdim=True)
    return signal


  def _ws_analytic_scale(self, factor: float, drive_max: float = 20.0) -> float:
    """
    RMS-compensation scalar for the tanh waveshaper. The shaper acts on unit sines, so the
    compensation is identical across all components — it is the ratio of the unit-sine RMS to
    the shaped-waveform RMS over one period. Computed once from a dense reference period and
    cached (recomputed only when the knob moves), replacing the old per-band reduction over the
    full [B, N, T] tensor (whose per-band variation was finite-window/near-nyquist noise).
    """
    if factor != self._ws_factor:
      m = max(0.0, min(factor, 1.0))
      g = (m / (1.0 - m + 1e-6)) * drive_max
      s = torch.sin(torch.linspace(0.0, 2.0 * math.pi, 8192))
      w = (1.0 - m) * s + m * torch.tanh(g * s)
      self._ws_scale = float(torch.sqrt(torch.mean(s * s)) / torch.sqrt(torch.mean(w * w) + 1e-12))
      self._ws_factor = factor
    return self._ws_scale

  def _waveshape_tanh(self, x: torch.Tensor, factor: float, drive_max: float = 20.0) -> torch.Tensor:
    """
    Per-component tanh waveshaper with (scalar) RMS compensation and dry/wet morph.

    Args:
      - x: torch.Tensor [B, N, T]; expected to be unit sines (torch.sin(phases))
      - factor: float in [0,1], 0 = bypass, 1 = strong saturation
      - drive_max: maximum drive used when factor=1
    """
    if factor <= 0.0:
      return x
    m = max(0.0, min(factor, 1.0))
    g = (m / (1.0 - m + 1e-6)) * drive_max
    return torch.lerp(x, torch.tanh(g * x), m) * self._ws_analytic_scale(m, drive_max)

  def _get_ramp(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Cached fixed-frequency phase ramp (ω·t mod 2π), t = 1..L; rebuilt only when the block length L
    changes. Pre-wrapped so forward_bank can add the per-band offset and feed torch.sin directly
    without a full-size modulo (args stay in [0, 4π), where sin is exact).
    """
    if self._ramp_len != length or self._phase_ramp.numel() == 0:
      t = torch.arange(length, device=device, dtype=dtype).view(1, 1, length) + 1.0
      self._phase_ramp = (self._omega.to(dtype) * t) % (2.0 * math.pi)
      self._ramp_len = length
    return self._phase_ramp

  def forward_bank(self, amplitudes: torch.Tensor, limit_components: float = 0.0, waveshaping_factor: float = 0.0) -> torch.Tensor:
    """
    Fixed-frequency oscillator bank. Numerically equivalent to forward() when there is no
    pitch-bend (shift ratios == 1), but much cheaper: frequencies == base_freqs are constant, so
    the phase ramp and nyquist gate are precomputed and the tanh RMS compensation is a scalar.

    Args:
      - amplitudes: torch.Tensor[B, n_sines, T_ctrl], per-band amplitudes at control rate.
      - limit_components / waveshaping_factor: as in forward().
    """
    batch_size = amplitudes.shape[0]
    if self.streaming and (not self._phases_initialized or self._phases.shape[0] != batch_size):
      self._phases = self._phase_init.unsqueeze(0).repeat(batch_size, 1)
      self._phases_initialized = True

    ws = max(0.0, min(waveshaping_factor, 1.0))

    # Fold the (time-invariant) nyquist gate and the waveshaper's scalar into the control-rate
    # amplitudes. Both commute with linear upsampling and per-band constant scaling, so this is
    # numerically equivalent yet removes two full-size (audio-rate) passes.
    amplitudes = amplitudes * self._nyquist_mask
    if ws > 0.0:
      amplitudes = amplitudes * self._ws_analytic_scale(ws)

    amplitudes = F.interpolate(amplitudes, scale_factor=float(self._resampling_factor), mode='linear')
    length = amplitudes.shape[-1]

    # ramp is pre-wrapped to [0, 2π) by _get_ramp; adding the per-band offset (also in [0, 2π))
    # keeps phases in [0, 4π), where torch.sin is exact — so we skip the full-size modulo (a
    # ~55ms/block pass) and only re-wrap the single carried-over column [B, N] for the next block.
    ramp = self._get_ramp(length, amplitudes.device, amplitudes.dtype)
    if self.streaming:
      phases = ramp + self._phases.unsqueeze(-1)
      self._phases.copy_(phases[:, :, -1] % (2.0 * math.pi))
    else:
      phases = ramp + self._phase_init.view(1, -1, 1)

    # limit_components: keep forward()'s exact per-sample top-k band selection (audio rate).
    limit_components = max(0.0, min(limit_components, 1.0))
    if limit_components > 0.0:
      max_sines = max(int(self._n_sines * (1.0 - limit_components)), 1)
      _, indices = torch.topk(amplitudes, max_sines, dim=1)
      mask = torch.zeros_like(amplitudes)
      mask.scatter_(1, indices, 1.0)
      amplitudes = amplitudes * mask

    if ws > 0.0:
      x = torch.sin(phases)
      g = (ws / (1.0 - ws + 1e-6)) * 20.0
      components = torch.lerp(x, torch.tanh(g * x), ws)
    else:
      components = torch.sin(phases)

    return torch.sum(amplitudes * components, dim=1, keepdim=True)

  def forward_frames(self, amplitudes: torch.Tensor, frequencies: torch.Tensor, waveshaping_factor: float = 0.0) -> torch.Tensor:
    """
    Oscillator bank whose frequencies may change every control frame (held for the frame, phase
    continuous across frames and blocks). Used by the merged-partials sine path.

    Args:
      - amplitudes, frequencies: torch.Tensor[B, n_sines, T_ctrl], per-band amplitude and Hz.
      - waveshaping_factor: as in forward_bank().
    """
    batch_size = amplitudes.shape[0]
    n = amplitudes.shape[1]
    n_ctrl = amplitudes.shape[2]
    rf = int(self._resampling_factor)
    if self.streaming and (not self._phases_initialized or self._phases.shape[0] != batch_size):
      self._phases = self._phase_init.unsqueeze(0).repeat(batch_size, 1)
      self._phases_initialized = True
    start = self._phases if self.streaming else self._phase_init.unsqueeze(0).repeat(batch_size, 1)

    ws = max(0.0, min(waveshaping_factor, 1.0))
    amplitudes = amplitudes * (frequencies < self._fs / 2).to(amplitudes.dtype)
    if ws > 0.0:
      amplitudes = amplitudes * self._ws_analytic_scale(ws)

    two_pi = 2.0 * math.pi
    omega = frequencies * (two_pi / self._fs)                       # [B, N, T]
    inc = (omega * rf) % two_pi                                     # phase advance per frame
    frame_start = (start.unsqueeze(-1) + torch.cumsum(inc, dim=-1) - inc) % two_pi
    tau = torch.arange(1, rf + 1, device=amplitudes.device, dtype=amplitudes.dtype)
    phases = (frame_start.unsqueeze(-1) + omega.unsqueeze(-1) * tau).reshape(batch_size, n, n_ctrl * rf)
    if self.streaming:
      self._phases.copy_((frame_start[:, :, -1] + inc[:, :, -1]) % two_pi)

    amplitudes = F.interpolate(amplitudes, scale_factor=float(rf), mode='linear')
    x = torch.sin(phases)
    if ws > 0.0:
      g = (ws / (1.0 - ws + 1e-6)) * 20.0
      x = torch.lerp(x, torch.tanh(g * x), ws)
    return torch.sum(amplitudes * x, dim=1, keepdim=True)

  @torch.jit.ignore
  @staticmethod
  def _bark_freqs(fs, n_sines, device='cpu'):
    # Use Bark scale to divide the range of frequencies
    freqs = torch.linspace(40, fs / 2, n_sines, device=device)
    bark = 6 * torch.arcsinh(freqs/600.)
    scaled = 30 + bark / max(bark) * freqs
    return scaled


  def _test(self, batch_size: int = 1, n_changes: int = 5, duration: float = 0.5, audiofile: str = 'sinewaves.wav'):
    # Generate a test signal of randomised sine frequencies and amplitudes
    freqs = torch.rand(batch_size, self._n_sines, n_changes, device=self._device) * 5000 + 40
    amps = torch.rand(batch_size, self._n_sines, n_changes, device=self._device) / self._n_sines

    freqs = F.interpolate(freqs, scale_factor=self._fs*duration/n_changes/self._resampling_factor, mode='nearest')
    amps = F.interpolate(amps, scale_factor=self._fs*duration/n_changes/self._resampling_factor, mode='nearest')

    freq_chunks = freqs.chunk(100, dim=-1)
    amp_chunks = amps.chunk(100, dim=-1)

    signal = torch.Tensor()
    for freq, amp in zip(freq_chunks, amp_chunks):
      signal = torch.cat((signal, self.generate(freq, amp)), dim=-1)

    batch_size = signal.shape[0]
    for i in range(batch_size):
      torchaudio.save(f"{i}-{audiofile}", signal[i], self._fs)


# @register_synth
# class ComplexSineSynth(BaseSynth):
#   """
#   Synthesizer that generates a mixture of complex sinusoids using the complex_oscillator.

#   Arguments:
#     - fs: int, the sampling rate of the output signal
#     - n_sines: int, the number of complex sinusoids to synthesize
#     - resampling_factor: int, up/down sampling factor for control signals
#     - streaming: bool, whether to run in streaming mode (maintain phase continuity)
#     - device: str, computation device
#   """
#   def __init__(
#     self,
#     fs: int = 44100,
#     n_sines: int = 500,
#     resampling_factor: int = 32,
#     streaming: bool = False,
#     device: str = 'cuda'
#   ):
#     super().__init__(fs=fs, resampling_factor=resampling_factor)
#     self._fs = fs
#     self._n_sines = n_sines
#     self._resampling_factor = resampling_factor
#     self.streaming = streaming
#     self._device = device
#     self.register_buffer("_phases", torch.empty(0))
#     self._phases_initialized = False

#   @property
#   def n_params(self):
#     # Each sine is parameterized by a complex number z
#     return 2 * self._n_sines  # real and imaginary parts per sine

#   @property
#   def jit_name(self):
#     return "ComplexSineSynth"

#   def forward(self, parameters: torch.Tensor, initial_phase: torch.Tensor = None) -> torch.Tensor:
#     """
#     parameters: [batch, 2*n_sines, n_samples] (real/imag pairs)
#     """
#     batch_size, param_dim, n_samples = parameters.shape
#     assert param_dim == 2 * self._n_sines, "Expected 2*n_sines parameters (real/imag pairs)"

#     # Split into real and imaginary parts
#     real = parameters[:, :self._n_sines, :]
#     imag = parameters[:, self._n_sines:, :]
#     z = torch.complex(real, imag)  # [batch, n_sines, n_samples]

#     # Streaming phase management (as before)
#     if self.streaming:
#       if (not self._phases_initialized) or (self._phases.shape[0] != batch_size or self._phases.shape[1] != self._n_sines):
#         self._phases = torch.ones(batch_size, self._n_sines, device=z.device, dtype=z.dtype)
#         self._phases_initialized = True
#       initial_phase = self._phases

#     signal = complex_oscillator(
#       z,
#       initial_phase=initial_phase,
#       N=n_samples,
#       constrain=True,
#       reduce=True
#     )  # [batch, n_sines, n_samples]

#     if self.streaming:
#       self._phases.copy_(signal[..., -1])

#     signal = signal.sum(dim=1, keepdim=True)  # [batch, 1, n_samples]
#     return signal
