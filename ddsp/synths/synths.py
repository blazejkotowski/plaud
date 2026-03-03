from torch import nn
import torch
import math
import torchaudio
import torch.nn.functional as F
import numpy as np

from ddsp.filterbank import FilterBank
from ddsp.synths.dsp import scale_function, remove_above_nyquist, upsample, amp_to_impulse_response, fft_convolve
# from ddsp.sgd.sinusoidal_gradient_descent.core import complex_oscillator


from typing import Type, Callable, Dict, Any, List, Optional

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


  @property
  def n_params(self):
    return len(self._filterbank.noisebands)


  @property
  def jit_name(self):
    return "NoiseBandSynth"


  def forward(self, amplitudes: torch.Tensor, pitch: Optional[torch.Tensor] = None, loudness: Optional[torch.Tensor] = None, limit_components: float = 0.0, waveshaping_factor: float = 0.0) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - amplitudes: torch.Tensor[batch_size, n_bands, sig_length], the predicted amplitudes of the noise bands
      - limit_components: float, the attenuation factor for the number of used bands
      - waveshaping_factor: float, does nothing, it is here for the JIT compatibility with other synth classes
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
    # limit_components = kwargs.get('limit_components', 0.0)
    # Clamp the limit components
    limit_components = max(0.0, min(limit_components, 1.0))

    # upsample the amplitudes
    upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=float(self._resampling_factor), mode='linear')

    # shift the noisebands to maintain the continuity of the noise signal
    noisebands = torch.roll(self._noisebands, shifts=-self._noisebands_shift, dims=-1)

    if self.training:
      # roll the noisebands randomly to avoid overfitting to the noise values
      # check whether model is training
      noisebands = torch.roll(noisebands, shifts=int(torch.randint(0, noisebands.shape[-1], size=(1,), device=self._device)), dims=-1)

    # fit the noisebands into the mplitudes
    repeats = math.ceil(upsampled_amplitudes.shape[-1] / noisebands.shape[-1])
    looped_bands = noisebands.repeat(1, repeats) # repeat
    looped_bands = looped_bands[:, :upsampled_amplitudes.shape[-1]] # trim

    # Save the noisebands shift for the next iteration
    self._noisebands_shift = (self._noisebands_shift + upsampled_amplitudes.shape[-1]) % self._noisebands.shape[-1]

    # Synthesize the signal
    # If the limit_components is higher than 0, we need to limit the number of bands
    # by choosing the first max_bands, in relation to their amplitudes
    limit_components = max(0.0, min(limit_components, 1.0))
    if limit_components > 0:
      # Calculate the number of bands to keep (at least 1)
      max_bands = int(self.n_params * (1 - limit_components))
      max_bands = max(max_bands, 1)

      # Get the indices of the max_bands largest amplitudes (mean over time)
      mean_amps = upsampled_amplitudes.mean(dim=-1)
      _, indices = torch.topk(mean_amps, max_bands, dim=1)
      # Create a mask of the same shape as the amplitudes
      mask = torch.zeros_like(upsampled_amplitudes)
      mask.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, upsampled_amplitudes.shape[-1]), 1)

      # Apply the mask to the amplitudes and looped_bands
      upsampled_amplitudes = upsampled_amplitudes * mask
      looped_bands = looped_bands * mask

    signal = torch.sum(upsampled_amplitudes * looped_bands, dim=1, keepdim=True)

    return signal

  @property
  def _noisebands(self):
    """Delegate the noisebands to the filterbank object."""
    return self._filterbank.noisebands


@register_synth
class HarmonicSynth(BaseSynth):
  """
  Additive harmonic synthesizer following the DDSP reference implementation.

  Pitch (f0 in Hz) and loudness come from feature extractors — they are NOT
  predicted by the decoder.  The decoder predicts per-harmonic amplitude
  envelopes which are passed through a `scale_function` (modified sigmoid),
  Nyquist-filtered, self-normalized (divided by their sum), and then scaled
  by the external loudness.

  Parameters from decoder:
    amplitudes: [B, n_harmonics, T_ctrl]

  Features from extractors:
    pitch:    [B, 1, T_ctrl]  — fundamental frequency in Hz
    loudness: [B, 1, T_ctrl]  — overall amplitude envelope

  The synth cannot work in latent-only mode.
  """

  def __init__(self,
              n_harmonics: int = 100,
              fs: int = 44100,
              resampling_factor: int = 32,
              streaming: bool = False,
              pitch_feature: str = "pitch",
              loudness_feature: str = "loudness",
              device: str = "cuda"):
    super().__init__(fs=fs, resampling_factor=resampling_factor)
    self._n_harmonics = int(n_harmonics)
    self.streaming = bool(streaming)
    self._device = device
    self.pitch_feature = pitch_feature
    self.loudness_feature = loudness_feature

    self.register_buffer("_phase", torch.empty(0))
    self._phase_initialized = False

  @property
  def n_params(self):
    return self._n_harmonics

  @property
  def jit_name(self):
    return "HarmonicSynth"

  def forward(
      self,
      amplitudes: torch.Tensor,
      pitch: Optional[torch.Tensor] = None,
      loudness: Optional[torch.Tensor] = None,
      limit_components: float = 0.0,
      waveshaping_factor: float = 0.0,
  ) -> torch.Tensor:
    """
    Args:
      amplitudes: [B, n_harmonics, T_ctrl] — raw amplitude params from decoder
      pitch:      [B, 1, T_ctrl]           — f0 in Hz (from feature extractor)
      loudness:   [B, 1, T_ctrl]           — amplitude envelope (from feature extractor)

    Returns:
      signal: [B, 1, T_audio]
    """

    assert amplitudes.dim() == 3, "amplitudes must be [B, n_harmonics, T]"
    assert pitch is not None, "HarmonicSynth requires pitch"
    assert loudness is not None, "HarmonicSynth requires loudness"
    assert pitch.dim() == 3 and pitch.shape[1] == 1, "pitch must be [B, 1, T]"
    assert loudness.dim() == 3 and loudness.shape[1] == 1, "loudness must be [B, 1, T]"
    B, H, T_ctrl = amplitudes.shape
    assert H == self._n_harmonics, f"Expected {self._n_harmonics} harmonics, got {H}"

    # Transpose to [B, T, H] / [B, T, 1] for the reference-style pipeline
    amplitudes = amplitudes.permute(0, 2, 1)  # [B, T, H]
    pitch = pitch.permute(0, 2, 1)            # [B, T, 1]
    loudness = loudness.permute(0, 2, 1)      # [B, T, 1]

    # Scale function: modified sigmoid -> positive amplitudes
    amplitudes = scale_function(amplitudes)  # [B, T, H]

    # Zero out harmonics above Nyquist
    amplitudes = remove_above_nyquist(amplitudes, pitch, self._fs)

    # Normalize distribution and scale by loudness
    amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-8)
    amplitudes = amplitudes * loudness  # [B, T, H]

    # Upsample to audio rate
    amplitudes = upsample(amplitudes, self._resampling_factor)  # [B, T_audio, H]
    pitch = upsample(pitch, self._resampling_factor)            # [B, T_audio, 1]

    # Phase integration
    omega = torch.cumsum(2.0 * math.pi * pitch / float(self._fs), dim=1)  # [B, T_audio, 1]

    # Streaming continuity
    if self.streaming:
      if (not self._phase_initialized) or (self._phase.shape[0] != B):
        self._phase = torch.zeros(B, 1, device=omega.device, dtype=omega.dtype)
        self._phase_initialized = True
      omega = omega + self._phase.unsqueeze(1)  # [B, T_audio, 1]
      self._phase.copy_(omega[:, -1, :].detach())

    # Harmonic overtones
    n_harm = amplitudes.shape[-1]
    omegas = omega * torch.arange(1, n_harm + 1, device=omega.device, dtype=omega.dtype)  # [B, T_audio, H]

    # Synthesize
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)  # [B, T_audio, 1]

    # Back to [B, 1, T_audio] (channel-first)
    signal = signal.permute(0, 2, 1)
    return signal


@register_synth
class FilteredNoiseSynth(BaseSynth):
  """
  Filtered noise synthesizer following the DDSP reference.

  The decoder predicts n_bands filter magnitudes at control rate.
  These are converted to impulse responses via inverse FFT and convolved
  with uniform white noise to produce the noise component.

  Parameters from decoder:
    noise_params: [B, n_bands, T_ctrl] — filter magnitudes (pre scale_function)
  """

  def __init__(self,
               n_bands: int = 65,
               fs: int = 44100,
               resampling_factor: int = 32,
               device: str = "cuda"):
    super().__init__(fs=fs, resampling_factor=resampling_factor)
    self._n_bands = int(n_bands)
    self._device = device

  @property
  def n_params(self):
    return self._n_bands

  @property
  def jit_name(self):
    return "FilteredNoiseSynth"

  def forward(self, noise_params: torch.Tensor,
              pitch: Optional[torch.Tensor] = None,
              loudness: Optional[torch.Tensor] = None,
              limit_components: float = 0.0,
              waveshaping_factor: float = 0.0) -> torch.Tensor:
    """
    Args:
      noise_params: [B, n_bands, T_ctrl]

    Returns:
      signal: [B, 1, T_audio]
    """

    B, C, T_ctrl = noise_params.shape
    assert C == self._n_bands

    # Transpose to [B, T, C]
    noise_params = noise_params.permute(0, 2, 1)

    # Apply scale function with a bias shift (start quiet)
    param = scale_function(noise_params - 5.0)  # [B, T, n_bands]

    # filter_size from irfft = 2 * (n_bands - 1); target must be >= filter_size
    filter_size = 2 * (self._n_bands - 1)
    block_size = max(self._resampling_factor, filter_size)

    # Convert to impulse response
    impulse = amp_to_impulse_response(param, block_size)  # [B, T, block_size]

    # Generate white noise
    noise = torch.rand(
        impulse.shape[0],
        impulse.shape[1],
        block_size,
        device=impulse.device,
        dtype=impulse.dtype,
    ) * 2.0 - 1.0

    # Convolve
    signal = fft_convolve(noise, impulse).contiguous()  # [B, T, block_size]

    # Reshape to [B, T*block_size] then trim to expected audio length
    signal = signal.reshape(B, -1)  # [B, T_ctrl * block_size]
    expected_len = T_ctrl * self._resampling_factor
    signal = signal[:, :expected_len]

    # To [B, 1, T_audio]
    signal = signal.unsqueeze(1)
    return signal


class Reverb(nn.Module):
  """
  Trainable reverb via a learnable impulse response, following DDSP reference.

  Applied as a post-processing step after all synths are summed.
  The dry/wet mix is learnable.
  """

  def __init__(self, length: int = 44100, fs: int = 44100):
    super().__init__()
    self._length = length
    self._fs = fs

    # Learnable impulse response (initialized to decaying noise)
    ir_init = torch.randn(1, 1, length) * torch.linspace(1, 0, length).unsqueeze(0).unsqueeze(0)
    self.ir = nn.Parameter(ir_init * 0.001)

    # Learnable dry/wet mix (initialized to mostly dry)
    self.dry_wet = nn.Parameter(torch.tensor([3.0]))  # sigmoid(3) ≈ 0.95 dry

  def forward(self, signal: torch.Tensor) -> torch.Tensor:
    """
    Args:
      signal: [B, 1, T_audio]

    Returns:
      signal: [B, 1, T_audio]
    """
    # Normalize IR
    ir = self.ir
    # Pad signal for linear convolution
    T = signal.shape[-1]
    n_fft = T + self._length - 1
    # Round up to next power-of-2 (TorchScript-compatible)
    power = 1
    while power < n_fft:
      power *= 2
    n_fft = power

    signal_fr = torch.fft.rfft(signal, n=n_fft)
    ir_fr = torch.fft.rfft(ir, n=n_fft)

    wet = torch.fft.irfft(signal_fr * ir_fr, n=n_fft)
    wet = wet[..., :T]  # trim to original length

    # Dry/wet mix
    mix = torch.sigmoid(self.dry_wet)  # 1 = fully dry, 0 = fully wet
    return mix * signal + (1.0 - mix) * wet

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


  def forward(self, parameters: torch.Tensor, pitch: Optional[torch.Tensor] = None, loudness: Optional[torch.Tensor] = None, limit_components: float = 0.0, waveshaping_factor: float = 0.0):
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

  def forward(self, parameters: torch.Tensor, pitch: Optional[torch.Tensor] = None, loudness: Optional[torch.Tensor] = None, limit_components: float = 0.0, waveshaping_factor: float = 0.0):
    """
    Waveshaping factor controls the amount of interpolation between noisebands and sinewaves when in the range [0, 0.5].
    On the other hand, in the rangee [0.5, 1], it controls the amount of waveshaping applied to the sinewaves.
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

    # Prepend the shift ratios to the existing parameters
    sine_parameters = torch.cat((torch.ones_like(parameters), parameters), dim=1)

    # Interpolation = 0 -> only noisebands
    # Interpolation = 1 -> only sinewaves
    if interpolation == 0.0:
      noise_signal = self._noiseband_synth(parameters, limit_components=limit_components)
      return noise_signal

    elif interpolation == 1.0:
      sine_signal = self._sine_synth(sine_parameters, limit_components=limit_components, waveshaping_factor=waveshaping_factor)
      return sine_signal

    else:
      noise_signal = self._noiseband_synth(parameters, limit_components=limit_components)
      sine_signal = self._sine_synth(sine_parameters, limit_components=limit_components, waveshaping_factor=waveshaping_factor)
      return (1-interpolation)*noise_signal + interpolation*sine_signal


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

  @property
  def n_params(self):
    return 2*self._n_sines

  @property
  def jit_name(self):
    return "SubbandSineSynth"


  def forward(self, parameters: torch.Tensor, pitch: Optional[torch.Tensor] = None, loudness: Optional[torch.Tensor] = None, limit_components: float = 0.0, waveshaping_factor: float = 0.0):
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


  def _waveshape_tanh(self, x: torch.Tensor, factor: float, drive_max: float = 20.0) -> torch.Tensor:
    """
    Per-component tanh waveshaper with RMS compensation and dry/wet morph.

    Args:
      - x: torch.Tensor [B, N, T] or broadcastable; expected in ~[-1,1] (e.g., torch.sin(phases))
      - factor: float in [0,1], 0 = bypass, 1 = strong saturation
      - drive_max: maximum drive used when factor=1

    Returns:
      - y: torch.Tensor, same shape as x
    """
    if factor <= 0.0:
      return x

    m = torch.clamp(torch.as_tensor(float(factor), device=x.device, dtype=x.dtype), 0.0, 1.0)
    # Map morph -> drive; steep near 1 for squarer corners
    g = (m / (1.0 - m + 1e-6)) * drive_max

    # Dry/wet blend around tanh saturator
    y = (1.0 - m) * x + m * torch.tanh(g * x)

    # --- RMS compensation (per component over time) ---
    eps = 1e-12
    # Base RMS of the input unit sine (computed from x to also handle masked/zero components)
    x_rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    y_rms = torch.sqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)

    # Avoid 0/0 for fully muted components
    safe_scale = torch.where(y_rms > 1e-12, x_rms / y_rms, torch.ones_like(y_rms))
    y = y * safe_scale
    # --------------------------------------------------

    return y

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
