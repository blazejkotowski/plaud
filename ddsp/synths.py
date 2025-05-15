from torch import nn
import torch
import math
import torchaudio
import torch.nn.functional as F
import numpy as np

from ddsp.filterbank import FilterBank

from typing import Type, Callable, Dict, Any

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

    # Lookup in global scope (assuming classes are imported)
    cls = globals()[cls_name]
    return cls(**params)



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


  def forward(self, amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - amplitudes: torch.Tensor[batch_size, n_bands, sig_length], the predicted amplitudes of the noise bands
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
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
    signal = torch.sum(upsampled_amplitudes * looped_bands, dim=1, keepdim=True)
    return signal

  @property
  def _noisebands(self):
    """Delegate the noisebands to the filterbank object."""
    return self._filterbank.noisebands



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
    self._streaming = streaming
    self._device = device

  @property
  def n_params(self):
    return self._n_harmonics + 1  # 1 pitch + n_harmonics amplitudes

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

    self._streaming = streaming
    self._device = device

    # self._base_freqs = torch.linspace(40, self._fs / 2, self._n_sines, device=self._device)
    # self._base_freqs =
    self.register_buffer('_base_freqs', self._bark_freqs(self._fs, self._n_sines, device=self._device))
    # shift ranges are the maximum shift according to the difference between the base frequencies
    shift_ranges = (self._base_freqs[1:] - self._base_freqs[:-1]) / 2
    shift_ranges = torch.cat((shift_ranges, shift_ranges[-1].view(1)))
    self.register_buffer('_shift_ranges', shift_ranges)

  @property
  def n_params(self):
    return 2*self._n_sines


  def forward(self, parameters: torch.Tensor):
    """
    Generates a mixture of sinewaves with the given frequencies and amplitudes per sample.

    Arguments:
      - parameters: torch.Tensor[batch_size, n_params, n_samples], the parameters of the synthesizer
    """
    shift_ratios = parameters[:, :self._n_sines, :] # n_sines shift ratios
    amplitudes = parameters[:, self._n_sines:, :] # n_sines amplitudes

    batch_size = shift_ratios.shape[0]

    shift_ratios = (shift_ratios - 1)*2 # shift from [0, 2] to [-2, 2] range

    # We only need to initialise phases buffer if we are in streaming mode
    # if self._streaming and (self._phases is None or self._phases.shape[0] != batch_size):
    #   # self._phases = torch.zeros(batch_size, self._n_sines, device=self._device)
    #   self._phases = torch.zeros(batch_size, self._n_sines)

    # We only need to initialise phases buffer if we are in streaming mode
    if self._streaming and (not self._phases_initialized or self._phases.shape[0] != batch_size):
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

    if self._streaming:
      # Shift the phases by the last phase from last generation
      # breakpoint()
      phases = (phases.permute(2, 0, 1) + self._phases).permute(1, 2, 0)

      # Copy the last phases for next iteration
      self._phases.copy_(phases[: ,: , -1] % (2 * math.pi))

    # Generate and sum the sinewaves
    signal = torch.sum(amplitudes * torch.sin(phases), dim=1, keepdim=True)
    return signal


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
