import torch
import numpy as np

from smstools.models.sineModel import sineModelAnal

from .base_extractor import BaseExtractor

class SinusoidsExtractor(BaseExtractor):
  def __init__(self,
               n_sines: int,
               fs: int = 44100,
               win_type: str = 'hann',
               win_size: int = 512,
               threshold: int = -60,
               min_sine_dur: float = 0.001,
               freq_dev_slope: float = 0.001,
               device: str = 'cuda'):
    """
    Args:
      n_sines: int, number of sinusoids to extract
      fs: int, sampling frequency
      win_type: str, type of window to use for FFT
      win_size: int, size of the window for FFT
      threshold: int, threshold for amplitude in dB
      minSineDuration: float, minimum duration of a sine in seconds
      freqDevSlope: float, frequency deviation slope
      device: str, device to use for computations (e.g., 'cuda' or 'cpu')
    """
    super(SinusoidsExtractor, self).__init__(resampling_factor=1, resample=False)
    self.n_sines = n_sines
    self.fs = fs
    self.win_type = win_type
    self.win_size = win_size
    self.threshold = threshold
    self.min_sine_dur = min_sine_dur
    self.freq_dev_slope = freq_dev_slope
    self.device = device

    self._window = getattr(torch, f'{self.win_type}_window')(self.win_size).cpu().numpy()

  def _calculate(self, audio: np.ndarray) -> torch.Tensor:
    """
    Extracts amplitudes and frequencies of sinusoids from the audio signal.
    Args:
      audio: np.ndarray, audio signal of shape (batch_size, n_samples)
    Returns:
      torch.Tensor, extracted frequencies and amplitudes of shape (batch_size, n_sines*2, n_frames).
    """
    batch_size = audio.shape[0]

    freqs = []
    amps = []
    for i in range(batch_size):
      curr_freqs, curr_amps, _ = sineModelAnal(
          audio[i],
          self.fs,
          self._window,
          self.win_size,
          self.win_size//4,
          self.threshold,
          maxnSines=self.n_sines,
          minSineDur=self.min_sine_dur,
          freqDevSlope=self.freq_dev_slope
      )
      freqs.append(curr_freqs)
      amps.append(curr_amps)

    freqs = np.stack(freqs, axis=0) / (self.fs / 2)  # Normalize frequencies to [0, 1]
    amps = (np.stack(amps, axis=0) + 120) / 120

    params = torch.from_numpy(np.concatenate([freqs, amps], axis=-1)).permute(0, 2, 1).to(dtype=torch.float32, device=self.device)

    return params
