import torch
import librosa
import torch.nn.functional as F

from .base_extractor import BaseExtractor

class PitchExtractor(BaseExtractor):
  """
  Extracts the pitch from an audio signal
  """

  def __init__(self, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'), *args, **kwargs):
    super(PitchExtractor, self).__init__(*args, **kwargs)

    self._fmin = fmin
    self._fmax = fmax

  def _calculate(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the pitch extractor with librosa. Extracts the pitch in Hz and returns
    and interpolated tensor with values for each audio sample.

    Args:
      - audio: torch.Tensor[batch_size, n_samples], the input audio tensor
    Returns:
      - pitches: torch.Tensor[batch_size, n_samples, 1], the extracted pitch
    """
    # Processes the batch of audio parallely
    pitches = librosa.yin(audio.numpy(), fmin=self._fmin, fmax=self._fmax)
    pitches = torch.tensor(pitches, dtype=torch.float32)
    pitches = F.interpolate(pitches.unsqueeze(0), size=audio.shape[-1], mode='linear').squeeze(0)

    return pitches.unsqueeze(-1)
