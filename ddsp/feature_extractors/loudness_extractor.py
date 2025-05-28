import torch
import librosa
import torch.nn.functional as F

from .base_extractor import BaseExtractor


class LoudnessExtractor(BaseExtractor):
  """
  Extracts the loudness from an audio signal
  """
  def __init__(self, *args, **kwargs):
    super(LoudnessExtractor, self).__init__(*args, **kwargs)

  def _calculate(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the loudness extractor with librosa. Extracts the loudness
    and returns and interpolated tensor with values for each audio sample.

    Args:
      - audio: torch.Tensor[batch_size, n_samples], the input audio tensor
    Returns:
      - loudness: torch.Tensor[batch_size, n_samples, 1], the extracted loudness
    """
    # Processes the batch of audio parallely
    loudness = librosa.feature.rms(audio.numpy())
    loudness = torch.tensor(loudness, dtype=torch.float32)
    loudness = F.interpolate(loudness.unsqueeze(0), size=audio.shape[-1], mode='linear').squeeze(0)

    return loudness.unsqueeze(-1)
