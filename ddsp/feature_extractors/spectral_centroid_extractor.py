import torch
import librosa
import torch.nn.functional as F

from .base_extractor import BaseExtractor

class SpectralCentroidExtractor(BaseExtractor):
  """
  Extracts the spectral centroid from an audio signal
  """
  def __init__(self, *args, **kwargs):
    super(SpectralCentroidExtractor, self).__init__(*args, **kwargs)

  def _calculate(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the spectral centroid extractor with librosa. Extracts the spectral centroid
    and returns and interpolated tensor with values for each audio sample.

    Args:
      - audio: torch.Tensor[batch_size, n_samples], the input audio tensor
    Returns:
      - spectral_centroid: torch.Tensor[batch_size, n_samples, 1], the extracted spectral centroid
    """
    # Processes the batch of audio parallely
    spectral_centroid = librosa.feature.spectral_centroid(y=audio.squeeze().numpy())
    spectral_centroid = torch.tensor(spectral_centroid, dtype=torch.float32)
    spectral_centroid = F.interpolate(spectral_centroid, size=audio.shape[-1], mode='linear')

    return spectral_centroid.squeeze()
