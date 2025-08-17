import torch
import librosa
import torch.nn.functional as F

from .base_extractor import BaseExtractor, register_feature_extractor

@register_feature_extractor
class LibrosaFeatureExtractor(BaseExtractor):
  """
  Extracts the loudness from an audio signal
  """
  FN_SPECTRAL_CENTROID = 'spectral_centroid'
  FN_ZERO_CROSSING = 'zero_crossing_rate'
  FN_LOUDNESS = 'rms'
  FN_SPECTRAL_FLATNESS = 'spectral_flatness'
  FN_SPECTRAL_ROLLOFF = 'spectral_rolloff'
  FN_SPECTRAL_BANDWIDTH = 'spectral_bandwidth'

  def __init__(self, feature_fn_name: str, *args, **kwargs):
    """
    Args:
      - feature_fn_name: str, the name of the librosa feature extraction function to use
      - args: additional arguments for the base extractor
      - kwargs: additional keyword arguments for the base extractor
    """
    super(LibrosaFeatureExtractor, self).__init__(*args, **kwargs)
    self._feature_fn = getattr(librosa.feature, feature_fn_name)
    librosa.feature.rms

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
    feat = self._feature_fn(y=audio.cpu().numpy(), center=False).squeeze(-2)
    if feat.ndim == 1:
      feat = feat[None, :]
    feat = torch.tensor(feat, dtype=torch.float32)
    feat = F.interpolate(feat.unsqueeze(0), size=audio.shape[-1], mode='linear').squeeze(0)

    return feat
