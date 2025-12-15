import torch
import librosa
import torch.nn.functional as F

from .base_extractor import BaseExtractor, register_feature_extractor
from .utils import normalize_feature, smoothen_feature

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

  def __init__(self, feature_fn_name: str, smoothing_kernel_size: int = 257, postprocess: bool = False,  *args, **kwargs):
    """
    Args:
      - feature_fn_name: str, the name of the librosa feature extraction function to use
      - args: additional arguments for the base extractor
      - postprocess: bool, whether to apply postprocessing:
        > clamping between .05 and .95 percentiles,
        > normalization between [0, 1]
        > and smoothing to the extracted features
      - kwargs: additional keyword arguments for the base extractor
    """
    super(LibrosaFeatureExtractor, self).__init__(*args, **kwargs)
    self._feature_fn = getattr(librosa.feature, feature_fn_name)
    self._smoothing_kernel_size = smoothing_kernel_size
    self._postprocess = postprocess


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
    feat = normalize_feature(feat, low=5.0, high=95.0, dim=-1)
    feat = F.interpolate(feat.unsqueeze(0), size=audio.shape[-1], mode='linear').squeeze(0)
    if self._smoothing_kernel_size > 1:
      print(f"Smoothing kernel size={self._smoothing_kernel_size}, applying smoothing to the feature")
      feat = smoothen_feature(feat, window_size=self._smoothing_kernel_size)  # Smoothen the feature
    else:
      print("Smoothing kernel size <=1, skipping smoothing of the feature")

    return feat
