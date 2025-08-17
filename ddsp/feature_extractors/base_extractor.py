import torch
import torch.nn.functional as F


class BaseExtractor(object):
  """
  Base class for feature extractors
  """

  def __init__(self, resampling_factor: int):
    """
    Args:
      - resampling_factor: int, the factor to resample the extracted features
    """
    self._resampling_factor = resampling_factor


  def __call__(self, audio: torch.Tensor, *args) -> torch.Tensor:
    """
    Args:
      - audio: torch.Tensor[batch_size, n_samples], the input audio tensor
      - args: Tuple, additional arguments
    Returns:
      - features: torch.Tensor[batch_size, n_samples, n_features], the extracted features
    """
    features = self._calculate(audio, *args)
    return F.interpolate(features.unsqueeze(1), scale_factor=float(1/self._resampling_factor), mode='linear').squeeze(1)


  def _calculate(self, audio: torch.Tensor, *args) -> torch.Tensor:
    """
    Implementation of the feature extractor
    """
    raise NotImplementedError
