import torch


class BaseExtractor(object):
  """Base class for feature extractors.

  Contract:
    - __call__(audio, fs) must return audio-rate features aligned to the input
      audio length (no internal downsampling in the output).
  """

  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, audio: torch.Tensor, fs: int, *args) -> torch.Tensor:
    """
    Args:
      - audio: torch.Tensor [T_audio] or [B, T_audio]
      - fs: int, sampling rate in Hz
    Returns:
      - features: torch.Tensor [T_audio, C] or [B, T_audio, C] (audio-rate)
    """
    return self._calculate(audio, fs, *args)

  def _calculate(self, audio: torch.Tensor, fs: int, *args) -> torch.Tensor:
    """Implementation of the feature extractor at audio rate."""
    raise NotImplementedError
