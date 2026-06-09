from tqdm import tqdm

import torch
import torchcrepe
import torch.nn.functional as F

from .base_extractor import BaseExtractor
from .utils import normalize_feature, smoothen_feature

class CrepePitchExtractor(BaseExtractor):
  """
  Extracts the pitch from an audio signal
  using CREPE algorithm
  """
  def init(self, postprocess: bool = False, smoothing_kernel_size: int = 257,  *args, **kwargs):
    """
    Args:
      - postprocess: bool, whether to apply postprocessing:
        > clamping between .05 and .95 percentiles,
        > normalization between [0, 1]
        > and smoothing to the extracted features
      - smoothing_kernel_size: in case of processing, the kernel size of the gaussian filtere
    """
    super(BaseExtractor, self).__init__(*args, **kwargs)
    self._smoothing_kernel_size = smoothing_kernel_size
    self._postprocess = postprocess

  def _calculate(self, audio: torch.Tensor, fs: int) -> torch.Tensor:
    """
    Args:
      - audio: torch.Tensor [T_audio] or [B, T_audio], the input audio tensor
      - fs: int, sampling rate
    Returns:
      - features: torch.Tensor [T_audio, C] or [B, T_audio, C], the extracted features at audio rate
    """

    squeeze_back = False
    if audio.ndim == 1:
      audio = audio.unsqueeze(0)
      squeeze_back = True

    chunk_duration = 10  # seconds
    chunk_samples = chunk_duration * fs
    pitches = []

    with torch.no_grad():
      # process in chunks to avoid memory overload
      for i in tqdm(range(0, audio.shape[-1], chunk_samples), 'pitch extraction'):
        chunk = audio[..., i:i+chunk_samples]
        pitch_chunk = torchcrepe.predict(chunk, fs, fmin=30, model='full', device='cuda')
        pitches.append(pitch_chunk)

    pitch = torch.cat(pitches, dim=-1)
    pitch = F.interpolate(pitch.unsqueeze(0), size=audio.shape[-1], mode='nearest').squeeze(0)

    # if self._postprocess:
    pitch = normalize_feature(pitch, low=5.0, high=95.0, dim=-1)

    if squeeze_back:
      pitch = pitch.squeeze(0)

    return pitch

