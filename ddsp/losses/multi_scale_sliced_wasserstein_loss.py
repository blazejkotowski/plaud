import torch

from typing import List

from .sliced_wasserstein_loss import SlicedWassersteinLoss

class MultiScaleSlicedWassersteinLoss(torch.nn.Module):
  """
  Multiscale Sliced Wasserstein Loss (MSS-Wasserstein Loss).

  Arguments:
    - windows: List[torch.Tensor], a list of window tensors to use for the ST
    - kwargs: Additional keyword arguments for the SlicedWassersteinLoss function.
  """
  def __init__(self, windows: List[torch.Tensor], **kwargs):
    super(MultiScaleSlicedWassersteinLoss, self).__init__()
    self.losses = [SlicedWassersteinLoss(window=window, **kwargs) for window in windows]

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the MSS-Wasserstein loss between two audio signals.

    Args:
      - x: torch.Tensor, the first audio signal
      - y: torch.Tensor, the second audio signal

    Returns:
      - loss: torch.Tensor, the computed MSS-Wasserstein loss
    """
    total_loss = 0.0
    for loss_fn in self.losses:
      total_loss += loss_fn(x, y)
    return total_loss / len(self.losses)

