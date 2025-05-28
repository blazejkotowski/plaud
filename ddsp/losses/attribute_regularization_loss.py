import torch
from ddsp.feature_extractors import BaseExtractor

from typing import List

class AttributeRegularizationLoss(torch.nn.Module):
  """
  Attribute Regularization Loss for Variational Auto-Encoders.
  This loss is used to regularize the latent space of a VAE by ensuring that
  the latent dimensions are aligned with the attributes of the audio signal.
  It computes the mean absolute error between the distances of the latent
  variables and the distances of the attributes extracted from the audio signal.

  The loss is computed as described in the paper:
  Pati, A. Lerch A, 2020, "Attribute-based Regularization of Latent Spaces for Variational Auto-Encoders"
  https://arxiv.org/abs/2004.05485

  Args:
    - feature_extractors: List[BaseExtractor], a list of feature extractors to compute the attributes
      from the audio signal. Each extractor should implement the `__call__` method.
  """
  def __init__(self, feature_extractors: List[BaseExtractor] = []):
    self._feature_extractors = feature_extractors


  def forward(self, z: torch.Tensor, y_audio: torch.Tensor) -> torch.Tensor:
    """
    Compute the attribute regularization loss.

    Args:
      - z: torch.Tensor[batch_size, latent_size], the latent variables
      - y_audio: torch.Tensor[batch_size, n_signal], the output audio signal
    Returns:
      - arg_reg_loss: torch.Tensor[1], the argument regularization loss
    """
    loss = torch.zeros(1, device=z.device)

    # MAE: mean absolute error between the tanh
    mae = lambda latent_distance, attribute_distance: torch.mean(torch.abs(torch.tanh(latent_distance) - torch.sign(attribute_distance)))

    # Self-distance matrix between the pair of values in a 1d tensor
    distance_matrix = lambda x: x.unsqueeze(0) - x.unsqueeze(1)

    # Calculate the attributes of the audio signal
    attributes = [extractor(y_audio.detach().cpu().squeeze()).to(y_audio) for extractor in self._feature_extractors]

    # Regularize the latent dimensions according to the attributes
    for dimension_id, attribute_values in enumerate(attributes):
      # distance for latent values
      latent_values = z[: dimension_id]
      latent_distance = distance_matrix(latent_values)

      # distance for the selected attribute values
      attribute_distance = distance_matrix(attribute_values)

      # compute the mae loss between latent distance and attribute distance
      loss = loss + mae(latent_distance, attribute_distance)

    # normalize the loss
    loss /= len(attributes)
    return loss
