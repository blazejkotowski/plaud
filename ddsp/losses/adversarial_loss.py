import torch
import torch.nn.functional as F

class AdversarialLoss(torch.nn.Module):
  """
  Adversarial loss, based on GAN discriminator module.
  """

  def __init__(self):
    super(AdversarialLoss, self).__init__()
    self.discriminator = Discriminator()

  def forward(self, y_real: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Args:
      y_real: [torch.Tensor], Real audio signal. Shape: [batch, n_samples]
      y_pred: [torch.Tensor], Predicted audio signal. Shape: [batch, n_samples]
    Returns:
      loss: [torch.Tensor], Adversarial loss. Shape: []
    """
    pred_real = self.discriminator(y_real)
    pred_fake_det = self.discriminator(y_pred.detach()) # Detach to avoid backprop through discriminator

    loss = 0

    # Hinge loss
    loss_real = F.relu(1+pred_real[-1]).mean()
    loss += loss_real
    loss_fake = F.relu(1-pred_fake_det[-1]).mean()
    loss += loss_fake

    # Feature matching loss
    for feat in range(len(pred_real)-1):
      loss += F.l1_loss(pred_real[feat], pred_fake[feat].detach())

    return loss


class Discriminator(torch.nn.Module):
  """
  MelGAN inspired multi-scale convolutional discriminator network.
  """
  def __init__(self):
    super(Discriminator, self).__init__()

  def foward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x: [torch.Tensor], Input audio signal. Shape: [batch, n_samples]
    Returns:
      pred: [torch.Tensor], Discriminator prediction. Shape: [batch, 1]
    """
    pass
