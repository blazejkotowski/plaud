import torch

from music2latent import EncoderDecoder
from music2latent.audio import to_representation_encoder

class M2LLoss(torch.nn.Module):
  """
  M2L Loss based on the Music2Latent model.
  This loss computes the mean squared error between the embeddings of two audio signals
  after passing them through the Music2Latent encoder.
  """
  def __init__(self):
    super().__init__()
    self.m2l = EncoderDecoder()
    self.m2l.gen.eval()

  def forward(self, x: torch.tensor, y: torch.tensor):
    # with torch.no_grad():
    x_repr = to_representation_encoder(x.detach().squeeze(1))
    x_emb = self.m2l.gen.encoder(x_repr, extract_features=True)

    y_repr = to_representation_encoder(y.squeeze(1))
    y_emb = self.m2l.gen.encoder(y_repr, extract_features=True)

    return torch.nn.functional.mse_loss(y_emb, x_emb).mean()
    # return 1-torch.nn.functional.cosine_similarity(x_emb, y_emb).mean()
