import torch
import laion_clap

class CLAPLoss(torch.nn.Module):
  """
  CLAP Loss based on the CLAP model.
  This loss computes the cosine similarity between the audio embeddings of two audio signals
  after passing them through the CLAP model.
  """

  def __init__(self):
    super().__init__()
    self.clap = laion_clap.CLAP_Module(enable_fusion=False, device='cuda')
    self.clap.requires_grad_(False)
    self.clap.load_ckpt()

  def forward(self, x: torch.tensor, y: torch.tensor):
    with torch.no_grad():
      x_emb = self.clap.get_audio_embedding_from_data(x.reshape(x.shape[0], -1).float(), use_tensor=True)
      y_emb = self.clap.get_audio_embedding_from_data(y.reshape(y.shape[0], -1).float(), use_tensor=True)

    return 1-torch.nn.functional.cosine_similarity(x_emb, y_emb).mean()
