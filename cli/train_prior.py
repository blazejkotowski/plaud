import os
import torch
import lightning as L
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from ddsp.prior.prior import Prior
from ddsp.prior.dataset import PriorSequenceDataset


@hydra.main(version_base=None, config_path="../configs", config_name="experiment_prior")
def main(cfg: DictConfig):
  """Hydra-driven training for Prior.

  Expected cfg fields:
    cfg.prior: {latent_size, embedding_dim, quantization_channels, nhead, num_layers, dropout, max_len, lr}
    cfg.dataset: {hdf5_path?, num_sequences, seq_len, latent_size}
    cfg.training: {batch_size, num_workers, max_epochs}
  """
  L.seed_everything(cfg.get('seed', 42))

  # Dataset
  ds = PriorSequenceDataset(
    hdf5_path=cfg.dataset.get('hdf5_path', None),
    num_sequences=cfg.dataset.get('num_sequences', 128),
    seq_len=cfg.dataset.get('seq_len', 256),
    latent_size=cfg.dataset.get('latent_size', cfg.prior.latent_size),
  )
  dl = DataLoader(ds, batch_size=cfg.training.get('batch_size', 16), shuffle=True, num_workers=cfg.training.get('num_workers', 4))

  # Model
  model = Prior(
    latent_size=ds.latent_size,
    embedding_dim=cfg.prior.embedding_dim,
    quantization_channels=cfg.prior.quantization_channels,
    nhead=cfg.prior.nhead,
    num_layers=cfg.prior.num_layers,
    dropout=cfg.prior.dropout,
    max_len=cfg.prior.max_len,
    lr=cfg.prior.lr,
    normalization_dict=cfg.get('normalization', None),
    device='cuda' if torch.cuda.is_available() else 'cpu'
  )

  # Trainer
  trainer = L.Trainer(
    max_epochs=cfg.training.get('max_epochs', 10),
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    log_every_n_steps=10,
  )

  trainer.fit(model, dl)


if __name__ == "__main__":
  main()


