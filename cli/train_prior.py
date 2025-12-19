import os
import os
import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig

from ddsp.prior.prior import Prior
from ddsp.prior.dataset import PriorSequenceDataset
from ddsp.utils import find_checkpoint


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
    in_memory=cfg.dataset.get('in_memory', True),
  )
  num_workers = cfg.training.get('num_workers', 4)
  pin_memory = torch.cuda.is_available()
  prefetch_factor = cfg.training.get('prefetch_factor', 2)
  dl = DataLoader(
    ds,
    batch_size=cfg.training.get('batch_size', 16),
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=num_workers > 0,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
  )

  # Output/checkpoints
  output_dir = cfg.training.get('output_dir', os.path.join('training', 'prior', 'default'))
  os.makedirs(output_dir, exist_ok=True)

  checkpoint_cb = ModelCheckpoint(
    dirpath=output_dir,
    filename='prior-epoch{epoch:04d}',
    save_last=True,
    save_top_k=1,
    monitor='loss',
    mode='min',
    save_on_train_epoch_end=True,
  )
  logger = TensorBoardLogger(save_dir=output_dir, name='logs')

  # Model
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    device=device
  )

  # Trainer
  resume = cfg.training.get('resume', True)
  force_restart = cfg.training.get('force_restart', False)
  ckpt_path = None
  if resume and not force_restart:
    ckpt_path = find_checkpoint(output_dir, return_none=True, typ='last')
    if ckpt_path is not None:
      print(f"Resuming from checkpoint: {ckpt_path}")
  elif force_restart:
    print("Force restart requested; starting from scratch.")

  trainer = L.Trainer(
    max_epochs=cfg.training.get('max_epochs', 10),
    accelerator='gpu' if device == 'cuda' else 'cpu',
    devices=1,
    log_every_n_steps=10,
    callbacks=[checkpoint_cb],
    logger=logger,
    default_root_dir=output_dir,
  )

  trainer.fit(model, dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
  main()


