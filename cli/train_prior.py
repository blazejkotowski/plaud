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
from ddsp.prior.lmdb_cache import build_or_load_prior_cache_from_cfg
from ddsp.interfaces import build_control_space
from ddsp.utils import find_checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig):
  """Hydra-driven training for Prior.

  Expected cfg fields:
    cfg.prior.model: {embedding_dim, quantization_channels, nhead, num_layers, dropout, lr}
    cfg.prior.dataset: {hdf5_path?, seq_len, stride_factor}
    cfg.prior.training: {batch_size, num_workers, max_epochs}
  """
  L.seed_everything(cfg.get('seed', 42))

  # Dataset (auto-cache controls/latents for prior)
  in_memory = cfg.prior.dataset.get('in_memory', True)

  # Infer control_space and synth configs from experiment config (same as DDSP training)
  control_space = build_control_space(cfg.data.control_space)
  synth_configs = []
  for s in cfg.model.synths:
    synth_configs.append({"class": s.type, "params": dict(s.params)})

  lmdb_path, stats = build_or_load_prior_cache_from_cfg(
    cfg,
    control_space=control_space,
    synth_configs=synth_configs,
    device='cuda' if torch.cuda.is_available() else 'cpu',
  )
  print(f"Prior cache: {lmdb_path} (rebuilt={stats.get('rebuilt', False)})")
  ds = PriorSequenceDataset(path=lmdb_path, in_memory=in_memory)
  num_workers = cfg.prior.training.get('num_workers', 4)
  pin_memory = torch.cuda.is_available()
  prefetch_factor = cfg.prior.training.get('prefetch_factor', 2)
  dl = DataLoader(
    ds,
    batch_size=cfg.prior.training.get('batch_size', 16),
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=num_workers > 0,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
  )

  # Output/checkpoints
  model_name = cfg.experiment.name
  training_dir = cfg.experiment.training_dir
  output_dir = os.path.join(training_dir, 'prior', model_name)
  os.makedirs(output_dir, exist_ok=True)

  checkpoint_cb = ModelCheckpoint(
    dirpath=output_dir,
    filename='best',
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
    num_controls=ds.num_controls,
    embedding_dim=cfg.prior.model.embedding_dim,
    quantization_channels=cfg.prior.model.quantization_channels,
    nhead=cfg.prior.model.nhead,
    num_layers=cfg.prior.model.num_layers,
    dropout=cfg.prior.model.dropout,
    max_len=cfg.prior.model.max_len,
    lr=cfg.prior.training.lr,
    normalization_dict=cfg.prior.dataset.get('normalization', None),
    device=device
  )

  # Trainer
  force_restart = cfg.prior.training.get('force_restart', False)
  ckpt_path = None
  if not force_restart:
    ckpt_path = find_checkpoint(output_dir, return_none=True, typ='last')
    if ckpt_path is not None:
      print(f"Resuming from checkpoint: {ckpt_path}")
  elif force_restart:
    print("Force restart requested; starting from scratch.")

  trainer = L.Trainer(
    max_epochs=cfg.prior.training.get('max_epochs', 10),
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


