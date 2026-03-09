import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import torch
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

import os
import shutil
import math

from torch.utils.data import DataLoader, Subset, random_split
from ddsp import DDSP, AudioFeatureDataset
from ddsp.synths import BendableNoiseBandSynth
from ddsp.callbacks import BetaWarmupCallback
from ddsp.interfaces import build_control_space
from ddsp.augmentations import build_audio_augmentation_pipeline
from ddsp.registry import SYNTHS

from ddsp.prior import Prior, PriorDataset
from ddsp.utils import find_checkpoint

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
  # Set default device/tensor type only during CLI execution
  if torch.cuda.is_available():
    try:
      torch.set_default_device('cuda')
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
    except Exception:
      pass
  # Shared audio params
  fs = int(cfg.audio.fs)
  resampling_factor = int(cfg.model.resampling_factor)
  chunk_duration_s = float(cfg.audio.chunk_duration_s)
  n_signal = int(fs * chunk_duration_s)

  # Paths
  dataset_path = cfg.data.dataset_path
  model_name = cfg.experiment.name
  training_dir = cfg.experiment.training_dir
  synth_training_path = os.path.join(training_dir, 'synth', model_name)
  os.makedirs(synth_training_path, exist_ok=True)

  # Control space
  control_space = build_control_space(cfg.model.control_space)

  # Optional audio augmentations
  transform_fn = build_audio_augmentation_pipeline(
    getattr(cfg.data, 'augmentations', None),
    n_signal=n_signal,
    sampling_rate=fs,
  )

  # Dataset
  synth_dataset = AudioFeatureDataset(
    dataset_path=dataset_path,
    n_signal=n_signal,
    sampling_rate=fs,
    resampling_factor=resampling_factor,
    control_space=control_space,
    transform_fn=transform_fn,
  )

  # Dataloaders
  batch_size = int(cfg.trainer.batch_size)
  generator = torch.Generator(device='cuda')
  synth_total_len = len(synth_dataset)
  synth_val_len = int(0.2 * synth_total_len)
  synth_indices = torch.randperm(synth_total_len, generator=generator)
  synth_val_indices = synth_indices[:synth_val_len]
  train_indices = synth_indices
  train_loader = DataLoader(Subset(synth_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
  val_loader = DataLoader(Subset(synth_dataset, synth_val_indices), batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)

  # Synths from config
  synth_configs = []
  for s in cfg.model.synths:
    synth_configs.append({"class": s.type, "params": dict(s.params)})

  # Capture the active Hydra config name for logging/hparams
  try:
    active_config_name = HydraConfig.get().job.config_name
  except Exception:
    active_config_name = None

  # Core model
  ddsp = DDSP(
    control_space=control_space,
    synth_configs=synth_configs,
    fs=fs,
    resampling_factor=resampling_factor,
    latent_smoothing_kernel=int(cfg.model.latent_smoothing_kernel),
    decoder_gru_layers=int(cfg.model.decoder_gru_layers),
    learning_rate=float(cfg.model.learning_rate),
    perceptual_loss_weight=float(getattr(cfg.model, 'perceptual_loss_weight', 0.0)),
    plateau_patience=int(cfg.model.plateau_patience),
    capacity=int(cfg.model.capacity),
    losses=[dict(l) for l in getattr(cfg, 'losses', [])],
    adversarial_loss=bool(cfg.adversarial.enabled),
    adv_g_start_epoch=int(cfg.adversarial.schedule.g_start_epoch),
    adv_d_start_epoch=int(cfg.adversarial.schedule.d_start_epoch),
    adv_gen_weight=float(cfg.adversarial.weights.gen),
    adv_disc_weight=float(cfg.adversarial.weights.disc),
    adv_fm_weight=float(cfg.adversarial.weights.fm),
    config_name=active_config_name,
    # Encoder type selection
    encoder_type=str(getattr(cfg.model, 'encoder_type', 'melspec')),
    encoder_hidden_channels=list(getattr(cfg.model, 'encoder_hidden_channels', [32, 64, 128])),
    encoder_strides=list(getattr(cfg.model, 'encoder_strides', [4, 4, 2])),
    # Frequency stability loss (for ComplexSineSynth)
    freq_stability_weight=float(getattr(cfg.model, 'freq_stability_weight', 0.0)),
    freq_dev_offset=float(getattr(cfg.model, 'freq_dev_offset', 20.0)),
    freq_dev_slope=float(getattr(cfg.model, 'freq_dev_slope', 0.01)),
    phase_continuity_weight=float(getattr(cfg.model, 'phase_continuity_weight', 0.0)),
    orth_embed_dim=int(getattr(cfg.model, 'orth_embed_dim', 0)),
    distill_weight=float(getattr(cfg.model, 'distill_weight', 0.0)),
    decoder_recons_weight=float(getattr(cfg.model, 'decoder_recons_weight', 0.0)),
    freeze_table=bool(getattr(cfg.model, 'freeze_table', False)),
    stft_n_fft=int(getattr(cfg.model, 'stft_n_fft', 0)),
    freq_bin_selection=bool(getattr(cfg.model, 'freq_bin_selection', False)),
    freq_n_fft=int(getattr(cfg.model, 'freq_n_fft', 4096)),
    # Per-chunk param table for direct synth optimization
    n_data_chunks=len(synth_dataset) if bool(getattr(cfg.model, 'use_param_table', False)) else 0,
    data_T_ctl=int(math.ceil(n_signal / resampling_factor)) if bool(getattr(cfg.model, 'use_param_table', False)) else 0,
  )

  # FFT-initialize the param table from dataset audio (places optimisation
  # in the correct frequency basin — proven 97.8% vs 22.5% from random init)
  if bool(getattr(cfg.model, 'use_param_table', False)):
    pretrained_table_path = str(getattr(cfg.model, 'pretrained_table_path', ''))
    if pretrained_table_path:
      print(f"Loading pre-trained param table from: {pretrained_table_path}")
      ckpt = torch.load(pretrained_table_path, map_location='cpu')
      table_data = ckpt['state_dict']['_synth_param_table']  # [n_chunks, n_raw, T]
    else:
      ddsp.init_param_table_from_fft(synth_dataset)
      table_data = ddsp._synth_param_table.data.clone()

    # Sort sines by mean frequency within each chunk for consistent ordering.
    # The STFT init assigns frequencies to arbitrary sine indices; sorting
    # ensures sine 0 is always the lowest freq, sine 1 the next, etc.
    # This eliminates permutation ambiguity when distilling to per-sine MLPs.
    pps = ddsp._params_per_sine
    n_chunks, n_raw, T = table_data.shape
    n_sines = n_raw // pps
    if n_sines * pps == n_raw and n_sines > 0:
      omega = table_data[:, :n_sines, :]  # [chunks, sines, T]
      mean_omega = omega.abs().mean(dim=-1)  # [chunks, sines]
      sort_idx = mean_omega.argsort(dim=-1)  # [chunks, sines] — ascending by freq
      # Rearrange all param groups by the frequency sort order
      sorted_table = torch.zeros_like(table_data)
      for c in range(n_chunks):
        idx = sort_idx[c]  # [sines]
        for g in range(pps):
          sorted_table[c, g*n_sines:(g+1)*n_sines, :] = table_data[c, g*n_sines:(g+1)*n_sines, :][idx]
      table_data = sorted_table
      print(f"  Sorted sines by frequency (consistent ordering across chunks)")
    ddsp._synth_param_table.data.copy_(table_data)
    print(f"  Table shape: {table_data.shape}, frozen={ddsp._freeze_table}")

  # Tensorboard
  tb_logger = TensorBoardLogger(synth_training_path, name=model_name)

  # Callbacks
  callbacks = []
  if 'beta' in cfg.model:
    callbacks.append(BetaWarmupCallback(beta=float(cfg.model.beta), start_steps=50, end_steps=100))

  callbacks.append(ModelCheckpoint(dirpath=synth_training_path, filename='best', monitor='val_loss', mode='min', save_top_k=1, save_last=True))

  # Trainer
  synth_trainer = L.Trainer(
    callbacks=callbacks,
    max_epochs=int(cfg.trainer.max_epochs),
    log_every_n_steps=int(cfg.trainer.log_every_n_steps),
    logger=tb_logger,
  )

  # Resume checkpoint
  synth_ckpt_path = find_checkpoint(synth_training_path, return_none=True, typ='last') if not bool(cfg.trainer.force_restart) else None
  if cfg.trainer.force_restart:
    print("Force restart set to True")
  elif synth_ckpt_path is not None:
    print(f"Resuming from checkpoint: {synth_ckpt_path}")

  # Train
  synth_trainer.fit(model=ddsp, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=synth_ckpt_path)
  print(f"Synthesizer training completed. Your checkpoints are in {synth_training_path}")


if __name__ == '__main__':
  main()
