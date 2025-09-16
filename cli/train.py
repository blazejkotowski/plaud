import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import torch
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_device('cuda')

import argparse
import os

from torch.utils.data import DataLoader, Subset, random_split
from ddsp import DDSP, AudioFeatureDataset
from ddsp.synths import BendableNoiseBandSynth
from ddsp.callbacks import BetaWarmupCallback

from ddsp.prior import Prior, PriorDataset
import shutil

from ddsp.utils import find_checkpoint

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_path', help='Directory of the training sound/sounds')
  parser.add_argument('--model_name', type=str, help='Name of the model')
  parser.add_argument('--training_dir', type=str, default='training', help='Directory to save the training logs')
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
  parser.add_argument('--n_bands', type=int, default=512, help='Number of bands of the filter bank')
  parser.add_argument('--model_fs', type=int, default=22050, help='Sampling rate of the model')
  parser.add_argument('--output_fs', type=int, default=44100, help='Sampling rate of the model')
  parser.add_argument('--resampling_factor', type=int, default=32, help='Synth resampling factor')
  parser.add_argument('--latent_size', type=int, default=2, help='Dimensionality of the latent space')
  parser.add_argument('--smoothing_kernel_size', type=int, default=65, help='Size of the smoothing kernel for the latent space')
  parser.add_argument('--gru_layers', type=int, default=2, help='Number of GRU layers in the decoder')
  parser.add_argument('--synth_learning_rate', type=float, default=1e-4, help='Synthesiser learning rate')
  parser.add_argument('--plateau_patience_steps', type=int, default=100000, help='Number of steps with no improvement after which learning rate will be reduced')
  parser.add_argument('--synth_capacity', type=int, default=16, help='Capacity of the model')
  parser.add_argument('--synth_max_epochs', type=int, default=500, help='Maximum number of epochs to train synthesiser')
  parser.add_argument('--beta', type=float, default=0.001, help='Beta parameter for the beta-VAE loss')
  parser.add_argument('--synth_force_restart', type=bool, default=False, help='Force restart the training. Ignore the existing checkpoint.')
  parser.add_argument('--adversarial_loss', type=bool, default=True, help='Use adversarial loss')
  parser.add_argument('--mixed_precision', type=bool, default=True, help='Use mixed precision training')
  parser.add_argument('--train_prior', type=bool, default=False, help='Train prior model')
  parser.add_argument('--prior_learning_rate', type=float, default=1e-4, help='Prior learning rate')
  parser.add_argument('--prior_capacity', type=int, default=4, help='Capacity of the prior model')
  parser.add_argument('--prior_max_epochs', type=int, default=1000, help='Maximum number of epochs to train prior')
  parser.add_argument('--prior_dataset_stride_factor', type=float, default=0.2, help='Stride factor for the prior dataset')
  parser.add_argument('--prior_force_restart', type=bool, default=False, help='Force restart the prior training. Ignore the existing checkpoint.')
  # parser.add_argument('--warmup_cycle', type=int, default=50, help='Number of epochs for a full beta cycle')
  config = parser.parse_args()

  #### SYNTHESIZER TRAINING ###
  print("Training synthesizer...")
  # Create training directory
  synth_training_path = os.path.join(config.training_dir, 'synth', config.model_name)
  os.makedirs(synth_training_path, exist_ok=True)

  n_signal = int(2 * config.model_fs) # 2s

  # Load Dataset
  synth_dataset = AudioFeatureDataset(
    dataset_path=config.dataset_path,
    n_signal=n_signal,
    sampling_rate=config.model_fs,
  )

  generator = torch.Generator(device='cuda')

  # Split into training and validation
  synth_total_len = len(synth_dataset)
  synth_val_len = int(0.2 * synth_total_len)
  synth_indices = torch.randperm(synth_total_len, generator=generator)

  synth_val_indices = synth_indices[:synth_val_len]
  train_indices = synth_indices # all dataset for overfitting

  train_set = Subset(synth_dataset, train_indices)
  val_set = Subset(synth_dataset, synth_val_indices)

  train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0, generator=generator)
  val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0, generator=generator)

  # The synth
  bnbn = BendableNoiseBandSynth.to_config(
    n_filters=config.n_bands,
    fs=config.model_fs,
    resampling_factor=config.resampling_factor,
  )

  # Core model
  ddsp = DDSP(
    synth_configs=[bnbn],
    fs=config.model_fs,
    resampling_factor=config.resampling_factor,
    latent_size=config.latent_size,
    num_params=config.latent_size,
    latent_smoothing_kernel=config.smoothing_kernel_size,
    decoder_gru_layers=config.gru_layers,
    learning_rate=config.synth_learning_rate,
    perceptual_loss_weight=0,
    plateau_patience=config.plateau_patience_steps,
    capacity=config.synth_capacity,
    adversarial_loss=config.adversarial_loss,
    n_features=2,
  )

  # Tensorboard
  tb_logger = TensorBoardLogger(synth_training_path, name=config.model_name)

  synth_training_callbacks = []

  # Warming up beta parameter
  synth_beta_warmup = BetaWarmupCallback(
    beta=config.beta,
    start_steps=50,
    end_steps=100,
  )
  synth_training_callbacks.append(synth_beta_warmup)

  # Define the checkpoint callback
  model_checkpoint = ModelCheckpoint(
    filename='best',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=True,
  )
  synth_training_callbacks.append(model_checkpoint)

  # Configure the trainer
  precision = 16 if config.mixed_precision else 32
  synth_trainer = L.Trainer(
    callbacks=synth_training_callbacks,
    max_epochs=config.synth_max_epochs,
    accelerator='cuda',
    precision=precision,
    log_every_n_steps=10,
    logger=tb_logger,
  )

  # Try to find previously trained checkpoint
  synth_ckpt_path = find_checkpoint(synth_training_path, return_none=True, typ='last') if not config.synth_force_restart else None
  if config.synth_force_restart:
    print("Force restart set to True")
  else:
    if synth_ckpt_path is not None:
      print(f"Resuming from checkpoint: {synth_ckpt_path}")
    else:
      print(f"Did not find any checkpoint at {synth_training_path}.")

  # Start synth training
  synth_trainer.fit(model=ddsp,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=synth_ckpt_path
  )
  print(f"Synthesizer training completed. Your checkpoints are in {synth_training_path}")

  ### PRIOR TRAINING ###
  if not config.train_prior:
    exit(0)


  print("Training prior...")
  prior_batch_size = 256
  sequence_length = 64

  embedding_dim = 8 * config.prior_capacity
  nhead = 2 * config.prior_capacity
  num_layers = 4
  quantization_channels = 16 * config.prior_capacity

  prior_model_name = f'{config.model_name}-prior'
  prior_training_path = os.path.join(config.training_dir, 'prior', prior_model_name)

  if config.prior_force_restart:
    shutil.rmtree(prior_training_path, ignore_errors=True)
    os.makedirs(prior_training_path, exist_ok=True)

  prior_dataset = PriorDataset(
    audio_dataset=synth_dataset,
    encoding_model=ddsp,
    sequence_length=sequence_length+1,
    sampling_rate=config.model_fs,
    device='cuda',
    stride_factor=config.prior_dataset_stride_factor
  )
  normalization_dict = prior_dataset.normalization_dict

  prior_train_set, prior_val_set = random_split(prior_dataset, [0.8, 0.2], generator=generator)
  prior_train_loader = DataLoader(prior_train_set, batch_size=prior_batch_size, shuffle=True, generator=generator)
  prior_val_loader = DataLoader(prior_val_set, batch_size=prior_batch_size, shuffle=False, generator=generator)

  prior = Prior(
    latent_size=prior_dataset[0].shape[-1],
    embedding_dim=embedding_dim,
    quantization_channels=quantization_channels,
    max_len=sequence_length,
    lr=config.prior_learning_rate,
    nhead=nhead,
    num_layers=num_layers,
    normalization_dict=normalization_dict
  )

  prior_logger = TensorBoardLogger(prior_training_path, name=prior_model_name)

  prior_callbacks = []
  prior_checkpoint_callback = ModelCheckpoint(
    dirpath=prior_training_path,
    filename='best',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=True,
  )
  prior_callbacks.append(prior_checkpoint_callback)

  early_stopping = EarlyStopping(monitor='val_acc', patience=1000, mode='max', stopping_threshold=0.95)
  prior_callbacks.append(early_stopping)

  prior_trainer = L.Trainer(
    callbacks=prior_callbacks,
    accelerator='cuda',
    log_every_n_steps=4,
    logger=prior_logger,
    max_epochs=config.prior_max_epochs
  )

  prior_ckpt_path = find_checkpoint(prior_training_path, return_none=True)
  if prior_ckpt_path is not None:
    print(f'Resuming prior training from checkpoint {prior_ckpt_path}')

  prior_trainer.fit(
    model=prior,
    train_dataloaders=prior_train_loader,
    val_dataloaders=prior_val_loader,
    ckpt_path=prior_ckpt_path
  )

  print(f"Prior training completed. Your checkpoints are in {prior_training_path}")
