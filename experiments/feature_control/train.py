
from ddsp import DDSP, AudioFeatureDataset
from ddsp.callbacks import BetaWarmupEpochCallback
from ddsp.synths import NoiseBandSynth
from ddsp.utils import find_checkpoint

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

from torch.utils.data import DataLoader, Subset

import torch
import gc
import subprocess

torch.set_float32_matmul_precision('medium')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_device('cuda')

import os

experiment_root = '/home/btadeusz/code/ddsp_vae/experiments/feature_control'
models_root = os.path.join(experiment_root, 'models')
training_root = os.path.join(experiment_root, 'training')

# Dataset parameters
chunk_duration = 2.0
sampling_rate = 22050
n_signal = chunk_duration * sampling_rate
batch_size = 8

# Model parameters
latent_size = num_params = 2
max_freq = sampling_rate//2
n_filters = 512
# gru_layers = 2
# latent_smoothing_kernel = 513
# resampling_factor = 128
# n_sines = 100

# Training parameters
warmup_start = 50
warmup_end = 100
beta = 0.1
max_epochs = 200
learning_rate = 1e-3

def get_dataset_split(dataset_path, validation_split=0.2):
  """
  Splits the dataset into training and validation sets.
  """
  generator=torch.Generator(device='cuda')

  dataset_A = AudioFeatureDataset(dataset_path=dataset_path, n_signal=n_signal)
  total_len = len(dataset_A)

  val_len = int(validation_split * total_len)  # 20% for validation
  indices = torch.randperm(total_len, generator=generator)

  val_indices = indices[:val_len]
  train_indices = indices[val_len:]

  train_set = Subset(dataset_A, train_indices)
  val_set = Subset(dataset_A, val_indices)

  train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0, generator=generator)
  val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=0, generator=generator)

  return train_loader, val_loader


def build_ddsp_model(smoothing_kernel=1, gru_layers=1, resampling_factor=32):
  """
  Builds the DDSP model with the specified configurations.
  """
  nbn = NoiseBandSynth.to_config(
    n_filters=n_filters,
    fs=sampling_rate,
    resampling_factor=resampling_factor,
  )

  # sines = SineSynth.to_config(
  #   n_sines=n_sines,
  #   fs=sampling_rate,
  # )

  ddsp = DDSP(
    # synth_configs=[nbn, sines],
    synth_configs=[nbn],
    fs=sampling_rate,
    latent_size=latent_size,
    latent_smoothing_kernel=smoothing_kernel,
    decoder_gru_layers=gru_layers,
    num_params=num_params,
    learning_rate=learning_rate,
    perceptual_loss_weight=0,
    plateau_patience=20,
    resampling_factor=resampling_factor,
  ).to('cuda')

  return ddsp


def build_trainer(model_training_path):
  training_callbacks = []
  beta_warmup = BetaWarmupEpochCallback(
    beta=beta,
    start_epoch=warmup_start,
    end_epoch=warmup_end
  )
  training_callbacks.append(beta_warmup)

  best_checkpoint_callback = ModelCheckpoint(
    filename='best',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=True,
    dirpath=model_training_path
  )
  training_callbacks.append(best_checkpoint_callback)

  trainer = Trainer(
    callbacks=training_callbacks,
    max_epochs=max_epochs,
    accelerator='cuda',
    # precision=16,
  )

  return trainer


def train_on_dataset(dataset_path, model_name=None, model_params={}):
  """
  Trains the DDSP model on a specific dataset.
  """
  train_loader, val_loader = get_dataset_split(dataset_path)

  model = build_ddsp_model(**model_params)

  model_training_path = os.path.join(training_root, model_name)
  trainer = build_trainer(model_training_path)

  ckpt = find_checkpoint(model_training_path, return_none=True, typ='best')
  print("Found ckpt: ", ckpt)
  trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)

  return model, trainer, train_loader, val_loader, model_training_path

def export_model(model_name, training_path):
  """
  Exports the trained model to a specified path.
  """
  synth_output_path = os.path.join(models_root, f'{model_name}.ts')
  subprocess.run([
      "python", "-m", "cli.export",
      "--model_directory", training_path,
      "--output_path", synth_output_path,
      "--type", "last",
      "--target_fs", "44100"
  ])

def cleanup_cuda(*objs):
  for o in objs:
    try:
        del o
    except:
        pass
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()  # helps with fragmentation on some drivers

# # Martsman Reactive
# model_name = 'martsman-feature-control-22050fs-reactive'
# dataset_path = '/mnt/mariadata/datasets/beat-artists/martsman/processed'
# model, trainer, train_loader, val_loader, training_path = train_on_dataset(dataset_path, model_name=model_name, model_params={'resampling_factor': 128, 'gru_layers': 2, 'smoothing_kernel': 513})
# export_model(model_name, training_path)
# cleanup_cuda(model, trainer, train_loader, val_loader)


# # Martsman Non-Reactive
# model_name = 'martsman-feature-control-22050fs'
# model, trainer, train_loader, val_loader, training_path = train_on_dataset(dataset_path, model_name=model_name, model_params={'resampling_factor': 32, 'gru_layers': 1, 'smoothing_kernel': 1})
# export_model(model_name, training_path)
# cleanup_cuda(model, trainer, train_loader, val_loader)


# Ankersmit Non-reactive
model_name = 'ankersmit-feature-control-22050fs'
dataset_path = '/mnt/mariadata/datasets/noise-artists/ankersmit/processed'
model, trainer, train_loader, val_loader, training_path = train_on_dataset(dataset_path, model_name=model_name, model_params={'resampling_factor': 32, 'gru_layers': 1, 'smoothing_kernel': 1})
export_model(model_name, training_path)
cleanup_cuda(model, trainer, train_loader, val_loader)


# Klein reactive
model_name = 'klein-feature-control-22050fs-reactive'
dataset_path = '/mnt/mariadata/datasets/noise-artists/klein/processed'
model, trainer, train_loader, val_loader, training_path = train_on_dataset(dataset_path, model_name=model_name, model_params={'resampling_factor': 128, 'gru_layers': 2, 'smoothing_kernel': 513})
export_model(model_name, training_path)
cleanup_cuda(model, trainer, train_loader, val_loader)


# Hecker reactive
model_name = 'hecker-feature-control-22050fs-reactive'
dataset_path = '/mnt/mariadata/datasets/noise-artists/hecker/processed'
model, trainer, train_loader, val_loader, training_path = train_on_dataset(dataset_path, model_name=model_name, model_params={'resampling_factor': 128, 'gru_layers': 2, 'smoothing_kernel': 513})
export_model(model_name, training_path)
cleanup_cuda(model, trainer, train_loader, val_loader)

