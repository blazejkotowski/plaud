#!/usr/bin/env python3
import os
import json
import argparse
import gc
import subprocess

import torch
from torch.utils.data import DataLoader, Subset

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from ddsp import DDSP, AudioFeatureDataset
from ddsp.callbacks import BetaWarmupEpochCallback
from ddsp.synths import NoiseBandSynth
from ddsp.utils import find_checkpoint

def get_args():
  p = argparse.ArgumentParser()
  # required
  p.add_argument("--dataset_path", required=True, type=str)
  p.add_argument("--model_name", required=True, type=str)
  p.add_argument("--experiment_root", required=True, type=str)
  # optional model params (JSON string or key=val pairs)
  p.add_argument("--model_params_json", type=str, default="{}")
  p.add_argument("--sampling_rate", type=int, default=22050)
  p.add_argument("--chunk_duration", type=float, default=2.0)
  p.add_argument("--batch_size", type=int, default=8)

  # training hyperparams
  p.add_argument("--beta", type=float, default=0.1)
  p.add_argument("--warmup_start", type=int, default=50)
  p.add_argument("--warmup_end", type=int, default=75)
  p.add_argument("--max_epochs", type=int, default=100)
  p.add_argument("--learning_rate", type=float, default=1e-3)
  p.add_argument("--precision", type=str, default="16-mixed")  # "32-true" if you prefer

  # export options
  p.add_argument("--export_target_fs", type=int, default=44100)
  p.add_argument("--export_type", type=str, default="last")  # or "best"

  return p.parse_args()

def get_dataset_split(dataset_path, n_signal, batch_size, validation_split=0.2):
  dataset_A = AudioFeatureDataset(dataset_path=dataset_path, n_signal=n_signal)
  total_len = len(dataset_A)
  val_len = int(validation_split * total_len)
  indices = torch.randperm(total_len).tolist()
  val_indices = indices[:val_len]
  train_indices = indices[val_len:]
  train_set = Subset(dataset_A, train_indices)
  val_set = Subset(dataset_A, val_indices)
  train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=0)
  return train_loader, val_loader

def build_ddsp_model(sampling_rate, learning_rate, model_params):
  n_filters = model_params.pop("n_filters", 512)
  latent_size = model_params.pop("latent_size", 2)
  num_params = model_params.pop("num_params", 2)
  resampling_factor = model_params.pop("resampling_factor", 32)
  nbn = NoiseBandSynth.to_config(n_filters=n_filters, fs=sampling_rate, resampling_factor=resampling_factor)
  ddsp = DDSP(
    synth_configs=[nbn],
    fs=sampling_rate,
    latent_size=latent_size,
    latent_smoothing_kernel=model_params.pop("latent_smoothing_kernel", 1),
    decoder_gru_layers=model_params.pop("gru_layers", 1),
    num_params=num_params,
    learning_rate=learning_rate,
    perceptual_loss_weight=0,
    plateau_patience=20,
    resampling_factor=resampling_factor,
  )
  return ddsp

def build_trainer(model_training_path, beta, warmup_start, warmup_end, max_epochs, precision):
  os.makedirs(model_training_path, exist_ok=True)
  callbacks = [
    BetaWarmupEpochCallback(beta=beta, start_epoch=warmup_start, end_epoch=warmup_end),
    ModelCheckpoint(
      filename="best",
      monitor="val_loss",
      mode="min",
      save_top_k=1,
      save_last=True,
      dirpath=model_training_path
    ),
  ]
  trainer = Trainer(
    callbacks=callbacks,
    max_epochs=max_epochs,
    accelerator="cuda",
    precision=precision,
  )
  return trainer

def export_model(models_root, model_name, training_path, export_type, target_fs):
  synth_output_path = os.path.join(models_root, f"{model_name}.ts")
  subprocess.run([
    "python", "-m", "cli.export",
    "--model_directory", training_path,
    "--output_path", synth_output_path,
    "--type", export_type,
    "--target_fs", str(target_fs)
  ], check=True)

def main():
  args = get_args()

  # Optional allocator tuning (can also be set by parent process env)
  os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

  experiment_root = args.experiment_root
  models_root = os.path.join(experiment_root, "models")
  training_root = os.path.join(experiment_root, "training")
  os.makedirs(models_root, exist_ok=True)
  os.makedirs(training_root, exist_ok=True)

  n_signal = int(args.chunk_duration * args.sampling_rate)

  model_params = json.loads(args.model_params_json)

  train_loader, val_loader = get_dataset_split(
    dataset_path=args.dataset_path,
    n_signal=n_signal,
    batch_size=args.batch_size
  )

  model = build_ddsp_model(
    sampling_rate=args.sampling_rate,
    learning_rate=args.learning_rate,
    model_params=model_params.copy()
  )

  model_training_path = os.path.join(training_root, args.model_name)
  trainer = build_trainer(
    model_training_path=model_training_path,
    beta=args.beta,
    warmup_start=args.warmup_start,
    warmup_end=args.warmup_end,
    max_epochs=args.max_epochs,
    precision=args.precision
  )

  ckpt = find_checkpoint(model_training_path, return_none=True, typ="last")
  print("Found ckpt:", ckpt)
  trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)

  export_model(
    models_root=models_root,
    model_name=args.model_name,
    training_path=model_training_path,
    export_type=args.export_type,
    target_fs=args.export_target_fs
  )

  # Best-effort cleanup before process exits
  del model, trainer, train_loader, val_loader
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()

if __name__ == "__main__":
  main()
