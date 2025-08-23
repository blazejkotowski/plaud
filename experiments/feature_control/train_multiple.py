#!/usr/bin/env python3
import os
import json
import subprocess

EXPERIMENT_ROOT = "/home/btadeusz/code/ddsp_vae/experiments/feature_control"

# Optional allocator tuning for all children
COMMON_ENV = os.environ.copy()
COMMON_ENV.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

COMMON_ARGS = [
  "--experiment_root", EXPERIMENT_ROOT,
  "--sampling_rate", "22050",
  "--chunk_duration", "2.0",
  "--batch_size", "8",
  "--max_epochs", "100",
  "--precision", "16-mixed",  # or "32-true" if you want full precision
]

def run_one(dataset_path, model_name, model_params):
  mp_json = json.dumps(model_params)
  cmd = [
    "python", "train_one.py",
    "--dataset_path", dataset_path,
    "--model_name", model_name,
    "--model_params_json", mp_json,
  ] + COMMON_ARGS
  print("\n=== Running:", model_name, "===\n")
  subprocess.run(cmd, check=True, env=COMMON_ENV)

def main():
  # Martsman Reactive
  run_one(
    dataset_path="/mnt/mariadata/datasets/beat-artists/martsman/processed",
    model_name="martsman-feature-control-22050fs-ultra-reactive",
    model_params={"resampling_factor": 512, "gru_layers": 2, "latent_smoothing_kernel": 513, "n_filters": 512}
  )

#  # Martsman Non-Reactive
#  run_one(
#    dataset_path="/mnt/mariadata/datasets/beat-artists/martsman/processed",
#    model_name="martsman-feature-control-22050fs",
#    model_params={"resampling_factor": 32, "gru_layers": 1, "latent_smoothing_kernel": 1, "n_filters": 512}
#  )
#
#  # Ankersmit Non-reactive
#  run_one(
#    dataset_path="/mnt/mariadata/datasets/noise-artists/ankersmit/processed",
#    model_name="ankersmit-feature-control-22050fs",
#    model_params={"resampling_factor": 32, "gru_layers": 1, "latent_smoothing_kernel": 1, "n_filters": 512}
#  )
#
#  # Klein reactive
#  run_one(
#    dataset_path="/mnt/mariadata/datasets/noise-artists/klein/processed",
#    model_name="klein-feature-control-22050fs-reactive",
#    model_params={"resampling_factor": 128, "gru_layers": 2, "latent_smoothing_kernel": 513, "n_filters": 512}
#  )
#
#  # Hecker reactive
#  run_one(
#    dataset_path="/mnt/mariadata/datasets/noise-artists/hecker/processed",
#    model_name="hecker-feature-control-22050fs-reactive",
#    model_params={"resampling_factor": 128, "gru_layers": 2, "latent_smoothing_kernel": 513, "n_filters": 512}
#  )

if __name__ == "__main__":
  main()
