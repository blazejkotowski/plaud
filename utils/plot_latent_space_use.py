import argparse
import torch
from sklearn.decomposition import PCA
import numpy as np
from ddsp import AudioDataset

import matplotlib.pyplot as plt


def main():
  parser = argparse.ArgumentParser(description="Plot DDSP latent space with PCA")
  parser.add_argument("--model_path", type=str, help="Path to exported DDSP model (.ts file)")
  parser.add_argument("--dataset_path", type=str, help="Path to audio dataset")
  args = parser.parse_args()

  model_name = args.model_path.split('/')[-1]

  # Constants
  SAMPLE_RATE = 44100  # Change if your dataset uses a different sample rate
  CHUNK_SECONDS = 1.5
  N_SIGNAL = int(SAMPLE_RATE * CHUNK_SECONDS)

  # Load model
  model = torch.jit.load(args.model_path).to('cuda')
  model.eval()

  # Load dataset
  dataset = AudioDataset(args.dataset_path, n_signal=N_SIGNAL)

  all_latents = []
  with torch.no_grad():
    for i in range(len(dataset)):
      audio = dataset[i].unsqueeze(0)  # Add batch dimension
      latents = model.encode(audio).squeeze(0)
      all_latents.append(latents.cpu().numpy())

  all_latents = np.hstack(all_latents).transpose()

  # PCA to 2D
  pca = PCA(n_components=2)
  latents_2d = pca.fit_transform(all_latents)

  # Plot
  plt.figure(figsize=(8, 6))
  plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.1, s=1)
  plt.title(f"{model_name}")
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.tight_layout()
  plt.savefig(f'{model_name}.png')
  plt.show()

if __name__ == "__main__":
  main()
