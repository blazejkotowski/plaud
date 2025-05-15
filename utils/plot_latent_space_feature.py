import argparse
import torch
from sklearn.decomposition import PCA
import librosa
from ddsp import AudioDataset
import numpy as np
import matplotlib.pyplot as plt


def main():
  parser = argparse.ArgumentParser(description="Plot DDSP latent space with PCA")
  parser.add_argument("--model_path", type=str, help="Path to exported DDSP model (.ts file)")
  parser.add_argument("--dataset_path", type=str, help="Path to audio dataset")
  args = parser.parse_args()

  # Constants
  SAMPLE_RATE = 44100  # Change if your dataset uses a different sample rate
  CHUNK_SECONDS = 1.5
  N_SIGNAL = int(SAMPLE_RATE * CHUNK_SECONDS)

  # Load model
  model = torch.jit.load(args.model_path)
  model.eval()

  # Load dataset
  dataset = AudioDataset(args.dataset_path, n_signal=N_SIGNAL)

  all_latents = []
  with torch.no_grad():
    for i in range(len(dataset)):
      audio = dataset[i].unsqueeze(0)  # Add batch dimension
      latents = model.encode(audio.cpu()).squeeze(0)
      all_latents.append(latents.cpu().numpy())

  all_latents = np.hstack(all_latents).transpose()

  # PCA to 2D
  pca = PCA(n_components=2)
  latents_2d = pca.fit_transform(all_latents)


  # Create grid in PCA space
  x = np.linspace(latents_2d[:, 0].min(), latents_2d[:, 0].max(), 100)
  y = np.linspace(latents_2d[:, 1].min(), latents_2d[:, 1].max(), 100)
  xx, yy = np.meshgrid(x, y)
  grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

  # Inverse transform to original latent space
  latents_grid = pca.inverse_transform(grid_points)

  loudness_map = np.zeros(len(latents_grid))
  with torch.no_grad():
    for i, latent in enumerate(latents_grid):
      latent_tensor = torch.tensor(latent, dtype=torch.float32)
      latent_tensor = latent_tensor.reshape(1, -1, 1)
      audio = model.decode(latent_tensor).cpu().numpy().squeeze()
      loudness = librosa.feature.rms(y=audio).mean()
      loudness_map[i] = loudness

  loudness_map = loudness_map.reshape(xx.shape)

  plt.figure(figsize=(8, 6))
  plt.imshow(loudness_map, extent=[x.min(), x.max(), y.min(), y.max()],
         origin='lower', aspect='auto', cmap='viridis')
  plt.xlabel('PCA 1')
  plt.ylabel('PCA 2')
  plt.title('Loudness Heatmap in Latent Space')
  plt.colorbar(label='Loudness (RMS)')
  plt.savefig('loudnes.png')

if __name__ == '__main__':
  main()
