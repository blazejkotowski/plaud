import torch
from torch.utils.data import Dataset
import librosa as li
import os
from glob import glob
import lmdb
import pickle
import hashlib

from ddsp.feature_extractors import SinusoidsExtractor

from typing import Callable, Optional

class AudioDataset(Dataset):
  def __init__(self, dataset_path: str, n_signal: int, sampling_rate: int = 44100, n_sines: Optional[int] = None, transform_fn: Callable = None, device: str = 'cuda'):
    """
    Arguments:
      - dataset_path: str, the path to the dataset
      - n_signal: int, the size of the audio chunks, in samples
      - sampling_rate: int, the sampling rate of the audio
      - n_sines: int, the number of sine waves to extract (default: None, meaning no extraction)
      - device: str, the device to use
    """
    self._device = device
    self._n_signal = n_signal
    self._sampling_rate = sampling_rate
    self._n_sines = n_sines
    self._transform_fn = transform_fn

    # Create cache key based on dataset parameters
    cache_key = self._create_cache_key(dataset_path, n_signal, sampling_rate, n_sines)
    self._db_path = os.path.join(os.path.dirname(dataset_path), f"audio_cache_{cache_key}.lmdb")

    # Try to load from LMDB first
    if os.path.exists(self._db_path):
      print(f"Loading from cache: {self._db_path}")
      self._load_from_cache()
    else:
      print(f"Creating new cache: {self._db_path}")
      self._create_cache(dataset_path)

  def _create_cache_key(self, dataset_path: str, n_signal: int, sampling_rate: int, n_sines: Optional[int]) -> str:
    """Create a unique cache key based on dataset parameters."""
    key_string = f"{dataset_path}_{n_signal}_{sampling_rate}_{n_sines}"
    return hashlib.md5(key_string.encode()).hexdigest()[:8]

  def _create_cache(self, dataset_path: str):
    """Load audio, process it, and save to LMDB."""
    # Load and process audio
    self._audio = self._load_dataset(dataset_path).to(self._device)
    self._dataset_length = int(len(self._audio) // self._n_signal)

    if self._n_sines is not None:
      self._extractor = SinusoidsExtractor(
        n_sines=self._n_sines,
        fs=self._sampling_rate,
        win_type='hann',
        win_size=512,
        threshold=-80,
        min_sine_dur=0.001,
        freq_dev_slope=0.001,
      )
      self._sine_params = self._extract_sinusoids()
    else:
      self._sine_params = None

    # Save to LMDB
    self._save_to_cache()

  def _save_to_cache(self):
    """Save audio and sine parameters to LMDB."""
    env = lmdb.open(self._db_path, map_size=int(1e12))  # 1TB max size

    with env.begin(write=True) as txn:
      # Save metadata
      metadata = {
        'dataset_length': self._dataset_length,
        'n_signal': self._n_signal,
        'sampling_rate': self._sampling_rate,
        'n_sines': self._n_sines
      }
      txn.put(b'metadata', pickle.dumps(metadata))

      # Save audio (move to CPU for serialization)
      audio_cpu = self._audio.cpu()
      txn.put(b'audio', pickle.dumps(audio_cpu))

      # Save sine parameters if they exist
      if self._sine_params is not None:
        sine_params_cpu = self._sine_params.cpu()
        txn.put(b'sine_params', pickle.dumps(sine_params_cpu))

    env.close()

  def _load_from_cache(self):
    """Load audio and sine parameters from LMDB."""
    env = lmdb.open(self._db_path, readonly=True)

    with env.begin() as txn:
      # Load metadata
      metadata = pickle.loads(txn.get(b'metadata'))
      self._dataset_length = metadata['dataset_length']

      # Load audio
      audio_cpu = pickle.loads(txn.get(b'audio'))
      self._audio = audio_cpu.to(self._device)

      # Load sine parameters if they exist
      sine_params_data = txn.get(b'sine_params')
      if sine_params_data is not None:
        sine_params_cpu = pickle.loads(sine_params_data)
        self._sine_params = sine_params_cpu.to(self._device)

        # Recreate extractor for consistency
        self._extractor = SinusoidsExtractor(
          n_sines=self._n_sines,
          fs=self._sampling_rate,
          win_type='hann',
          win_size=512,
          threshold=-80,
          min_sine_dur=0.001,
          freq_dev_slope=0.001,
        )
      else:
        self._sine_params = None

    env.close()

  def __len__(self):
    return self._dataset_length

  def get_audio(self, idx):
    sample_start = int(idx * self._n_signal)
    sample_end = int(sample_start + self._n_signal)

    if sample_end > len(self._audio):
      audio = torch.cat([self._audio[sample_start:], self._audio[:sample_end - len(self._audio)]])
    else:
      audio = self._audio[sample_start:sample_end]

    if self._transform_fn is not None:
      audio = self._transform_fn(audio)

    return audio

  def __getitem__(self, idx):
    idx = idx % self._dataset_length

    audio = self.get_audio(idx)

    if self._n_sines is not None:
      return audio, self._sine_params[idx]
    else:
      return audio

  def _load_dataset(self, path: str) -> torch.Tensor:
    """
    Load and concat entire dataset into single tensor.

    Arguments:
      - path: str, path to the dataset
    Returns:
      - audio: torch.Tensor, the audio tensor
    """
    audio = torch.tensor([], device=self._device)
    for filepath in glob(os.path.join(path, '**', '*.wav'), recursive=True):
      x = self._load_file(filepath)
      audio = torch.concat([audio, torch.from_numpy(x).to(self._device)], dim = 0)
    return audio

  def _load_file(self, path: str):
    """
    Load an audio file from a path.
    """
    audio, _ = li.load(path, sr = self._sampling_rate, mono = True)
    return audio

  def _extract_sinusoids(self):
    """
    Process sine parameters from the audio dataset.
    This is a placeholder function, as the actual processing logic is not provided.
    """
    # process audio in chunks
    sine_params = []
    for i in range(len(self)):
      chunk = self.get_audio(i)
      params = self._extractor(chunk.unsqueeze(0).cpu().numpy())
      sine_params.append(params)

    return torch.stack(sine_params, dim=1).squeeze()
