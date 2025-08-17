import torch
from torch.utils.data import Dataset
import librosa as li
import os
from glob import glob
import lmdb
import pickle
import hashlib

from ddsp.feature_extractors import LibrosaFeatureExtractor
from ddsp.feature_extractors.utils import normalize_feature, smoothen_feature

from typing import Callable

class AudioFeatureDataset(Dataset):
  def __init__(self, dataset_path: str, n_signal: int, sampling_rate: int = 44100, transform_fn: Callable = None, device: str = 'cuda'):
    """
    Arguments:
      - dataset_path: str, the path to the dataset
      - n_signal: int, the size of the audio chunks, in samples
      - sampling_rate: int, the sampling rate of the audio
      - device: str, the device to use
    """
    self._device = device
    self._n_signal = n_signal
    self._sampling_rate = sampling_rate
    self._transform_fn = transform_fn

    self._extractors = {
      'loudness': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_LOUDNESS, resampling_factor=1),
      'spectral_centroid': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_CENTROID, resampling_factor=1),
      'spectral_flatness': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_FLATNESS, resampling_factor=1),
      'spectral_bandwidth': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_BANDWIDTH, resampling_factor=1),
    }

    # Create cache key based on dataset parameters
    cache_key = self._create_cache_key(dataset_path, n_signal, sampling_rate)
    self._db_path = os.path.join(os.path.dirname(dataset_path), f"audio_cache_{cache_key}.lmdb")

    # Try to load from LMDB first
    if os.path.exists(self._db_path):
      print(f"Loading from cache: {self._db_path}")
      self._load_from_cache()
    else:
      print(f"Creating new cache: {self._db_path}")
      self._create_cache(dataset_path)

  def _create_cache_key(self, dataset_path: str, n_signal: int, sampling_rate: int) -> str:
    """Create a unique cache key based on dataset parameters."""
    key_string = f"{dataset_path}_{n_signal}_{sampling_rate}"
    return hashlib.md5(key_string.encode()).hexdigest()[:8]

  def _create_cache(self, dataset_path: str):
    """Load audio, process it, and save to LMDB."""
    # Load and process audio
    self._audio = self._load_dataset(dataset_path).to(self._device)
    self._dataset_length = int(len(self._audio) // self._n_signal)

    # Extract features
    self._features = self._extract_features()

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
      }
      txn.put(b'metadata', pickle.dumps(metadata))

      # Save audio (move to CPU for serialization)
      txn.put(b'audio', pickle.dumps(self._audio.cpu()))

      # save features
      txn.put(b'features', pickle.dumps(self._features.cpu()))

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

      # Load features if they exist
      features_data = txn.get(b'features')
      if features_data is not None:
        features_cpu = pickle.loads(features_data)
        self._features = features_cpu.to(self._device)

    env.close()


  def __len__(self):
    return self._dataset_length


  def get_datapoint(self, idx):
    sample_start = int(idx * self._n_signal)
    sample_end = int(sample_start + self._n_signal)

    if sample_end > len(self._audio):
      audio = torch.cat([self._audio[sample_start:], self._audio[:sample_end - len(self._audio)]])
      features = torch.cat([self._features[sample_start:], self._features[:sample_end - len(self._features)]])

    else:
      audio = self._audio[sample_start:sample_end]
      features = self._features[sample_start:sample_end]

    if self._transform_fn is not None:
      audio = self._transform_fn(audio)

    return audio, features


  def __getitem__(self, idx):
    idx = idx % self._dataset_length

    audio, features = self.get_datapoint(idx)

    return audio, features


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


  def _extract_features(self):
    """
    Extract features from the audio dataset.
    This is a placeholder function, as the actual feature extraction logic is not provided.
    """
    # process audio in chunks
    features = []
    for extractor in self._extractors.values():
      feat = extractor(self._audio)
      feat = smoothen_feature(feat, window_size=256+1)  # Smoothen the feature
      feat = normalize_feature(feat, low=5.0, high=95.0, dim=-1)  # Normalize the feature
      features.append(feat)

    features = torch.stack(features, dim=-1).squeeze(0)  # Stack features along the last dimension
    return features
