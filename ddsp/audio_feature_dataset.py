import torch
from torch.utils.data import Dataset
import librosa as li
import os
from glob import glob
import lmdb
import pickle
import hashlib
import numpy as np
import math

from ddsp.feature_extractors import LibrosaFeatureExtractor

from typing import Callable


class AudioFeatureDataset(Dataset):
  def __init__(self, dataset_path: str, n_signal: int, sampling_rate: int = 44100, smoothing_kernel_size: int = 257, transform_fn: Callable = None, device: str = 'cuda'):
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
      'loudness': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_LOUDNESS, resampling_factor=1, smoothing_kernel_size=smoothing_kernel_size, postprocess=True),
      'spectral_centroid': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_CENTROID, resampling_factor=1, smoothing_kernel_size=smoothing_kernel_size, postprocess=True),
      # 'spectral_flatness': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_FLATNESS, resampling_factor=1, postprocess=True),
      # 'spectral_bandwidth': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_BANDWIDTH, resampling_factor=1, postprocess=True),
    }

    # Create cache key based on dataset parameters
    cache_key = self._create_cache_key(dataset_path, n_signal, sampling_rate, smoothing_kernel_size)
    self._db_path = os.path.join(os.path.dirname(dataset_path), f"audio_cache_{cache_key}.lmdb")

    # Try to load from LMDB first
    if os.path.exists(self._db_path):
      print(f"Loading from cache: {self._db_path}")
      self._load_from_cache()
    else:
      print(f"Creating new cache: {self._db_path}")
      self._create_cache(dataset_path)

  def _create_cache_key(self, dataset_path: str, n_signal: int, sampling_rate: int, smoothing_kernel_size: int) -> str:
    """Create a unique cache key based on dataset parameters."""
    key_string = f"{dataset_path}_{n_signal}_{sampling_rate}_{smoothing_kernel_size}"
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


  def _iter_chunks(self, N: int, chunk_samps: int):
    for i in range(0, N, chunk_samps):
        j = min(i + chunk_samps, N)
        yield i, j


  def _put_with_resize(self, env, put_fn):
    """Run a write block; auto-grow map_size on MapFullError."""
    while True:
      try:
        with env.begin(write=True) as txn:
          put_fn(txn)
        break
      except lmdb.MapFullError:
        info = env.info()
        env.set_mapsize(int(info["map_size"] * 1.5))


  def _save_to_cache(self):
    """
    Save audio/features to LMDB in chunks to stay under LMDB's ~2 GiB per-value limit.
    """
    chunk_samps = 10_000_000
    N = int(self._audio.shape[0])
    F = int(self._features.shape[-1])  # expect 4

    env = lmdb.open(
      self._db_path,
      map_size=16 * 1024**3,  # 16 GiB to start
      readahead=False,
      writemap=False,
      max_dbs=1,
      sync=True,
      metasync=True,
      subdir=True,
    )

    # write metadata
    def put_meta(txn):
      metadata = {
        "dataset_length": self._dataset_length,
        "n_signal": self._n_signal,
        "sampling_rate": self._sampling_rate,
        "total_samps": N,
        "feat_dim": F,
        "chunk_samps": chunk_samps,
        "dtype": "float32",
      }
      print(f"Saving metadata: {metadata}")
      txn.put(b"metadata", pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))

    self._put_with_resize(env, put_meta)

    bytes_in_txn = 0
    txn = env.begin(write=True)
    try:
      for idx, (a, b) in enumerate(self._iter_chunks(N, chunk_samps)):
        aud = self._audio[a:b].detach().to("cpu").contiguous().to(torch.float32)
        fea = self._features[a:b].detach().to("cpu").contiguous().to(torch.float32)

        aud_bytes = aud.numpy().tobytes(order="C")
        fea_bytes = fea.numpy().tobytes(order="C")

        txn.put(f"audio:{idx:08d}".encode(), aud_bytes)
        txn.put(f"feat:{idx:08d}".encode(), fea_bytes)

        bytes_in_txn += len(aud_bytes) + len(fea_bytes)

        if bytes_in_txn >= 256 * 1024**2:  # commit every ~256 MiB
          txn.commit()
          txn = env.begin(write=True)
          bytes_in_txn = 0

      txn.commit()
    finally:
      env.close()


  def _load_from_cache(self):
    """Load audio/features from LMDB chunks and reassemble."""
    env = lmdb.open(self._db_path, readonly=True, lock=False, readahead=True, subdir=True)

    with env.begin() as txn:
      meta = pickle.loads(txn.get(b"metadata"))
      self._dataset_length = int(meta["dataset_length"])
      N = int(meta.get("total_samps", 0)) or (self._dataset_length * self._n_signal)
      F = int(meta.get("feat_dim", 4))
      cs = int(meta["chunk_samps"])
      dtype = np.float32

      n_chunks = math.ceil(N / cs)
      audio_cpu = torch.empty(N, dtype=torch.float32)
      feats_cpu = torch.empty((N, F), dtype=torch.float32)

      pos = 0
      for idx in range(n_chunks):
        a_key = f"audio:{idx:08d}".encode()
        f_key = f"feat:{idx:08d}".encode()

        a_buf = txn.get(a_key)
        f_buf = txn.get(f_key)
        if a_buf is None or f_buf is None:
          raise RuntimeError(f"Missing chunk {idx} in LMDB")

        a_arr = np.frombuffer(a_buf, dtype=dtype)
        n_samps = a_arr.shape[0]
        f_arr = np.frombuffer(f_buf, dtype=dtype).reshape(n_samps, F)

        audio_cpu[pos:pos+n_samps] = torch.from_numpy(a_arr)
        feats_cpu[pos:pos+n_samps] = torch.from_numpy(f_arr)
        pos += n_samps

    env.close()
    self._audio = audio_cpu.to(self._device, non_blocking=True)
    self._features = feats_cpu.to(self._device, non_blocking=True)


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
      features.append(feat)

    features = torch.stack(features, dim=-1).squeeze(0)  # Stack features along the last dimension
    return features
