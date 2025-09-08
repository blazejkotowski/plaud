from torch.utils.data import Dataset
from torch import jit
import torch
import torch.nn.functional as F

# import librosa as li
# from glob import glob
# import os
import math
from typing import Tuple, List, Dict, Optional

torch.set_default_dtype(torch.float32)

from ddsp.audio_feature_dataset import AudioFeatureDataset
from ddsp.feature_extractors import LibrosaFeatureExtractor

class PriorDataset(Dataset):
  def __init__(self,
               audio_dataset: AudioFeatureDataset,
               encoding_model_path: str,
               sequence_length: int,
               sampling_rate: int = 44100,
               stride_factor: float = 1.0,
               device: Optional[str] = None):
    self._device = device
    self._sequence_length = sequence_length
    self._sampling_rate = sampling_rate
    self._stride_factor = stride_factor

    self._audio_ds = audio_dataset
    self._audio = self._audio_ds._audio
    self._features = self._audio_ds._features

    vae = jit.load(encoding_model_path, map_location=device).pretrained.to(device)
    vae.encoder.streaming = False
    self._vae = vae
    self._resampling_factor = vae._resampling_factor

    self._encodings, self.normalization_dict = self._build()

  def __len__(self) -> int:
    return len(self._encodings)

  def __getitem__(self, idx: int) -> torch.Tensor:
    return self._encodings[idx]

  @torch.no_grad()
  def _build(self) -> Tuple[List[torch.Tensor], dict]:
    windows = []
    print("Encoding audio dataset...")

    l_chunk = self._sampling_rate * 40  # process in 40s chunks (as before)
    for i_chunk in range(math.ceil(self._audio.size(0) / l_chunk)):
      audio = self._audio[i_chunk*l_chunk : (i_chunk+1)*l_chunk].to(self._device)
      features = self._features[i_chunk*l_chunk:(i_chunk+1)*l_chunk].to(self._device)

      mu, scale = self._vae.encoder(audio.unsqueeze(0))
      z, _ = self._vae.encoder.reparametrize(mu, scale)
      z = self._vae._smooth_latents(z).squeeze(0)

      # align features to latent frames
      feat = features.t().unsqueeze(0)
      feat = F.interpolate(feat, scale_factor=1/self._resampling_factor, mode='linear', align_corners=False)  # [1, F, Tz]
      feat = feat.squeeze(0).t()

      # prepend features to latents
      seq = torch.cat([feat, z], dim=-1)

      stride = max(1, int(self._sequence_length * self._stride_factor))
      n_chunks = (seq.shape[0] - self._sequence_length) // stride + 1

      for i in range(n_chunks):
        windows.append(seq[i*stride : i*stride + self._sequence_length])

    X = torch.cat(windows, dim=0) if len(windows) and windows[0].dim()==2 else torch.stack(windows, dim=0)  # not used; just for stats
    mean = X.mean(dim=0) if X.ndim==2 else X.mean(dim=(0,1))
    var  = X.var(dim=0, unbiased=False) if X.ndim==2 else X.var(dim=(0,1), unbiased=False)
    minv = X.min(dim=0).values if X.ndim==2 else X.amin(dim=(0,1))
    maxv = X.max(dim=0).values if X.ndim==2 else X.amax(dim=(0,1))
    stats = {'mean': mean, 'var': var, 'min': minv, 'max': maxv}

    return windows, stats



# from torch.utils.data import Dataset
# from torch import jit
# import torch
# import torch.nn.functional as F

# import librosa as li
# from glob import glob
# import os
# from typing import Tuple, List, Dict, Optional

# torch.set_default_dtype(torch.float32)

# from ddsp.feature_extractors import LibrosaFeatureExtractor

# class PriorDataset(Dataset):
#   def __init__(self,
#                encoding_model_path: str,
#                audio_dataset_path: str,
#                sequence_length: int,
#                sampling_rate: int = 44100,
#                stride_factor: float = 1.0,
#                device: Optional[str] = None,):
#     """
#     Arguments:
#       - encoding_model_path: str, the path to the encoding model
#       - audio_dataset_path: str, the path to the dataset
#       - sequence_length: int, the length of the preceding latent code sequence, in frames
#       - sampling_rate: int, the sampling rate of the audio
#       - stride_factor: float, the stride factor for the sequence (0.25 means 25% overlap)
#       - device: str, the device to use. None will use the originally saved device. [None, 'cuda', 'cpu'].
#       - extractors: dict[str, extractor], mapping names -> callable(audio_tensor)->Tensor
#                     Each extractor should accept a 1D torch.Tensor [T] (on any device)
#                     and return a Tensor shaped [1, n_feat, T_feat] or [1, T_feat] or [T_feat]
#                     (we’ll coerce shapes and time-align).
#     """
#     self._device = device
#     self._sequence_length = sequence_length
#     self._sampling_rate = sampling_rate
#     self._stride_factor = stride_factor

#     # feature extractors (can be set/updated later as well)
#     self._extractors = {
#       'loudness': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_LOUDNESS, resampling_factor=1, postprocess=True),
#       'spectral_centroid': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_CENTROID, resampling_factor=1, postprocess=True),
#       # 'spectral_flatness': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_FLATNESS, resampling_factor=1, postprocess=True),
#       # 'spectral_bandwidth': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_BANDWIDTH, resampling_factor=1, postprocess=True),
#     }

#     vae_model = jit.load(encoding_model_path, map_location=device).pretrained.to(device)
#     vae_model.encoder.streaming = False
#     self._vae = vae_model
#     self._resampling_factor = vae_model._resampling_factor  # frames hop in samples (assumed)

#     audio_tensors = self._load_audio_dataset(audio_dataset_path)
#     encodings = self._encode_audio_dataset(audio_tensors)
#     self._encodings, self.normalization_dict = self._normalize(encodings)

#   def __len__(self) -> int:
#     return len(self._encodings)

#   def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
#     """
#     Returns:
#       - a window tensor of shape [sequence_length, n_latents + n_features]
#     """
#     return self._encodings[idx]

#   # ---------- NEW: feature extraction + alignment ----------
#   @torch.no_grad()
#   def _extract_features_for_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor:
#     """
#     Run all extractors on a chunk of audio and align to the number of latent frames.
#     Returns a [target_frames, n_features] tensor (on self._device).
#     """
#     feats_list = []
#     for extractor in self._extractors.values():
#       feat = extractor(audio_chunk)  # arbitrary shape
#       feat = feat.to(self._device)

#       # -> [target_frames, n_feat]
#       feats_list.append(feat.squeeze(0).transpose(0, 1))  # [n_feat, T] -> [T, n_feat]

#     # Concatenate all features on the last dim: [T, sum(n_feat_i)]
#     features = torch.cat(feats_list, dim=-1)
#     return features

#   # ---------------------------------------------------------

#   def _normalize(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], dict]:
#     """
#     Collect stats; return x unchanged (current behavior).
#     """
#     all_x = torch.cat(x, dim=0)

#     with torch.no_grad():
#       mean = all_x.mean(dim=0).to(self._device)
#       minv = all_x.min(dim=0).values.to(self._device)
#       maxv = all_x.max(dim=0).values.to(self._device)
#       var = all_x.var(dim=0, unbiased=False).to(self._device)

#     normalization_dict = {'mean': mean, 'var': var, 'min': minv, 'max': maxv}
#     return x, normalization_dict

#   @torch.no_grad()
#   def _encode_audio_dataset(self, audio_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
#     """
#     Encode dataset to latents and prepend aligned features.
#     Returns a list of windows: each [sequence_length, n_features + n_latents]
#     """
#     print("Encoding audio dataset...")
#     encodings = []

#     for audio in audio_tensors:
#       # process in 40s chunks (as before)
#       for audio_chunk in audio.split(self._sampling_rate * 40):
#         if audio_chunk.numel() < self._resampling_factor:
#           continue

#         # [1, T] into encoder (expects [B, T])
#         mu, scale = self._vae.encoder(audio_chunk.unsqueeze(0))
#         latents, _ = self._vae.encoder.reparametrize(mu, scale)
#         latents = self._vae._smooth_latents(latents).squeeze(0)  # [T_lat, D_lat]

#         T_lat = latents.size(0)

#         # --- NEW: extract + align features for this chunk ---
#         features = self._extract_features_for_chunk(audio_chunk)  # [T_lat, D_feat]

#         # downsample features
#         features = F.interpolate(features.permute(0, 2, 1), scale_factor=1/self._resampling_factor, mode='linear').permute(0, 2, 1)

#         # prepend features to latents -> [T_lat, D_feat + D_lat]
#         latents_aug = torch.cat([features, latents], dim=-1)

#         # windowing with stride
#         stride = int(self._sequence_length * self._stride_factor)
#         if stride <= 0:
#           stride = 1
#         num_chunks = (T_lat - self._sequence_length) // stride + 1
#         for i in range(num_chunks):
#           window = latents_aug[i * stride : i * stride + self._sequence_length]  # [L, D_feat+D_lat]
#           encodings.append(window)

#     return encodings

#   def _load_audio_dataset(self, path: str) -> List[torch.Tensor]:
#     audio_tensors = []
#     for filepath in glob(os.path.join(path, '**', '*.wav'), recursive=True):
#       x = self._load_audio_file(filepath)
#       audio = torch.from_numpy(x).to(self._device)
#       if audio.size(0) >= self._resampling_factor:
#         audio_tensors.append(audio)
#     return audio_tensors

#   def _load_audio_file(self, path: str):
#     audio, _ = li.load(path, sr=self._sampling_rate, mono=True)
#     return audio

# # from torch.utils.data import Dataset
# # from torch import jit
# # import torch

# # import librosa as li
# # from glob import glob
# # import os

# # from typing import Tuple, List

# # torch.set_default_dtype(torch.float32)

# # from ddsp.feature_extractors import LibrosaFeatureExtractor

# # class PriorDataset(Dataset):
# #   def __init__(self,
# #                encoding_model_path: str,
# #                audio_dataset_path: str,
# #                sequence_length: int,
# #                sampling_rate: int = 44100,
# #                stride_factor: float = 1,
# #                device: str = None):
# #     """
# #     Arguments:
# #       - encoding_model_path: str, the path to the encoding model
# #       - audio_dataset_path: str, the path to the dataset
# #       - sequence_length: int, the length of the preceding latent code sequence, in samples
# #       - sampling_rate: int, the sampling rate of the audio
# #       - stride_factor: float, the stride factor for the sequence (0.25 means 25% overlap)
# #       - device: str, the device to use. None will use the originally saved device. [None, 'cuda', 'cpu'].
# #     """
# #     self._device = device
# #     self._sequence_length = sequence_length
# #     self._sampling_rate = sampling_rate
# #     self._stride_factor = stride_factor

# #     self._extractors = {
# #       'loudness': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_LOUDNESS, resampling_factor=1, postprocess=True),
# #       'spectral_centroid': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_CENTROID, resampling_factor=1, postprocess=True),
# #       # 'spectral_flatness': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_FLATNESS, resampling_factor=1, postprocess=True),
# #       # 'spectral_bandwidth': LibrosaFeatureExtractor(LibrosaFeatureExtractor.FN_SPECTRAL_BANDWIDTH, resampling_factor=1, postprocess=True),
# #     }


# #     vae_model = jit.load(encoding_model_path, map_location=device).pretrained.to(device)
# #     vae_model.encoder.streaming = False
# #     self._vae = vae_model
# #     self._resampling_factor = vae_model._resampling_factor

# #     # self._encoder = vae_model.encoder
# #     # self._encoder.streaming = False

# #     audio_tensors = self._load_audio_dataset(audio_dataset_path)

# #     encodings = self._encode_audio_dataset(audio_tensors)
# #     self._encodings, self.normalization_dict = self._normalize(encodings)


# #   def __len__(self) -> int:
# #     return len(self._encodings)


# #   def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
# #     """
# #     Get the latent code for a given index.
# #     Arguments:
# #       - idx: int, the index of the audio
# #     Returns:
# #       - x: torch.Tensor[n_frames, n_latents], the preceding latent code sequence
# #       - y: torch.Tensor[n_latents], the target latent code
# #     """
# #     return self._encodings[idx]


# #   def _normalize(self, x: List[torch.Tensor]) -> torch.Tensor:
# #     """
# #     Normalize the latent codes.
# #     Arguments:
# #       - x: List[torch.Tensor], the sequences of latent codes
# #     Returns:
# #       - x: torch.Tensor, the normalized latent codes
# #     """
# #     all_x = torch.cat(x, dim = 0)

# #     with torch.no_grad():
# #       mean = all_x.mean(dim=0).to(self._device)
# #       min = all_x.min(dim=0).values.to(self._device)
# #       max = all_x.max(dim=0).values.to(self._device)
# #       var = all_x.var(dim=0).to(self._device)

# #     normalization_dict = {'mean': mean, 'var': var, 'min': min, 'max': max}

# #     # normalized = [torch.from_numpy((item.cpu().numpy() - mean) / var, device=self._device) for item in x]
# #     # normalized = x

# #     return x, normalization_dict


# #   def _encode_audio_dataset(self, audio_tensors: List[torch.Tensor]):
# #     """
# #     Encode the entire audio dataset into latent codes.
# #     Arguments:
# #       - audio_tensors: List[torch.Tensor], the audio tensors
# #     Returns:
# #       - encodings: Dict[int, torch.Tensor], the encoded latent codes
# #     """
# #     print("Encoding audio dataset...")
# #     encodings = []

# #     for audio in audio_tensors:
# #       for audio_chunk in audio.split(self._sampling_rate*40):
# #         with torch.no_grad():
# #           mu, scale = self._vae.encoder(audio_chunk.unsqueeze(0))
# #           latents, _ = self._vae.encoder.reparametrize(mu, scale)
# #           latents = self._vae._smooth_latents(latents)
# #           latents = latents.squeeze(0)
# #           # mu_scale = torch.cat([mu, scale], dim = -1).squeeze(0)

# #           # mu_scale = mu.squeeze(0) # try only mu
# #           # mu_scale = mu_scale[..., :1] # try only one (the first) latent variable

# #           # Overlapping, shifting window chunks
# #           # for i in range(mu_scale.size(0) - (self._sequence_length)):
# #           #   encodings.append(mu_scale[i:i+self._sequence_length])

# #           # # Non-overlapping chunks
# #           # for chunk in mu_scale.split(self._sequence_length):
# #           #   if chunk.size(0) == self._sequence_length:
# #           #     encodings.append(chunk)

# #           # Stratified sampling with stride
# #           stride = int(self._sequence_length * self._stride_factor)
# #           num_chunks = (latents.size(0) - self._sequence_length) // stride + 1
# #           for i in range(num_chunks):
# #             encodings.append(latents[i*stride:i*stride+self._sequence_length])

# #     return encodings


# #   def _load_audio_dataset(self, path: str) -> List[torch.Tensor]:
# #     """
# #     Load and concat entire dataset into single tensor.

# #     Arguments:
# #       - path: str, path to the dataset
# #     Returns:
# #       - audio: torch.Tensor, the audio tensor
# #     """
# #     audio_tensors = []
# #     for filepath in glob(os.path.join(path, '**', '*.wav'), recursive=True):
# #       x = self._load_audio_file(filepath)
# #       audio = torch.from_numpy(x).to(self._device)
# #       if audio.size(0) >= self._resampling_factor:
# #         audio_tensors.append(audio)

# #     return audio_tensors


# #   def _load_audio_file(self, path: str):
# #     """
# #     Load an audio file from a path.
# #     """
# #     audio, _ = li.load(path, sr = self._sampling_rate, mono = True)
# #     return audio



# # import torch
# # from torch.utils.data import Dataset

# # class DummyMultivariateSequenceDataset(Dataset):
# #   def __init__(self, num_features, seq_length, n_examples):
# #     """
# #     Args:
# #         num_features (int): Number of features in each data point.
# #         seq_length (int): Length of each sequence.
# #         n_examples (int): Number of examples in the dataset.
# #     """
# #     self.num_features = num_features
# #     self.seq_length = seq_length
# #     self.n_examples = n_examples

# #     # Generate the synthetic data
# #     self.data = self._generate_data()

# #   def _generate_data(self):
# #     # Generate random data for features
# #     data = torch.randn(self.n_examples, self.seq_length + 1, self.num_features)

# #     # Ensure that the data is easy to predict: make the target (next step) a linear combination of the input
# #     for i in range(self.seq_length):
# #         data[:, i + 1, :] = data[:, i, :] * 0.5 + torch.randn(self.n_examples, self.num_features) * 0.1

# #     return data

# #   def __len__(self):
# #     return self.n_examples

# #   def __getitem__(self, idx):
# #     return self.data[idx, :self.seq_length +1, :]
