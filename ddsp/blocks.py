import torch
import torch.nn as nn
import torch.nn.functional as F
import cached_conv as cc
import math
from torchaudio.transforms import MFCC, MelSpectrogram
import torch.jit as jit

from typing import Tuple, List

def _make_mlp(in_size: int, hidden_layers: int, hidden_size: int) -> cc.CachedSequential:
  """
  Constructs a multi-layer perceptron.
  Args:
  - in_size: int, the input layer size
  - hidden_layers: int, the number of hidden layers
  - hidden_size: int, the size of each hidden layer
  Returns:
  - mlp: cc.CachedSequential, the multi-layer perceptron
  """
  sizes = [in_size]
  sizes.extend(hidden_layers * [hidden_size])

  return _make_sequential(sizes)


def _make_sequential(sizes: List[int]):
  """
  Constructs a sequential model.
  Args:
  - sizes: List[int], the sizes of the layers
  Returns:
  - mlp: cc.CachedSequential, the sequential model
  """
  layers = []
  for i in range(len(sizes)-1):
    layers.append(nn.Linear(sizes[i], sizes[i+1]))
    layers.append(nn.LayerNorm(sizes[i+1]))
    layers.append(nn.LeakyReLU())

  return nn.Sequential(*layers)

def _scaled_sigmoid(x: torch.Tensor):
  """
  Custom activation function for the output layer. It is a scaled sigmoid function,
  guaranteeing that the output is always positive.
  Args:
    - x: torch.Tensor, the input tensor
  Returns:
    - y: torch.Tensor, the output tensor
  """
  return 2*torch.pow(torch.sigmoid(x), math.log(10)) + 1e-18

def _is_batch_size_one(x: torch.Tensor):
  """
  Check if the batch size of a tensor is one.
  Args:
    - x: torch.Tensor, the input tensor
  Returns:
    - bool, True if the batch size is one, False otherwise
  """
  return x.shape[0] == 1


class WaveformEncoder(nn.Module):
  """
  Variational encoder that operates directly on raw waveforms using
  strided 1D convolutions for progressive downsampling.
  """
  def __init__(self,
               sample_rate: int = 44100,
               layer_sizes: List[int] = [128, 64, 32],
               latent_size: int = 16,
               resampling_factor: int = 32,
               hidden_channels: List[int] = [32, 64, 128],
               strides: List[int] = [4, 4, 2],
               kernel_multiplier: int = 4,
               streaming: bool = False):
    """
    Arguments:
      - sample_rate: int, the sample rate of the input audio
      - layer_sizes: List[int], the sizes of the layers in the bottleneck (after conv)
      - latent_size: int, the size of the output latent space
      - resampling_factor: int, total downsampling factor (product of strides)
      - hidden_channels: List[int], channel sizes for each conv layer
      - strides: List[int], stride for each conv layer (must multiply to resampling_factor)
      - kernel_multiplier: int, kernel_size = stride * kernel_multiplier
      - streaming: bool, streaming mode (realtime)
    """
    super().__init__()

    self.streaming = streaming
    self.resampling_factor = resampling_factor
    self.sample_rate = sample_rate

    # Validate strides
    stride_product = 1
    for s in strides:
      stride_product *= s
    if stride_product != resampling_factor:
      raise ValueError(
        f"Product of strides {strides} = {stride_product} must equal "
        f"resampling_factor = {resampling_factor}"
      )

    # Build strided convolutional layers
    conv_layers = []
    in_ch = 1  # mono audio input
    for i, (out_ch, stride) in enumerate(zip(hidden_channels, strides)):
      kernel_size = stride * kernel_multiplier
      # Use cached_conv for streaming compatibility
      padding = cc.get_padding(kernel_size, stride)
      conv_layers.append(
        cc.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
      )
      conv_layers.append(nn.GroupNorm(min(8, out_ch), out_ch))
      conv_layers.append(nn.LeakyReLU(0.2))
      in_ch = out_ch

    self.conv_encoder = nn.Sequential(*conv_layers)
    self.conv_out_channels = hidden_channels[-1]

    # Normalization before GRU
    self.normalization = nn.LayerNorm(self.conv_out_channels)

    # GRU for temporal modeling
    self.gru = nn.GRU(self.conv_out_channels, layer_sizes[0], batch_first=True)
    self.register_buffer('_hidden_state', torch.zeros(1, 1, layer_sizes[0]), persistent=False)

    # Bottleneck MLP
    self.bottleneck = _make_sequential(layer_sizes)

    # Output projection to mu and logvar
    self.mu_logvar_out = nn.Linear(layer_sizes[-1], 2 * latent_size)

  def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass of the encoder.
    Arguments:
      - audio: torch.Tensor, the input audio tensor [batch_size, n_samples]
    Returns:
      - mu, logvar: Tuple[torch.Tensor, torch.Tensor], each [batch_size, T_ctl, latent_size]
    """
    # Add channel dimension: [B, T] -> [B, 1, T]
    x = audio.unsqueeze(1)

    # Strided convolutions: [B, 1, T] -> [B, C, T_ctl]
    x = self.conv_encoder(x)

    # Permute to [B, T_ctl, C] for GRU
    x = x.permute(0, 2, 1)

    # Normalize
    x = self.normalization(x)

    # GRU with streaming support
    if self.streaming and _is_batch_size_one(x):
      x, hx = self.gru(x, self._hidden_state)
      self._hidden_state.copy_(hx)
    else:
      x, _ = self.gru(x)

    # Bottleneck MLP
    x = self.bottleneck(x)

    # Project to mu and logvar
    z = self.mu_logvar_out(x)
    mu, logvar = z.chunk(2, dim=-1)

    return mu, logvar

  def reparametrize(self, mean: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reparametrize the latent variable z.
    Args:
      - mean: torch.Tensor[batch_size, n_latents, latent_size], the mean
      - scale: torch.Tensor[batch_size, n_latents, latent_size], the scale (pre-softplus)
    Returns:
      - z: torch.Tensor[batch_size, n_latents, latent_size], the reparametrized latent variable
      - kl: torch.Tensor[1], the KL divergence
    """
    std = F.softplus(scale) + 1e-4
    var = std * std
    logvar = torch.log(var)

    z = torch.randn_like(mean) * std + mean

    # Calculate KL divergence
    kl_weight = 1.0 / (z.shape[1] * z.shape[2])
    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return z, kl * kl_weight


class VariationalEncoder(nn.Module):
  def __init__(self,
               sample_rate: int = 44100,
               layer_sizes: List[int] = [128, 64, 32],
               latent_size: int = 16,
               resampling_factor: int = 32,
               n_melbands: int = 128,
               streaming: bool = False):
    """
    Arguments:
      - sample_rate: int, the sample rate of the input audio
      - layer_sizes: List[int], the sizes of the layers in the bottleneck
      - latent_size: int, the size of the output latent space
      - resampling_factor: int, the factor by which to downsample the mfccs
      - n_mfcc : int, the number of mfccs to extract
      - streaming: bool, streaming mode (realtime)
    """
    super().__init__()

    self.streaming = streaming

    self.resampling_factor = resampling_factor
    # self.mfcc = MFCC(sample_rate = sample_rate, n_mfcc = n_mfcc)
    self.melspec = MelSpectrogram(sample_rate, n_mels=n_melbands)

    self.normalization = nn.LayerNorm(n_melbands)

    self.gru = nn.GRU(n_melbands, layer_sizes[0], batch_first = True)
    self.register_buffer('_hidden_state', torch.zeros(1, 1, layer_sizes[0]), persistent=False)

    self.bottleneck = _make_sequential(layer_sizes)

    self.mu_logvar_out = nn.Linear(layer_sizes[-1], 2*latent_size)


  def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass of the encoder.
    Arguments:
      - audio: torch.Tensor, the input audio tensor [batch_size, n_samples]
    Returns:
      - mu, logvar: Tuple[torch.Tensor, torch.Tensor], the latent space tensor
    """
    # Calculate Mel spectrogram
    melspec = self.melspec(audio)

    # Expand the Mel spectrogram to match the audio length
    melspec = F.interpolate(melspec, size = audio.shape[-1], mode = 'nearest')

    # Downsample the input representation
    x = F.interpolate(melspec, scale_factor = 1/self.resampling_factor, mode = 'linear')

    # Reshape to [batch_size, signal_length, n_melbands]
    x = x.permute(0, 2, 1)

    # Normalize the input
    x = self.normalization(x)

    # Pass through the GRU layer
    if self.streaming and _is_batch_size_one(x):
      x, hx = self.gru(x, self._hidden_state)
      self._hidden_state.copy_(hx)
    else:
      x, _ = self.gru(x)

    # Pass through bottleneck
    x = self.bottleneck(x)

    # Pass through the dense layer
    z = self.mu_logvar_out(x)

    mu, logvar = z.chunk(2, dim = -1)

    return mu, logvar


  def reparametrize(self, mean: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reparametrize the latent variable z.
    Args:
      - z: torch.Tensor[batch_size, n_latents, latent_size], the latent variable
    Returns:
      - z: torch.Tensor[batch_size, n_latents, latent_size], the reparametrized latent variable
      - kl: torch.Tensor[1], the KL divergence
    """
    std = F.softplus(scale) + 1e-4
    var = std * std
    logvar = torch.log(var)

    z = torch.randn_like(mean) * std + mean

    # Calculate KL divergence
    kl_weight = 1.0 / (z.shape[1] * z.shape[2])  # KL weight for averaging
    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return z, kl*kl_weight

  # def reparametrize(self, mu, logvar):
  #   sigma = torch.sqrt(torch.exp(logvar))
  #   eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size()).to(mu.device) # perche' lo devo mandare a device?
  #   z = mu + sigma * eps
  #   return z, torch.tensor([1])



class Decoder(nn.Module):
  def __init__(self,
               n_params: int = 500,
               latent_size: int = 16,
               n_features: int = 4,
               layer_sizes: List[int] = [32, 64, 128],
               output_mlp_layers: int = 3,
               gru_layers: int = 1,
               streaming: bool = False):
    """
    Arguments:
      - n_params: int, the number of synthesis parameters
      - latent_size: int, the size of the latent space
      - layer_sizes: List[int], the sizes of the layers in the bottleneck
      - output_mlp_layers: int, the number of layers in the output MLP
      - gru_layers: int, the number of GRU layers in the decoder
      - streaming: bool, streaming mode (realtime)
    """
    super().__init__()

    self.n_params = n_params
    self.streaming = streaming

    # MLP mapping from the latent space
    self.input_latent_bottleneck = _make_sequential([latent_size] + layer_sizes)

    # MLP mapping from the input features
    self.input_features_bottleneck = _make_sequential([n_features] + layer_sizes)

    # transformed latents + transformed features + original features
    self.hidden_size = layer_sizes[-1] * 2 + n_features

    # Intermediate GRU layer
    self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=gru_layers, batch_first=True)
    self.register_buffer('_hidden_state', torch.zeros(gru_layers, 1, self.hidden_size), persistent=False)

    # Intermediary 3-layer MLP
    self.inter_mlp = _make_mlp(self.hidden_size, output_mlp_layers, self.hidden_size)

    # Output layer predicting noiseband amplitudes, and sine frequencies and amplitudes
    self.output_params = nn.Linear(self.hidden_size, n_params)


  def forward(self, features: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the decoder.
    Arguments:
      - features: torch.Tensor, the input features tensor [batch_size, n_features, n_signal]
      - z: torch.Tensor, the latent space tensor [batch_size, latent_size, n_signal]
    Returns:
      - synth_params: torch.Tensor, the predicted synthesiser parameters
    """
    # Pass latents through the input MLP
    z_transformed = self.input_latent_bottleneck(z)

    # Pass features through the input MLP
    features_transformed = self.input_features_bottleneck(features)

    # Concatenate the transformed latents, transformed features and features
    x = torch.cat((z_transformed, features_transformed, features), dim=-1)

    # Pass through the GRU layer
    if self.streaming and _is_batch_size_one(z):
      x, hx = self.gru(x, self._hidden_state)
      self._hidden_state.copy_(hx)
    else:
      x, _ = self.gru(x)

    # Pass through the intermediary MLP
    x = self.inter_mlp(x)

    # Pass through the output layer
    output = _scaled_sigmoid(self.output_params(x))

    return output.permute(0, 2, 1)


  def positional_encoding(self, d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model, requires_grad=False)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
