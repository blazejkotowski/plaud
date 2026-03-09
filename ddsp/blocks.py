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
               streaming: bool = False,
               raw_param_ranges: List[tuple] = None,
               orth_embed_dim: int = 0,
               stft_n_fft: int = 0,
               stft_embed_dim: int = 128,
               freq_bin_selection: bool = False,
               freq_n_fft: int = 4096,
               params_per_sine: int = 4):
    """
    Arguments:
      - n_params: int, the number of synthesis parameters
      - latent_size: int, the size of the latent space
      - layer_sizes: List[int], the sizes of the layers in the bottleneck
      - output_mlp_layers: int, the number of layers in the output MLP
      - gru_layers: int, the number of GRU layers in the decoder
      - streaming: bool, streaming mode (realtime)
      - raw_param_ranges: list of (start, end) tuples for param indices that
        should skip _scaled_sigmoid (synths with raw_output=True handle their own activation)
      - orth_embed_dim: int, when > 0, enables a shortcut head for raw-output
        synths. Maps encoder latent z directly to synth params via single Linear,
        bypassing the deep decoder that kills frequency gradients in sinusoidal
        synthesis with MSE loss. The value is used as a flag (any > 0 enables it).
      - stft_n_fft: int, when > 0, computes an STFT of the input audio and feeds
        compressed spectral features into per-sine MLPs for direct frequency hints.
      - stft_embed_dim: int, dimensionality of the compressed STFT features.
      - freq_bin_selection: bool, when True, per-sine MLPs predict frequency via
        classification over STFT bins instead of direct regression. This turns the
        intractable continuous frequency optimization into discrete bin selection
        with smooth cross-entropy gradients.
      - freq_n_fft: int, FFT size for frequency bins (4096 → 10.8 Hz resolution).
    """
    super().__init__()

    self.n_params = n_params
    self.streaming = streaming
    self._raw_param_ranges = raw_param_ranges or []
    self._orth_embed_dim = orth_embed_dim
    self._stft_n_fft = stft_n_fft
    self._stft_embed_dim = stft_embed_dim
    self._freq_bin_selection = freq_bin_selection
    self._freq_n_fft = freq_n_fft
    self._pps = params_per_sine  # params per sine (4 for ComplexSine, 2 for SimpleSine)

    # Register frequency bin centers (omega in rad/sample) for bin selection.
    # Only positive frequencies [0, π]: bins 0..n_fft/2
    if freq_bin_selection:
      n_freq_bins = freq_n_fft // 2 + 1
      bin_omega = torch.arange(n_freq_bins, dtype=torch.float32) * (2 * math.pi / freq_n_fft)
      self.register_buffer('_bin_omega', bin_omega)  # [n_freq_bins]
      self._n_freq_bins = n_freq_bins
      # Learnable temperature for softmax sharpness (start at 1.0, can anneal)
      self._freq_temperature = nn.Parameter(torch.tensor(1.0))
    else:
      self._n_freq_bins = 0

    # STFT spectral hint for per-sine MLPs: computes STFT of input audio and
    # compresses the magnitude spectrum to a low-dim embedding. This gives the
    # per-sine MLPs direct access to spectral peaks, enabling Hz-level frequency
    # accuracy that a generic learned encoder z alone cannot provide.
    self._stft_compressor = None
    if stft_n_fft > 0:
      n_bins = stft_n_fft // 2 + 1
      self._stft_compressor = nn.Sequential(
        nn.Linear(n_bins, stft_embed_dim),
        nn.LeakyReLU(0.2),
      )

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

    # Per-sine independent MLPs for raw-output synths (e.g. ComplexSineSynth).
    # Each sine gets its own small MLP: z → hidden → 4 params (omega, phi, amp_start, amp_end).
    # No weight sharing between sines eliminates gradient interference that kills
    # the deep decoder (MLP→GRU→MLP) when sign-normalized gradients cancel out.
    # Formula: raw_params = per_sine_mlp(z) + sigmoid(alpha) * decoder_output
    # With alpha=-5 (sigmoid≈0.007), the per-sine MLPs dominate early in training.
    self._n_sines = 0
    self._params_per_sine = 0
    if orth_embed_dim > 0 and self._raw_param_ranges:
      n_raw_params = sum(end - start for start, end in self._raw_param_ranges)
      # Infer n_sines and params_per_sine from raw_param_ranges
      self._params_per_sine = self._pps
      self._n_sines = n_raw_params // self._params_per_sine
      assert self._n_sines * self._params_per_sine == n_raw_params, \
        f"n_raw_params={n_raw_params} not divisible by params_per_sine={self._params_per_sine}"

      # Per-sine independent weights: no shared hidden layers between sines
      # Input includes z (latent_size) + optional STFT features (stft_embed_dim)
      sine_hidden = 64
      sine_input_size = latent_size + (stft_embed_dim if stft_n_fft > 0 else 0)
      self._sine_input_size = sine_input_size

      if freq_bin_selection:
        # Bin selection mode: per-sine MLPs output logits over frequency bins
        # + (params_per_sine - 1) other params + 1 residual offset.
        # Architecture: input → hidden → two heads:
        #   - freq_head: hidden → n_freq_bins (logits for bin classification)
        #   - other_head: hidden → params_per_sine (residual_offset + other params)
        # For ComplexSine (pps=4): residual_offset, phi, amp_s, amp_e
        # For SimpleSine (pps=2): residual_offset, amp
        n_other = self._params_per_sine  # 1 residual_offset + (pps-1) other params
        self._sine_w1 = nn.Parameter(torch.randn(self._n_sines, sine_input_size, sine_hidden) * (2.0 / sine_input_size)**0.5)
        self._sine_b1 = nn.Parameter(torch.zeros(self._n_sines, 1, sine_hidden))
        # Frequency head: [n_sines, hidden, n_freq_bins]
        self._sine_freq_w = nn.Parameter(torch.randn(self._n_sines, sine_hidden, n_freq_bins) * (2.0 / sine_hidden)**0.5)
        self._sine_freq_b = nn.Parameter(torch.zeros(self._n_sines, 1, n_freq_bins))
        # Other params head: [n_sines, hidden, n_other]
        self._sine_other_w = nn.Parameter(torch.randn(self._n_sines, sine_hidden, n_other) * (2.0 / sine_hidden)**0.5)
        self._sine_other_b = nn.Parameter(torch.zeros(self._n_sines, 1, n_other))
      else:
        # Original mode: per-sine MLPs output 4 params directly
        # Layer 1: [n_sines, input_size, hidden] + bias [n_sines, hidden]
        # Layer 2: [n_sines, hidden, params_per_sine] + bias [n_sines, params_per_sine]
        self._sine_w1 = nn.Parameter(torch.randn(self._n_sines, sine_input_size, sine_hidden) * (2.0 / sine_input_size)**0.5)
        self._sine_b1 = nn.Parameter(torch.zeros(self._n_sines, 1, sine_hidden))
        self._sine_w2 = nn.Parameter(torch.randn(self._n_sines, sine_hidden, self._params_per_sine) * (2.0 / sine_hidden)**0.5)
        self._sine_b2 = nn.Parameter(torch.zeros(self._n_sines, 1, self._params_per_sine))
      # Blend factor: sigmoid(-5) ≈ 0.007, so per-sine MLPs dominate early
      self._residual_alpha = nn.Parameter(torch.tensor(-5.0))


  def forward(self, features: torch.Tensor, z: torch.Tensor,
              audio: torch.Tensor = None, hop_size: int = 1024) -> torch.Tensor:
    """
    Forward pass of the decoder.
    Arguments:
      - features: torch.Tensor, the input features tensor [batch_size, n_features, n_signal]
      - z: torch.Tensor, the latent space tensor [batch_size, latent_size, n_signal]
      - audio: torch.Tensor (optional), raw audio [batch_size, n_samples] for STFT hint
      - hop_size: int, hop size for STFT computation (should match resampling_factor)
    Returns:
      - synth_params: torch.Tensor, the predicted synthesiser parameters [B, n_params, T]
      When freq_bin_selection is enabled, also stores self._last_freq_logits
      [B, n_sines, T, n_freq_bins] for cross-entropy loss computation.
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
    raw = self.output_params(x)

    # Apply _scaled_sigmoid selectively: skip for raw-output synth params
    if self._raw_param_ranges:
      # Build output without in-place ops for autograd compatibility
      sigmoid_output = _scaled_sigmoid(raw)
      parts = []
      prev_end = 0
      for start, end in self._raw_param_ranges:
        if start > prev_end:
          parts.append(sigmoid_output[:, :, prev_end:start])
        parts.append(raw[:, :, start:end])
        prev_end = end
      if prev_end < raw.shape[-1]:
        parts.append(sigmoid_output[:, :, prev_end:])
      output = torch.cat(parts, dim=-1)
    else:
      output = _scaled_sigmoid(raw)

    # Per-sine MLP prediction for raw-output params.
    # Each sine has its own MLP, so sign-normalized gradients don't interfere.
    if self._orth_embed_dim > 0 and self._raw_param_ranges and self._n_sines > 0:
      # Build per-sine MLP input: [z] or [z; stft_features]
      sine_input = z
      stft_mag_raw = None  # raw STFT magnitude for bin selection logit bias
      if self._stft_compressor is not None and audio is not None:
        with torch.no_grad():
          stft = torch.stft(audio, n_fft=self._stft_n_fft, hop_length=hop_size,
                            window=torch.hann_window(self._stft_n_fft, device=audio.device),
                            return_complex=True)
          stft_mag_raw = stft.abs()  # [B, n_bins, T_stft]
        stft_mag_perm = stft_mag_raw.permute(0, 2, 1)  # [B, T_stft, n_bins]
        T = min(z.shape[1], stft_mag_perm.shape[1])
        stft_mag_perm = stft_mag_perm[:, :T, :]
        z_for_sine = z[:, :T, :]
        stft_embed = self._stft_compressor(stft_mag_perm)  # [B, T, stft_embed_dim]
        sine_input = torch.cat([z_for_sine, stft_embed], dim=-1)
      elif self._stft_compressor is not None:
        zero_stft = torch.zeros(z.shape[0], z.shape[1], self._stft_embed_dim, device=z.device, dtype=z.dtype)
        sine_input = torch.cat([z, zero_stft], dim=-1)

      # Shared layer 1: input → hidden
      # sine_input: [B, T, sine_input_size]
      h = torch.einsum('btl,slh->bsth', sine_input, self._sine_w1) + self._sine_b1.unsqueeze(0)
      h = F.leaky_relu(h)
      # h: [B, n_sines, T, hidden]

      if self._freq_bin_selection:
        # === Bin selection mode ===
        # Frequency head: logits over bins
        freq_logits = torch.einsum('bsth,shf->bstf', h, self._sine_freq_w) + self._sine_freq_b.unsqueeze(0)
        # freq_logits: [B, n_sines, T, n_freq_bins]

        # Use STFT magnitude as logit bias when available — peaks in the spectrum
        # naturally guide bin selection toward actual frequencies in the audio.
        if stft_mag_raw is not None and self._stft_n_fft == self._freq_n_fft:
          # stft_mag_raw: [B, n_bins, T_stft] → [B, 1, T, n_bins]
          T_logit = freq_logits.shape[2]
          stft_bias = torch.log(stft_mag_raw[:, :, :T_logit] + 1e-8)  # [B, n_bins, T]
          stft_bias = stft_bias.permute(0, 2, 1).unsqueeze(1)  # [B, 1, T, n_bins]
          freq_logits = freq_logits + stft_bias

        # Store logits for CE loss (before temperature scaling)
        self._last_freq_logits = freq_logits

        # Soft bin selection: softmax → weighted average of bin frequencies
        temp = F.softplus(self._freq_temperature) + 0.01  # ensure > 0
        attn = F.softmax(freq_logits / temp, dim=-1)  # [B, n_sines, T, n_freq_bins]
        omega = (attn * self._bin_omega).sum(dim=-1)  # [B, n_sines, T]
        # Detach omega_base so audio-domain loss gradients only flow through
        # the sub-bin residual_offset (below), not through the bin selection
        # logits. CE loss operates directly on freq_logits and is unaffected.
        # Without this, sign-normalized synth gradients destroy CE convergence.
        omega = omega.detach()

        # Other params head: [B, n_sines, T, pps] where pps = params_per_sine
        # Layout: [residual_offset, ...other_params]
        # For pps=4: [residual_offset, phi, amp_s, amp_e]
        # For pps=2: [residual_offset, amp]
        other_raw = torch.einsum('bsth,shp->bstp', h, self._sine_other_w) + self._sine_other_b.unsqueeze(0)
        # other_raw: [B, n_sines, T, pps]
        residual_offset = other_raw[..., 0]  # [B, n_sines, T]
        other_params = [other_raw[..., i] for i in range(1, self._params_per_sine)]  # pps-1 other params

        # Sub-bin refinement: offset in [-0.5, 0.5] bins → rad/sample
        bin_width = 2 * math.pi / self._freq_n_fft
        omega = omega + torch.tanh(residual_offset) * 0.5 * bin_width

        # Assemble into param-grouped layout: [omega_all, other1_all, other2_all, ...]
        # For pps=4: [omega, phi, amp_s, amp_e]; for pps=2: [omega, amp]
        all_components = [omega] + other_params  # list of [B, n_sines, T]
        sine_grouped = torch.stack(all_components, dim=-1)  # [B, n_sines, T, pps]
        sine_grouped = sine_grouped.permute(0, 2, 3, 1)  # [B, T, pps, n_sines]
        B, T_out = sine_grouped.shape[0], sine_grouped.shape[1]
        ns = self._n_sines
        pps = self._params_per_sine
        sine_grouped = sine_grouped.reshape(B, T_out, pps * ns)  # [B, T, pps*n_sines]
      else:
        # === Original regression mode ===
        sine_raw = torch.einsum('bsth,shp->bstp', h, self._sine_w2) + self._sine_b2.unsqueeze(0)
        # sine_raw: [B, n_sines, T, params_per_sine]
        sine_raw = sine_raw.permute(0, 2, 1, 3).reshape(z.shape[0], z.shape[1], -1)  # [B, T, n_sines*4]
        B, T_out, _ = sine_raw.shape
        ns = self._n_sines
        pp = self._params_per_sine
        sine_grouped = sine_raw.view(B, T_out, ns, pp)  # [B, T, n_sines, 4]
        sine_grouped = sine_grouped.permute(0, 1, 3, 2).reshape(B, T_out, ns * pp)  # [B, T, 4*n_sines]
        self._last_freq_logits = None

      scale = torch.sigmoid(self._residual_alpha)
      # Build updated output without in-place ops
      parts = []
      prev_end = 0
      raw_offset = 0
      for start, end in self._raw_param_ranges:
        if start > prev_end:
          parts.append(output[:, :, prev_end:start])
        n = end - start
        sc_slice = sine_grouped[:, :T_out, raw_offset:raw_offset + n]  # [B, T, n]
        parts.append(sc_slice + scale * output[:, :T_out, start:end])
        raw_offset += n
        prev_end = end
      if prev_end < output.shape[-1]:
        parts.append(output[:, :T_out, prev_end:])
      output = torch.cat(parts, dim=-1)
    else:
      self._last_freq_logits = None

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
