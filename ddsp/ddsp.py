import lightning as L
import torch

import numpy as np
import auraloss

from ddsp.blocks import VariationalEncoder, Decoder
from ddsp.synths import BaseSynth, SineSynth, NoiseBandSynth
from sklearn.decomposition import PCA

from typing import List, Tuple, Dict, Any

from ddsp.losses import M2LLoss


class DDSP(L.LightningModule):
  """
  A neural network that learns how to resynthesise signal, predicting amplitudes of
  precalculated, loopable noise bands.

  Args:
    - synth_builders: List[Callable[[], BaseSynth]], the differentiable synthesiser builders
    - latent_size: int, number of latent dimensions
    - fs : int, the sampling rate of the input signal
    - encoder_ratios: List[int], the capacity ratios for encoder layers
    - latent_smoothing_kernel: int, the kernel size for the smoothing filter
    - num_params: int, the number of parameters to predict
    - n_melbands: int, the number of MFCCs to extract
    - decoder_ratios: List[int], the capacity ratios for decoder layers
    - capacity: int, the capacity of the model
    - resampling_factor: int, internal up / down sampling factor for control signal and noisebands
    - learning_rate: float, the learning rate for the optimizer
    - streaming: bool, whether to run the model in streaming mode
    - perceptual_loss_weight: float, the weight for the perceptual loss
    - kld_weight: float, the weight for the KLD loss
    - device: str, the device to run the model on
  """
  def __init__(self,
               synth_configs: List[Dict[Any, Any]] = [],
               latent_size: int = 16,
               num_params: int = 4,
               fs: int = 44100,
               encoder_ratios: List[int] = [8, 4, 2],
               latent_smoothing_kernel: int = 129,
               n_melbands: int = 128,
               decoder_ratios: List[int] = [2, 4, 8],
               decoder_gru_layers: int = 1,
               capacity: int = 64,
               resampling_factor: int = 32,
               learning_rate: float = 1e-3,
               kld_weight: float = 0.00025,
               perceptual_loss_weight: float = 1.0,
               streaming: bool = False,
               device: str = 'cuda'):
    super().__init__()
    # Save hyperparameters in the checkpoints
    self.save_hyperparameters()
    self._synth_configs = synth_configs
    self._device = device
    self.fs = fs
    self.latent_size = latent_size
    self.num_params = num_params
    self.resampling_factor = resampling_factor
    self._latent_smoothing_kernel = latent_smoothing_kernel

    # self.synths = synths
    # self.synths = torch.nn.ModuleList([builder() for builder in self._synth_builders])
    self.synths = torch.nn.ModuleList([
      BaseSynth.from_config(cfg).to(self._device) for cfg in self._synth_configs
    ])

    # Latent space analyzis buffers
    
    self.register_buffer('_latent_pca', torch.eye(self.latent_size))
    self.register_buffer('_latent_mean', torch.zeros(self.latent_size))
    self.register_buffer('_latent_quantiles', torch.zeros(2, self.latent_size))

    self._total_synth_params = sum([s.n_params for s in self.synths])
    # self.register_buffer('_total_synth_params', torch.tensor(total_params, dtype=torch.long))
    # self._total_synth_params = sum([s.n_params for s in self.synths])
    # total_synth_params = 1000

    # self.synths = [NoiseBandSynth(n_filters=1000, fs=fs, resampling_factor=resampling_factor)]

    # ELBO regularization params
    self._beta = 0
    self._kld_weight = kld_weight

    # Define the neural network
    ## Encoder to extract latents from the input audio signal
    self.encoder = VariationalEncoder(
      layer_sizes=(np.array(encoder_ratios)*capacity).tolist(),
      sample_rate=fs,
      latent_size=latent_size,
      streaming=streaming,
      n_melbands=n_melbands,
      resampling_factor=self.resampling_factor,
    )

    ## Decoder to predict the amplitudes of the noise bands
    self.decoder = Decoder(
      n_params=self._total_synth_params,
      latent_size=latent_size,
      layer_sizes=(np.array(decoder_ratios)*capacity).tolist(),
      gru_layers=decoder_gru_layers,
      streaming=streaming,
    )

    # Define the loss
    self._mr_stft_loss = self._construct_mrstft_loss() # MRSTFT loss
    self._mr_mel_loss = self._construct_mel_loss() # MEL-scale loss
    # self._mr_chroma_loss = self._construct_chroma_loss() # CHROMA-scale loss

    # Perceptual loss
    self._m2l_loss = M2LLoss()
    self._perceptual_loss_weight = perceptual_loss_weight
    # self._recons_loss = CLAPLoss() # CLAP loss

    # Learning rate
    self._learning_rate = learning_rate

    # Validation inputs and outputs
    self._last_validation_in = None
    self._last_validation_out = None
    self._validation_index = 1


  def streaming(self, streaming: bool) -> None:
    """Set streaming mode"""
    self.streaming = streaming
    for synth in self.synths:
      synth.streaming = streaming

    self.decoder.streaming = streaming


  def analyze_latent_space(self, z: torch.Tensor) -> None:
    """
    Saves the PCA decomposition and the mean of the latent space

    Args:
      z: torch.Tensor[n, latent_size]
    """
    # Extract and store the mean
    latent_mean = z.mean(0)
    self._latent_mean.copy_(latent_mean)

    # Center the latents
    z -= latent_mean

    # Fit and strore the PCA
    latent_pca = PCA(n_components=self.latent_size)
    latent_pca.fit(z.cpu().numpy())
    self._latent_pca.copy_(torch.from_numpy(latent_pca.components_))

    # Compute 5th and 95th percentile for each component
    quantiles = torch.quantile(z, torch.tensor([0.01, 0.99]), dim=0)
    self._latent_quantiles.copy_(quantiles)


  def normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the given latents using the saved mean and quantils

    Args:
      z: torch.Tensor[batch_size, n, latent_size]
    Returns:
      z: torch.Tensor[batch_size, n, latent_size], normalized latents
    """
    # Center the latents
    z_centered = z - self._latent_mean

    # Normalize the latents using the quantiles to range [-1, 1]
    quantiles = self._latent_quantiles
    z_normalized = 2*(z_centered - quantiles[0]) / (quantiles[1] - quantiles[0])-1

    # Ensure the latents are in the range [-1, 1]
    z_normalized = torch.clamp(z_normalized, -1.0, 1.0)

    return z_normalized


  def denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes the given latents using the saved mean and quantils

    Args:
      z: torch.Tensor[batch_size, n, latent_size]
    Returns:
      z: torch.Tensor[batch_size, n, latent_size], denormalized latents
    """
    # Ensure the latents are in the range [-1, 1]
    z_clamped = torch.clamp(z, -1.0, 1.0)

    # Denormalize the latents using the quantiles
    quantiles = self._latent_quantiles

    # From [-1, 1] to [quantiles[0], quantiles[1]]
    z_denormalized = (z_clamped + 1) / 2 * (quantiles[1] - quantiles[0]) + quantiles[0]

    # Add the mean back
    z_denormalized += self._latent_mean

    return z_denormalized



  def latents_to_params(self, z: torch.Tensor) -> torch.Tensor:
    """
    Converts the given latents to params using the saved PCA and mean

    Args:
      - z: torch.Tensor[batch_size, n, latent_size]
    Returns:
      - params: torch.Tensor[batch_size, n, num_params]
    """
    # Do not do anything if no projection is necessary
    if z.shape[-1] == self.num_params:
      return z

    z_centered = z - self._latent_mean
    pca = self._latent_pca[:self.num_params, :]
    params = torch.matmul(pca, z_centered.permute(0, 2, 1))
    return params.permute(0, 2, 1)


  def params_to_latents(self, params: torch.Tensor) -> torch.Tensor:
    """
    Converts given params to the latent space using the saved PCA and mean.
    It fills the missing information with gaussian noise with the right mean.

    Args:
      - params: torch.Tensor[batch_size, n, num_params]
    Returns:
      - latents: torch.Tensor[batch_size, n, latent_size]
    """
    # Do not do anything if no projection is necessary
    if params.shape[-1] == self.latent_size:
      return params

    # params: [batch_size, n, num_params]
    batch_size, n, num_params = params.shape
    latent_size = self.latent_size

    # Get PCA and mean
    pca = self._latent_pca[:num_params, :]  # [num_params, latent_size]
    latent_mean = self._latent_mean  # [latent_size]

    # Invert PCA: x = pca @ z.T  =>  z = pca_pinv @ x.T
    pca_pinv = torch.linalg.pinv(pca)  # [latent_size, num_params]
    params_t = params.permute(0, 2, 1)  # [batch_size, num_params, n]
    z_centered = torch.matmul(pca_pinv, params_t)  # [batch_size, latent_size, n]
    z_centered = z_centered.permute(0, 2, 1)  # [batch_size, n, latent_size]

    # Fill missing latent dims with noise if needed
    if num_params < latent_size:
      noise = torch.randn(batch_size, n, latent_size - num_params, device=params.device)
      z_centered[..., num_params:] = noise

    # Add mean back
    latents = z_centered + latent_mean
    return latents



  def forward(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the network.
    Args:
      - audio: torch.Tensor[batch_size, n_signal], the input audio signal
    Returns:
      - signal: torch.Tensor, the synthesized signal
    """
    mu, scale = self.encoder(audio)

    # Reparametrization trick
    z, _ = self.encoder.reparametrize(mu, scale)

    # smooth the latent envelopes
    z = self._smooth_latents(z)

    # Predict the parameters of the synthesiser and synthesize
    synth_params = self.decoder(z)
    signal = self._synthesize(synth_params)
    return signal


  def training_step(self, x_audio: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """
    Compute the loss for a batch of data

    Args:
      batch:
        Tuple[
            torch.Tensor[batch_size, n_signal],
            torch.Tensor[params_number, batch_size, n_signal]
          ], audio, control_params
      batch_idx: int, index of the batch (unused)
    Returns:
      loss: torch.Tensor[batch_size, 1], tensor of loss
    """
    _, losses = self._autoencode(x_audio)

    self.log("recons_loss", losses["recons_loss"], prog_bar=True, logger=True)
    self.log("kld_loss", losses["kld_loss"], prog_bar=True, logger=True)
    self.log("m2l_loss", losses["m2l_loss"], prog_bar=True, logger=True)
    self.log("train_loss", losses["loss"], prog_bar=True, logger=True)
    self.log("beta", self._beta, prog_bar=True, logger=True)
    self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

    return losses["loss"]


  def validation_step(self, x_audio: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """Compute the loss for validation data"""
    with torch.no_grad():
      y_audio, losses = self._autoencode(x_audio)

    loss = losses["recons_loss"]

    self.log("val_loss", loss, prog_bar=True, logger=True)

    # if self._last_validation_in is None:
    #   self._last_validation_in = x_audio
    #   self._last_validation_out = y_audio.squeeze(1)

    return y_audio


  def configure_optimizers(self):
    """Configure the optimizer for the model"""
    # return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, verbose=False, threshold=1e-3)

    scheduler = {
      'scheduler': lr_scheduler,
      'monitor': 'val_loss',
      'interval': 'epoch'
    }

    return [optimizer], [scheduler]


  def _smooth_latents(self, z: torch.Tensor) -> torch.Tensor:
    """
    smooth the latent envelopes using a low-pass filter

    Args:
      - z: torch.Tensor[batch_size, n_latent, latent_size], the latent envelopes
    Returns:
      - z: torch.Tensor[batch_size, n_latent, latent_size], the smoothed latent envelopes
    """
    if self._latent_smoothing_kernel == 1:
      return z

    # smooth the latent envelopes, individually
    # Apply a simple moving average (low-pass filter) along the time axis for each latent dimension
    padding = self._latent_smoothing_kernel // 2
    z_smooth = torch.nn.functional.avg_pool1d(
      z.transpose(1, 2), kernel_size=self._latent_smoothing_kernel, stride=1, padding=padding
    ).transpose(1, 2)
    return z_smooth


  def _autoencode(self, x_audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Autoencode the audio signal
    Args:
      - x_audio: torch.Tensor[batch_size, n_signal], the input audio signal
    Returns:
      - y_audio: torch.Tensor[batch_size, n_signal], the autoencoded audio signal
      - losses: Dict[str, torch.Tensor], the losses computed during the autoencoding
    """
    # Encode the audio signal
    mu, scale = self.encoder(x_audio)

    # Reparametrization trick
    z, kld_loss = self.encoder.reparametrize(mu, scale)

    # smooth the latent envelopes
    z = self._smooth_latents(z)

    # Predict the parameters of the synthesiser
    synth_params = self.decoder(z)

    # Synthesize the output signal
    y_audio = self._synthesize(synth_params)

    # Compute the reconstruction loss
    recons_loss = self._reconstruction_loss(y_audio.float(), x_audio.float())

    # Compute the total loss using β parameter
    loss = recons_loss + self._kld_weight * self._beta * kld_loss

    m2l_loss = 0
    # Compute the perceptual loss with music2latent
    if self._perceptual_loss_weight > 0.0:
      m2l_loss = self._m2l_loss(y_audio, x_audio)
      loss += self._perceptual_loss_weight * m2l_loss
    # loss = m2l_loss

    # Construct losses dictionary
    losses = {
      "recons_loss": recons_loss,
      "kld_loss": kld_loss,
      "m2l_loss": m2l_loss,
      "loss": loss
    }

    return y_audio, losses


  # def on_validation_epoch_end(self):
  #   """At the end of the validation epoch, log the validation audio"""
  #   if self._last_validation_out is not None:
  #     device = self._last_validation_out.device
  #   else:
  #     device = self.device

  #   # audio = torch.FloatTensor(0).to(device) # Concatenated audio
  #   # silence = torch.zeros(1, int(self.fs/2)).to(device) # 0.5s silence
  #   # for input, output in zip(self._last_validation_in, self._last_validation_out):
  #   #   audio = torch.cat((audio, input.unsqueeze(0), silence, output.unsqueeze(0), silence.repeat(1, 3)), dim=-1)

  #   # audio = audio.clip_(-1, 1) # Clip the audio to stay in range
  #   # self.logger.experiment.add_audio("audio_validation", audio, self._validation_index, self.fs)

  #   self._last_validation_in = None
  #   self._last_validation_out = None
  #   self._validation_index += 1


  def _synthesize(self, params: torch.Tensor, sines_number_attenuation: float = 0.0, noise_amp_attenuation: float = 0.0, sines_amp_attenuation: float = 0.0) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - params: torch.Tensor[batch_size, params, sig_length], the predicted synth parameters
      - sines_number_attenuation: float, attenuation factor for the number of sines
      - noise_amp_attenuation: float, attenuation factor for the noise amplitude
      - sines_amp_attenuation: float, attenuation factor for the sines amplitude
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
    params_idx = 0
    audio = []

    # Clamping to [0, 1]
    noise_amp = max(0.0, min(1.0 - noise_amp_attenuation, 1.0))
    sines_amp = max(0.0, min(1.0 - sines_amp_attenuation, 1.0))

    for synth in self.synths:
      synth_params = params[:, params_idx:params_idx+synth.n_params, :]
      params_idx += synth.n_params
      if isinstance(synth, SineSynth):
        audio.append(synth(synth_params, sines_number_attenuation=sines_number_attenuation)*sines_amp)
      elif isinstance(synth, NoiseBandSynth):
        audio.append(synth(synth_params)*noise_amp)

      # audio.append(synth(synth_params))

    return torch.sum(torch.hstack(audio), dim=1, keepdim=True)


  def _reconstruction_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the reconstruction loss using weighted sum of perceptually
    weighted multi-resolution chroma STFT and mel STFT with a regular MR-STFT"""
    if y.dim() == 2:
      y = y.unsqueeze(1)

    if x.shape[-1] != y.shape[-1]:
      # Fit the signals to the same length
      min_length = min(x.shape[-1], y.shape[-1])
      x = x[..., :min_length]
      y = y[..., :min_length]

    w_stft = 0.7
    w_mel = 0.3
    w_chroma = 0.1

    loss = (w_stft * self._mr_stft_loss(y, x) \
        # + w_chroma * self._mr_chroma_loss(y, x) \
        + w_mel * self._mr_mel_loss(y, x) \
      ) / (w_stft + w_mel + w_chroma)
    return loss

  @torch.jit.ignore
  def _construct_mel_loss(self):
    """Construct the loss function for the model: a multi-resolution STFT loss"""
    fft_sizes = np.array([32768, 16384, 8192, 4096, 2048])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[32768, 16384, 8192, 4096, 2048],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes,
                                                scale='mel',
                                                n_bins=256,
                                                sample_rate=self.fs,
                                                perceptual_weighting=True).to(self._device)

  @torch.jit.ignore
  def _construct_chroma_loss(self):
    fft_sizes = np.array([16384, 8192, 4096])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[16384, 8192, 4096],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes,
                                                scale='chroma',
                                                n_bins=128,
                                                sample_rate=self.fs,
                                                perceptual_weighting=True).to(self._device)

  @torch.jit.ignore
  def _construct_mrstft_loss(self):
    """Construct the loss function for the model: a multi-resolution STFT loss"""
    fft_sizes = np.array([1024, 512, 128])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 512, 128],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes).to(self._device)

