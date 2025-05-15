import lightning as L
import torch

import numpy as np
import auraloss
import laion_clap
from music2latent import EncoderDecoder
from music2latent.audio import to_representation_encoder

from ddsp.blocks import VariationalEncoder, Decoder
from ddsp.synths import BaseSynth, SineSynth, NoiseBandSynth

from typing import List, Tuple, Dict, Any

class CLAPLoss(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.clap = laion_clap.CLAP_Module(enable_fusion=False, device='cuda')
    self.clap.requires_grad_(False)
    self.clap.load_ckpt()

  def forward(self, x: torch.tensor, y: torch.tensor):
    with torch.no_grad():
      x_emb = self.clap.get_audio_embedding_from_data(x.reshape(x.shape[0], -1).float(), use_tensor=True)
      y_emb = self.clap.get_audio_embedding_from_data(y.reshape(y.shape[0], -1).float(), use_tensor=True)

    return 1-torch.nn.functional.cosine_similarity(x_emb, y_emb).mean()

class M2LLoss(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.m2l = EncoderDecoder()
    self.m2l.gen.eval()

  def forward(self, x: torch.tensor, y: torch.tensor):
    # with torch.no_grad():
    x_repr = to_representation_encoder(x.detach().squeeze(1))
    x_emb = self.m2l.gen.encoder(x_repr, extract_features=True)

    y_repr = to_representation_encoder(y.squeeze(1))
    y_emb = self.m2l.gen.encoder(y_repr, extract_features=True)

    return torch.nn.functional.mse_loss(y_emb, x_emb).mean()
    # return 1-torch.nn.functional.cosine_similarity(x_emb, y_emb).mean()


class DDSP(L.LightningModule):
  """
  A neural network that learns how to resynthesise signal, predicting amplitudes of
  precalculated, loopable noise bands.

  Args:
    - synth_builders: List[Callable[[], BaseSynth]], the differentiable synthesiser builders
    - latent_size: int, number of latent dimensions
    - fs : int, the sampling rate of the input signal
    - encoder_ratios: List[int], the capacity ratios for encoder layers
    - n_melbands: int, the number of MFCCs to extract
    - decoder_ratios: List[int], the capacity ratios for decoder layers
    - capacity: int, the capacity of the model
    - resampling_factor: int, internal up / down sampling factor for control signal and noisebands
    - learning_rate: float, the learning rate for the optimizer
    - streaming: bool, whether to run the model in streaming mode
    - kld_weight: float, the weight for the KLD loss
    - device: str, the device to run the model on
  """
  def __init__(self,
               synth_configs: List[Dict[Any, Any]] = [],
               latent_size: int = 16,
               fs: int = 44100,
               encoder_ratios: List[int] = [8, 4, 2],
               n_melbands: int = 128,
               decoder_ratios: List[int] = [2, 4, 8],
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
    self.resampling_factor = resampling_factor

    # self.synths = synths
    # self.synths = torch.nn.ModuleList([builder() for builder in self._synth_builders])
    self.synths = torch.nn.ModuleList([
      BaseSynth.from_config(cfg) for cfg in self._synth_configs
    ])

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
      features=False
    )

    ## Decoder to predict the amplitudes of the noise bands
    self.decoder = Decoder(
      n_params=self._total_synth_params,
      latent_size=latent_size,
      layer_sizes=(np.array(decoder_ratios)*capacity).tolist(),
      streaming=streaming,
    )

    # Define the loss
    self._mr_stft_loss = self._construct_mrstft_loss() # MRSTFT loss
    self._mr_mel_loss = self._construct_mel_loss() # MEL-scale loss
    self._mr_chroma_loss = self._construct_chroma_loss() # CHROMA-scale loss

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


  def configure_optimizers(self):
    """Configure the optimizer for the model"""
    # return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=False, threshold=1e-2)

    scheduler = {
      'scheduler': lr_scheduler,
      'monitor': 'val_loss',
      'interval': 'epoch'
    }

    return [optimizer], [scheduler]


  def _synthesize(self, params: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - params: torch.Tensor[batch_size, params, sig_length], the predicted synth parameters
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
    params_idx = 0
    audio = []
    for synth in self.synths:
      synth_params = params[:, params_idx:synth.n_params, :]
      params_idx += synth.n_params
      audio.append(synth(synth_params))

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

    w_stft = 0.5
    w_mel = 1.0
    w_chroma = 0.1

    loss = (w_stft * self._mr_stft_loss(y, x) \
        + w_chroma * self._mr_chroma_loss(y, x) \
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
    fft_sizes = np.array([1024, 512, 128, 64, 32])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 512, 128, 64, 32],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes).to(self._device)

