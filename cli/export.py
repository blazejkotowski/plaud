import nn_tilde
import argparse
import torch
import torch.nn.functional as F
import lightning as L
import cached_conv as cc

import time

from ddsp.utils import find_checkpoint

from ddsp import DDSP
from ddsp.prior import Prior

torch.enable_grad(False)
torch.set_printoptions(threshold=10000)

class ScriptedDDSP(nn_tilde.Module):
  def __init__(self,
               pretrained: DDSP,
               prior_model: Prior = None,
               target_fs: float = 16000.0):
    super().__init__()

    self.pretrained = pretrained

    if prior_model is None:
      prior_model = FakePrior()
    else:
      prior_model = PriorWrapper(prior_model)

    self.prior_model = prior_model


    self.resample_ratio = target_fs / self.pretrained.fs
    print("resample ratio:", self.resample_ratio )
    # # # Calculate the input ratio
    # x_len = 2**14
    # x = torch.zeros(1, 1, x_len) for _ in range(self.pretrained.n_control_params)
    # y, _ = self.pretrained(x)
    # in_ratio = y.shape[-1] / x_len
    # print(f"in_ratio: {in_ratio}")

    # self.register_buffer("prior_buffer", torch.randn(1, self.prior_model._max_len, self.prior_model._num_params))

    self.register_attribute("limit_components", 0.0)
    self.register_attribute("noise_amplitude_attenuation", 0.0)
    self.register_attribute("sines_amplitude_attenuation", 0.0)
    self.register_attribute("waveshaping", 0.0)

    # self.register_method(
    #   "forward",
    #   in_channels = 1,
    #   in_ratio = 1,
    #   out_channels = 1,
    #   out_ratio = 1,
    #   input_labels=['(signal) Audio Input'],
    #   output_labels=['(signal) Audio Output'],
    #   test_method=True,
    # )

    total_params = self.pretrained.num_params + self.pretrained.n_features

    self.register_method(
      "decode",
      in_channels = total_params,
      in_ratio = int(self.pretrained.resampling_factor * self.resample_ratio),
      out_channels = 1, # number of output audio channels
      # out_ratio = 1/3,
      out_ratio = 1,
      input_labels=[f'(signal) Latent Dimension {i}' for i in range(1, total_params+1)],
      output_labels=['(signal) Audio Output'],
      test_method=True,
    )

    self.register_method(
      "encode",
      in_channels = 1,
      in_ratio = 1,
      out_channels = self.pretrained.num_params,
      out_ratio = self.pretrained.resampling_factor,
      input_labels=['(signal) Audio Input'],
      output_labels=[f'(signal) Latent Dimension {i}' for i in range(1, self.pretrained.num_params+1)],
      test_method=True
    )

    if not isinstance(self.prior_model, FakePrior):
      self.register_method(
        "prior",
        in_channels=total_params + 2, # latent transposition + temperature + prediction_strength
        in_ratio=self.pretrained.resampling_factor,
        out_channels=total_params,
        out_ratio=self.pretrained.resampling_factor,
        input_labels=[f'(signal) Transposition {i}' for i in range(1, total_params+1)] + ['(signal) Temperature', '(signal) Prediction strenght'],
      )

  @torch.jit.export
  def decode(self, params: torch.Tensor):
    # params = params.permute(0, 2, 1)
    # print('params', params.shape)
    # print(self.pretrained.latent_size)
    # print(self.pretrained.num_params)
    latents = params[:, self.pretrained.n_features:, :]
    features = params[:, :self.pretrained.n_features, :]
    # print("Params shape:", params.shape)

    latents = latents.permute(0, 2, 1)
    # latents = self.pretrained.params_to_latents(latents)
    # latents = self.pretrained.denormalize_latents(latents)
    features = features.permute(0, 2, 1)

    synth_params = self.pretrained.decoder(features, latents)
    audio = self.pretrained._synthesize(synth_params, waveshaping_factor=self.waveshaping[0], limit_components=self.limit_components[0])
    # print("Audio shape before interpolation:", audio.shape)

    if self.resample_ratio != 1:
      audio = F.interpolate(audio, scale_factor=self.resample_ratio, mode='linear')

    # print("Audio shape after interpolation:", audio.shape)

    return audio.float()


  @torch.jit.export
  def encode(self, audio: torch.Tensor):
    # if self.resampler is not None:
      #  audio = self.resampler.to_model_sampling_rate(audio)

    mu, scale = self.pretrained.encoder(audio.squeeze(1))
    # latents = self.pretrained.encoder.reparametrize(mu, logvar)
    latents, _ = self.pretrained.encoder.reparametrize(mu, scale)
    latents = self.pretrained._smooth_latents(latents)
    latents = self.pretrained.normalize_latents(latents)
    latents = self.pretrained.latents_to_params(latents)
    latents = latents.permute(0, 2, 1).float()

    return latents

  # @torch.jit.export
  # def forward(self, audio: torch.Tensor):
  #   return self.pretrained(audio.squeeze(1)).float()

  @torch.jit.export
  def prior(self, x: torch.Tensor):
    return self.prior_model(x)

  @torch.jit.export
  def get_waveshaping(self) -> float:
    return self.waveshaping[0]

  @torch.jit.export
  def set_waveshaping(self, value: float):
    self.waveshaping = (value, )
    return 0


  @torch.jit.export
  def get_limit_components(self) -> float:
    return self.limit_components[0]

  @torch.jit.export
  def set_limit_components(self, value: float):
    self.limit_components = (value, )
    return 0

  @torch.jit.export
  def get_noise_amplitude_attenuation(self) -> float:
    return self.noise_amplitude_attenuation[0]

  @torch.jit.export
  def set_noise_amplitude_attenuation(self, value: float):
    self.noise_amplitude_attenuation = (value, )
    return 0

  @torch.jit.export
  def get_sines_amplitude_attenuation(self) -> float:
    return self.sines_amplitude_attenuation[0]

  @torch.jit.export
  def set_sines_amplitude_attenuation(self, value: float):
    self.sines_amplitude_attenuation = (value, )
    return 0


class FakePrior(torch.nn.Module):
  def forward(self, x: torch.Tensor):
    return torch.zeros_like(x)


class PriorWrapper(torch.nn.Module):
  def __init__(self, prior: Prior):
    super().__init__()

    self.prior = prior

    self.max_len = self.prior._max_len
    self.init_primer_len = self.max_len // 4
    self.current_buffer_len = self.init_primer_len
    self.register_buffer("prior_buffer", torch.randn(1, self.max_len, self.prior._num_params))
    # self.prior_buffer = torch.randn(1, self.max_len, self.prior._num_params)


  def append_to_buffer(self, x: torch.Tensor):
    """
    Appends the input tensor to the prior buffer, if the buffer is full,
    it is reset to the initial primer length.

    Args:
      x, torch.Tensor[batch_size, seq_len, num_params]
    """
    x = x[:1, ...] # only first in batch
    seq_len = x.shape[1]

    if self.current_buffer_len + seq_len > self.max_len:
      self.prior_buffer[:, :self.init_primer_len-seq_len, :] = self.prior_buffer[:, -self.init_primer_len+seq_len:, :].clone()
      self.prior_buffer[:, self.init_primer_len:self.init_primer_len+seq_len, :] = x
      self.current_buffer_len = self.init_primer_len
    else:
      self.prior_buffer[:, self.current_buffer_len:self.current_buffer_len+seq_len, :] = x
      self.current_buffer_len += seq_len


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x, torch.Tensor[batch_size, num_params + 1, seq_len]
    """
    # self.append_to_buffer(x.permute(0, 2, 1))

    # Ignore the batch dimension
    transposition = x[:1, :-2, :]
    temperature = x[:1, -2, :]
    prediction_annealing = 1 - x[:1, -1, :]

    steps = x.shape[-1]

    output = torch.zeros(1, steps, self.prior._num_params)
    local_buffer = self.prior_buffer.clone()
    current_len = self.current_buffer_len

    for i in range(steps):
      prime = local_buffer[:, :current_len, :]
      logits = self.prior(prime)
      latent = self.prior.sample(logits, temperature=temperature[0, i])[:, -1:, :]

      # transpose
      # print('latent', latent.shape, 'transpositions', transposition[:, :, i].shape)
      latent *= prediction_annealing[:, i]
      latent += transposition[:, :, i]

      local_buffer[:, current_len:current_len+1, :] = latent
      output[:, i, :] = latent[:, 0, :]

      current_len += 1

      if current_len >= self.max_len:
        local_buffer[:, :self.init_primer_len, :] = local_buffer[:, -self.init_primer_len:, :].clone()
        current_len = self.init_primer_len

    if x.size(0) > 1:
      output = output.repeat_interleave(x.size(0), dim=0)

    self.append_to_buffer(output)

    return output.permute(0, 2, 1).float()

class ONNXDDSP(torch.nn.Module):
  def __init__(self,
               pretrained: DDSP):
    super().__init__()

    self.pretrained = pretrained

  def decode(self, latents: torch.Tensor):
    synth_params = self.pretrained.decoder(latents.permute(0, 2, 1))
    audio = self.pretrained._synthesize(synth_params)
    return audio

  def encode(self, audio: torch.Tensor):
    mu, scale = self.pretrained.encoder(audio.squeeze(1))
    latents, _ = self.pretrained.encoder.reparametrize(mu, scale)
    return latents.permute(0, 2, 1)

  def forward(self, audio: torch.Tensor):
    return self.pretrained(audio.squeeze(1))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_directory', type=str, help='Path to the model training')
  parser.add_argument('--prior_directory', type=str, default=None, help='Path to the prior model training')
  parser.add_argument('--output_path', type=str, help='Directory to save the autoencoded audio')
  parser.add_argument('--streaming', type=bool, default=True, help='Whether to use streaming mode')
  parser.add_argument('--type', default='best', help='Type of model to export', choices=['best', 'last'])
  parser.add_argument('--target_fs', type=float, default=16000.0, help='Target sampling rate for the exported model')
  config = parser.parse_args()

  cc.use_cached_conv(config.streaming)

  checkpoint_path = find_checkpoint(config.model_directory, typ=config.type)
  print(f"exporting model from checkpoint: {checkpoint_path}")

  format = config.output_path.split('.')[-1]
  if format not in ['ts', 'onnx']:
    raise ValueError(f'Invalid format: {format}, supported formats are: ts, onnx')

  prior = None
  if config.prior_directory is not None:
    prior_checkpoint_path = find_checkpoint(config.prior_directory, typ=config.type)
    print("exporting prior model from checkpoint: ", prior_checkpoint_path)
    prior = Prior.load_from_checkpoint(prior_checkpoint_path, strict=False).to('cpu')
    prior.eval()
    for k in prior._normalization_dict.keys():
      prior._normalization_dict[k] = prior._normalization_dict[k].to('cpu')

    prior._trainer = L.Trainer()

  ddsp = DDSP.load_from_checkpoint(checkpoint_path, strict=False, streaming=True, device='cpu').to('cpu')
  ddsp.streaming(True)
  if format == 'onnx':
    ddsp.eval()
    scripted = ONNXDDSP(ddsp).to('cpu')
    torch.onnx.dynamo_export(
      scripted,
      torch.zeros(1, 1, 2**14),
    ).save(config.output_path)
  elif format == 'ts':
    # ugly workaround for the torchscript
    ddsp._trainer = L.Trainer()
    ddsp._recons_loss = None
    ddsp._mr_stft_loss = None
    ddsp._mr_mel_loss = None
    ddsp._mr_chroma_loss = None
    ddsp._m2l_loss = None
    ddsp._sliced_wasserstein_loss = None

    ddsp.eval()


    scripted = ScriptedDDSP(ddsp, prior, config.target_fs).to('cpu')
    scripted.export_to_ts(config.output_path)

    print("Model exported to: ", config.output_path)
