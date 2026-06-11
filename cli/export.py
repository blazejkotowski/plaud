import nn_tilde
import argparse
import os
import math
import torch
import torch.nn.functional as F
import lightning as L
import cached_conv as cc
from typing import Optional

import time

from ddsp.utils import find_checkpoint

from ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace
import torch
from ddsp.prior import Prior, PriorDiscrete
from ddsp.latent_compressor import LatentCompressor

torch.enable_grad(False)
torch.set_printoptions(threshold=10000)

class ScriptedDDSP(nn_tilde.Module):
  def __init__(self,
               pretrained: DDSP,
               prior_model: torch.nn.Module = None,
               target_fs: float = 16000.0):
    super().__init__()

    self.pretrained = pretrained

    self.resample_ratio = target_fs / self.pretrained.fs

    numerator = self.pretrained.resampling_factor * target_fs
    denominator = int(self.pretrained.fs)

    if numerator % denominator != 0:
      gcd = math.gcd(self.pretrained.resampling_factor, denominator)
      compatible_step = denominator // gcd
      raise ValueError(
        "target_fs={} is incompatible with nn~ export. "
        "Choose a multiple of {} to keep decode buffers aligned.".format(target_fs, compatible_step)
      )

    self._nn_decode_ratio = int(numerator // denominator)

    if prior_model is None:
      prior_model = FakePrior()
    elif isinstance(prior_model, Prior):
      prior_model = PriorWrapper(prior_model, resample_ratio=self.resample_ratio)
    # else: assume it's already a wrapper module compatible with ScriptedDDSP.prior()

    self.prior_model = prior_model


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

    n_channels = self.pretrained.n_channels

    self.register_method(
      "decode",
      in_channels = total_params,
      in_ratio = self._nn_decode_ratio,
      out_channels = n_channels, # number of output audio channels
      # out_ratio = 1/3,
      out_ratio = 1,
      input_labels=[f'(signal) Latent Dimension {i}' for i in range(1, total_params+1)],
      output_labels=[f'(signal) Audio Output {i}' for i in range(1, n_channels+1)],
      test_method=True,
    )

    if self.pretrained.latent_size > 0:
      self.register_method(
        "encode",
        in_channels = n_channels,
        in_ratio = 1,
        out_channels = self.pretrained.num_params,
        out_ratio = self.pretrained.resampling_factor,
        input_labels=[f'(signal) Audio Input {i}' for i in range(1, n_channels+1)],
        output_labels=[f'(signal) Latent Dimension {i}' for i in range(1, self.pretrained.num_params+1)],
        test_method=True
      )

    if not isinstance(self.prior_model, FakePrior):
      self.register_method(
        "prior",
        in_channels=total_params + 2, # latent transposition + temperature + prediction_strength
        in_ratio=self._nn_decode_ratio,
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

    if self.pretrained.latent_size == 0:
      latents = torch.zeros(latents.size(0), latents.size(1), 1, device=latents.device)

    synth_params = self.pretrained.decoder(features, latents)
    audio = self.pretrained._synthesize(synth_params, waveshaping_factor=self.waveshaping[0], limit_components=self.limit_components[0])
    # print("Audio shape before interpolation:", audio.shape)

    if self.resample_ratio != 1:
      audio = F.interpolate(audio, scale_factor=self.resample_ratio, mode='linear')

    # print("Audio shape after interpolation:", audio.shape)

    return audio.float()


  @torch.jit.export
  def encode(self, audio: torch.Tensor):
    if self.pretrained.encoder is None:
      raise RuntimeError("encode() requested but the pretrained model has no encoder (latent_size=0)")
    # Encoder downmixes the [B, n_channels, T] input internally
    mu, scale = self.pretrained.encoder(audio)
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
  def __init__(self, prior: Prior, resample_ratio: float = 1.0):
    super().__init__()

    self.prior = prior
    self.resample_ratio = resample_ratio

    self.max_len = self.prior._max_len
    self.init_primer_len = self.max_len // 4
    self.current_buffer_len = self.init_primer_len
    self.register_buffer("prior_buffer", torch.zeros(1, self.max_len, self.prior._num_controls))
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

    # TODO: Is this correct?
    if self.current_buffer_len + seq_len > self.max_len:
      if seq_len >= self.init_primer_len:
        # If seq_len is greater than or equal to init_primer_len, just keep the last init_primer_len elements from x
        self.prior_buffer[:, :self.init_primer_len, :] = x[:, -self.init_primer_len:, :]
      else:
        # Shift the last init_primer_len - seq_len elements to the beginning
        self.prior_buffer[:, :self.init_primer_len - seq_len, :] = self.prior_buffer[:, -self.init_primer_len + seq_len:, :].clone()
        # Place x at the end of the primer region
        self.prior_buffer[:, self.init_primer_len - seq_len:self.init_primer_len, :] = x
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

    output = torch.zeros(1, steps, self.prior._num_controls)
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

    if self.resample_ratio != 1:
      output = F.interpolate(output.permute(0, 2, 1), scale_factor=self.resample_ratio, mode='linear').permute(0, 2, 1)

    return output.permute(0, 2, 1).float()


class LatentCompressorDecodeOnly(torch.nn.Module):
  def __init__(self, vq: torch.nn.Module, decoder: torch.nn.Module, compression_ratio: int):
    super().__init__()
    self.vq = vq
    self.decoder = decoder
    self.compression_ratio = int(compression_ratio)

  def decode_codes(self, indices: torch.Tensor, output_len: Optional[int] = None) -> torch.Tensor:
    z_q = self.vq.embed(indices)
    x_hat = self.decoder(z_q, None)

    if output_len is not None:
      T_out = int(x_hat.shape[1])
      if T_out > output_len:
        x_hat = x_hat[:, :output_len, :]
      elif T_out < output_len:
        x_hat = F.pad(x_hat, (0, 0, 0, output_len - T_out))

    return x_hat

  def forward(self, indices: torch.Tensor, output_len: Optional[int] = None) -> torch.Tensor:
    return self.decode_codes(indices, output_len=output_len)


class PriorDiscreteWrapper(torch.nn.Module):
  def __init__(self, prior: PriorDiscrete, compressor: torch.nn.Module, resample_ratio: float = 1.0):
    super().__init__()

    self.prior = prior
    self.compressor = compressor
    self.resample_ratio = resample_ratio

    self.max_len = int(self.prior._max_len)
    self.init_primer_len = int(self.max_len // 4)
    self.num_codebooks = int(self.prior.num_codebooks)
    self.codebook_size = int(self.prior.codebook_size)
    # START token id (prior-only; must never be fed to the compressor's VQ).
    self.start_id = int(getattr(self.prior, 'start_token_id', self.codebook_size))
    self.compression_ratio = int(getattr(self.compressor, 'compression_ratio', 32))

    # Infer control dimension via a dummy decode.
    with torch.no_grad():
      _dummy = torch.zeros(1, 1, self.num_codebooks, dtype=torch.long)
      _num_controls = int(compressor.decode_codes(_dummy).shape[2])

    # Cold start from the learned START token at position 0, length 1.
    _init_buf = torch.zeros(1, self.max_len, self.num_codebooks, dtype=torch.long)
    _init_buf[:, 0, :] = self.start_id
    self.register_buffer("token_buffer", _init_buf)
    # Scalar state as 0-d long tensors so TorchScript reliably persists mutations across calls.
    self.register_buffer("_current_len", torch.tensor(1, dtype=torch.long))
    # One token's worth of decoded control frames; _ctrl_pos is the next frame to emit.
    # Initialised with _ctrl_pos == compression_ratio to signal "empty" on first call.
    self.register_buffer("_ctrl_buf", torch.zeros(1, self.compression_ratio, _num_controls))
    self.register_buffer("_ctrl_pos", torch.tensor(self.compression_ratio, dtype=torch.long))

    # The conv decoder needs a few tokens of right-context or the emitted frames are
    # boundary-corrupted. We decode with a small lookahead and emit a lagging token
    # (=> ~decode_lookahead tokens of latency), which makes the streaming decode match
    # a full-sequence decode exactly.
    self.decode_ctx = 8
    self.decode_lookahead = 2
    # Index into token_buffer of the next real token to emit (1 == first token after START).
    self.register_buffer("_emit_idx", torch.tensor(1, dtype=torch.long))

  @torch.jit.export
  def reset_state(self):
    # Clean cold start: only the START token in the buffer, nothing decoded yet.
    self.token_buffer.zero_()
    self.token_buffer[:, 0, :] = self.start_id
    self._current_len.fill_(1)
    self._emit_idx.fill_(1)
    self._ctrl_buf.zero_()
    self._ctrl_pos.fill_(self.compression_ratio)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, total_params + 2, steps]
    transposition = x[:1, :-2, :]
    temperature   = x[:1, -2, :]
    prediction_annealing = 1 - x[:1, -1, :]

    steps = int(x.shape[-1])
    if steps <= 0:
      return x[:1, :-2, :].float()

    local_buf   = self.token_buffer.clone()
    current_len = int(self._current_len.item())   # tokens generated so far (incl START)
    emit_idx    = int(self._emit_idx.item())      # buffer index of the next token to emit
    ctrl_buf    = self._ctrl_buf.clone()          # [1, CR, D]
    ctrl_pos    = int(self._ctrl_pos.item())      # next frame to emit

    num_controls = int(ctrl_buf.shape[2])
    out = torch.zeros(1, steps, num_controls)
    frames_written = 0

    CR = self.compression_ratio
    # Emit exactly `steps` control frames, generating new tokens on demand.
    for _iter in range(steps + CR):
      if frames_written >= steps:
        break

      if ctrl_pos >= CR:
        # Generate ahead so emit_idx has `decode_lookahead` tokens of right-context.
        while current_len < emit_idx + 1 + self.decode_lookahead:
          ti   = int(min(steps - 1, frames_written))
          temp = torch.clamp(temperature[:, ti:ti+1], min=1e-4)

          prime      = local_buf[:, :current_len, :]
          logits     = self.prior(prime)
          next_logits = logits[:, -1, :, :]
          probs      = torch.softmax(next_logits / temp.unsqueeze(-1), dim=-1)
          samples    = torch.multinomial(probs.reshape(-1, self.codebook_size), 1).reshape(1, self.num_codebooks)

          if current_len >= self.max_len:
            shift = current_len - self.init_primer_len
            local_buf[:, :self.init_primer_len, :] = local_buf[:, shift:current_len, :].clone()
            current_len = self.init_primer_len
            emit_idx = emit_idx - shift

          local_buf[:, current_len:current_len+1, :] = samples.unsqueeze(1)
          current_len += 1

        # Decode emit_idx with `decode_lookahead` tokens of right-context, then keep
        # exactly that token's block. Clamp so a leading START token never indexes the VQ.
        lo = max(0, emit_idx - self.decode_ctx)
        hi = emit_idx + 1 + self.decode_lookahead
        decode_win = local_buf[:, lo:hi, :].clamp(min=0, max=self.codebook_size - 1)
        decoded    = self.compressor.decode_codes(decode_win)
        off        = emit_idx - lo
        ctrl_buf   = decoded[:, off * CR:(off + 1) * CR, :].detach()
        ctrl_pos   = 0
        emit_idx   += 1

      take = min(steps - frames_written, CR - ctrl_pos)
      out[:, frames_written:frames_written + take, :] = ctrl_buf[:, ctrl_pos:ctrl_pos + take, :]
      frames_written += take
      ctrl_pos       += take

    out = out.permute(0, 2, 1)  # [1, D, steps]
    out = out * prediction_annealing.unsqueeze(1) + transposition

    # Persist all state as tensors so TorchScript keeps mutations across calls.
    self.token_buffer.copy_(local_buf)
    self._current_len.fill_(current_len)
    self._emit_idx.fill_(emit_idx)
    self._ctrl_buf.copy_(ctrl_buf)
    self._ctrl_pos.fill_(ctrl_pos)

    if x.size(0) > 1:
      out = out.repeat_interleave(x.size(0), dim=0)
    if self.resample_ratio != 1:
      out = F.interpolate(out, scale_factor=self.resample_ratio, mode='linear')

    return out.float()


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
    if self.pretrained.encoder is None:
      raise RuntimeError("encode() requested but the pretrained model has no encoder (latent_size=0)")
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
  parser.add_argument('--prior_kind', default='mulaw', choices=['mulaw', 'discrete'], help='Which prior checkpoint type to load')
  parser.add_argument('--compressor_checkpoint', type=str, default=None, help='LatentCompressor checkpoint (required for prior_kind=discrete)')
  config = parser.parse_args()

  cc.use_cached_conv(config.streaming)

  checkpoint_path = find_checkpoint(config.model_directory, typ=config.type)
  print(f"exporting model from checkpoint: {checkpoint_path}")

  format = config.output_path.split('.')[-1]
  if format not in ['ts', 'onnx']:
    raise ValueError(f'Invalid format: {format}, supported formats are: ts, onnx')

  prior = None
  prior_discrete = None
  compressor = None
  if config.prior_directory is not None:
    prior_checkpoint_path = None
    if config.prior_kind == 'discrete':
      if config.type == 'best':
        # Prefer best_acc checkpoint for discrete prior.
        for root, _, files in os.walk(config.prior_directory):
          for file in files:
            if 'best_acc' in file and file.endswith('.ckpt'):
              p = os.path.join(root, file)
              if prior_checkpoint_path is None or os.path.getctime(p) > os.path.getctime(prior_checkpoint_path):
                prior_checkpoint_path = p
      if prior_checkpoint_path is None:
        prior_checkpoint_path = find_checkpoint(config.prior_directory, typ=config.type)
      print("exporting discrete prior model from checkpoint: ", prior_checkpoint_path)

      if config.compressor_checkpoint is None:
        raise RuntimeError("--compressor_checkpoint is required when --prior_kind=discrete")

      prior_discrete = PriorDiscrete.load_from_checkpoint(prior_checkpoint_path, strict=False).to('cpu')
      prior_discrete.eval()
      prior_discrete._trainer = L.Trainer()

      compressor_full = LatentCompressor.load_from_checkpoint(config.compressor_checkpoint, strict=False).to('cpu')
      compressor_full.eval()
      compressor_full._trainer = L.Trainer()

      if getattr(compressor_full, 'use_skip_connections', True):
        raise RuntimeError("LatentCompressor must have use_skip_connections=False for codes-only decoding")
      if getattr(compressor_full, 'vq', None) is None:
        raise RuntimeError("LatentCompressor must have vq_enabled=True for discrete-prior export")

      compressor = LatentCompressorDecodeOnly(
        vq=compressor_full.vq,
        decoder=compressor_full.decoder,
        compression_ratio=int(getattr(compressor_full, 'compression_ratio', 32)),
      ).to('cpu')
      compressor.eval()
    else:
      prior_checkpoint_path = find_checkpoint(config.prior_directory, typ=config.type)
      print("exporting prior model from checkpoint: ", prior_checkpoint_path)
      prior = Prior.load_from_checkpoint(prior_checkpoint_path, strict=False).to('cpu')
      prior.eval()
      if prior._normalization_dict is not None:
        for k, v in prior._normalization_dict.items():
          if torch.is_tensor(v):
            prior._normalization_dict[k] = v.to('cpu')

      prior._trainer = L.Trainer()

  # Reconstruct minimal ControlSpace from checkpoint hyperparameters
  ckpt = torch.load(checkpoint_path, map_location='cpu')
  hparams = ckpt.get('hyper_parameters', {})
  feature_dim = int(hparams.get('feature_dim', 0))
  latent_size = int(hparams.get('latent_size', 0))
  fields = []
  if feature_dim > 0:
    fields.append(ControlField(name='features', dim=feature_dim, source='feature', extractor=None))
  if latent_size > 0:
    fields.append(ControlField(name='latents', dim=latent_size, source='latent', extractor=None))
  if len(fields) == 0:
    raise RuntimeError("Checkpoint missing feature_dim/latent_size hparams; cannot reconstruct ControlSpace for export.")
  control_space = ControlSpace(tuple(fields))

  ddsp = DDSP.load_from_checkpoint(checkpoint_path, strict=False, streaming=True, device='cpu', control_space=control_space).to('cpu')
  ddsp.streaming(True)

  if config.prior_kind == 'discrete' and prior_discrete is not None:
    prior = PriorDiscreteWrapper(prior_discrete, compressor, resample_ratio=(config.target_fs / float(ddsp.fs)))

  if format == 'onnx':
    ddsp.eval()
    scripted = ONNXDDSP(ddsp).to('cpu')
    torch.onnx.dynamo_export(
      scripted,
      torch.zeros(1, ddsp.n_channels, 2**14),
    ).save(config.output_path)
  elif format == 'ts':
    # ugly workaround for the torchscript
    ddsp._trainer = L.Trainer()
    ddsp._recons_loss = None
    ddsp._mr_stft_loss = None
    ddsp._mr_mel_loss = None
    ddsp._mr_chroma_loss = None
    ddsp._m2l_loss = None
    ddsp._discriminator = None
    ddsp._sliced_wasserstein_loss = None

    ddsp.eval()


    scripted = ScriptedDDSP(ddsp, prior, config.target_fs).to('cpu')
    # Registering/scripting the nn~ methods can advance the discrete-prior wrapper's
    # token buffer with junk; clear it so the saved model starts from a clean START.
    if isinstance(prior, PriorDiscreteWrapper):
      prior.reset_state()
      print("discrete prior wrapper state after reset:",
            int(prior._current_len.item()), int(prior._emit_idx.item()))
    scripted.export_to_ts(config.output_path)

    print("Model exported to: ", config.output_path)
