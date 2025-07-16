import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import nn

from .synths import BaseSynth, register_synth

from scipy.signal.windows import blackmanharris, triang

from matplotlib import pyplot as plt

@register_synth
class SpectralSineSynth(BaseSynth):

  def __init__(self, fs=44100, n_sines=500, fft_size: int = 512, streaming=False, device='cuda'):
    super().__init__()
    self._fs = fs
    self._n_sines = n_sines
    self.streaming = streaming
    self._device = device

    self.fft_size = fft_size  # FFT size
    self.hop_size = self.fft_size // 4  # Hop size

    self.register_buffer("_phases", torch.empty(0), persistent=False)
    self._phases_initialized = False

    # Synthesis window (triangular window normalized by Blackman-Harris)
    bh = torch.from_numpy(blackmanharris(self.fft_size)).to(device) # blackmanharris window
    bh = bh / bh.sum() # normalized blackmanharris window
    ow = torch.from_numpy(triang(self.hop_size*2)).to(device) # triangular window
    sw = torch.zeros(self.fft_size, device=device)
    sw[self.fft_size//2 - self.hop_size : self.fft_size//2 + self.hop_size] = ow # add triangular window
    sw[self.fft_size//2 - self.hop_size : self.fft_size//2 + self.hop_size] /= bh[self.fft_size//2 - self.hop_size : self.fft_size//2 + self.hop_size] # normalized synthesis window
    self.register_buffer("_synthesis_window", sw)

  @property
  def resampling_factor(self):
    return self.hop_size

  @property
  def jit_name(self):
    return "SpectralSineSynth"

  @property
  def n_params(self):
    return 2 * self._n_sines


  def forward_old(self, parameters: torch.Tensor, sines_number_attenuation: float = 0.0):
    """
    parameters: [batch_size, 2*n_sines, n_frames]
    Returns waveform of shape [batch_size, 1, n_frames * hop_size]
    """
    batch_size, n_params, n_frames = parameters.shape
    assert n_params == self.n_params, f"Expected {self.n_params} params, got {n_params}"

    tfreq = parameters[:, :self._n_sines, :]  # [batch_size, n_sines, n_frames]
    tmag = parameters[:, self._n_sines:, :]  # [batch_size, n_sines, n_frames]

    # tfreq = tfreq * self._fs / 4  # scale from [0,2] to [0, fs/2]

    # Attenuate sines above Nyquist
    # tmag = tmag * (tfreq < self._fs / 2).float()

    if self.streaming:
      if not self._phases_initialized or self._phases.shape[0] != batch_size:
        self._phases = torch.zeros(batch_size, self._n_sines, device=tfreq.device) * 2 * math.pi
        self._phases_initialized = True
    else:
      self._phases = torch.zeros(batch_size, self._n_sines, device=tfreq.device) * 2 * math.pi

    ysize = self.hop_size * (n_frames + 3) # output sound size
    y = torch.zeros(batch_size, 1, ysize, device=tfreq.device) # initialize output array

    sw = self._synthesis_window  # [self.fft_size]
    hN = self.fft_size // 2 # half of FFT size for synthesis

    last_freq = tfreq[:, :, 0] # [batch_size, n_sines] # initialize synthesis frequencies
    curr_phase = self._phases.clone() # [batch_size, n_sines]

    # For each frame
    for frame_id in range(n_frames):
      curr_freq = tfreq[:, :, frame_id]  # [batch_size, n_sines]
      curr_mag = tmag[:, :, frame_id]    # [batch_size, n_sines]

      if frame_id > 0:
        # propagate phases (from last frames)
        delta_phi = (math.pi * (last_freq + curr_freq) / self._fs) * self.hop_size
        curr_phase = (curr_phase + delta_phi)

      # Generate sines in the spectrum
      Y = self._gen_spec_sines_diff(curr_freq, curr_mag, curr_phase)  # [B, N]

      last_freq = curr_freq # save frequenct for phase propagation
      curr_phase = curr_phase % (2 * np.pi)

      # IFFT and overlap-add
      yw = torch.fft.fftshift(torch.fft.ifft(Y, n=self.fft_size)).real # compute inverse FFT
      yw = yw * sw[None, None, :]  # [B, 1, N] # apply synthesis window

      start = frame_id * self.hop_size # compute the start index for overlap-add
      y[:, :, start:start + self.fft_size] += yw # overlap-add

    if self.streaming:
      self._phases = curr_phase.detach()

    # Trim extra windows
    y = y[:, :, hN:-hN] # delete half of the first and last windows
    return y

  def _gen_spec_sines(self, freq: torch.Tensor, mag: torch.Tensor, phase: torch.Tensor):
    batch_size, n_sines = freq.shape
    hN = self.fft_size // 2
    Y = torch.zeros(batch_size, self.fft_size, dtype=torch.complex64, device=freq.device) # initialize output complex spectrum

    for i in range(n_sines):
      loc = freq[:, i] * self.fft_size / self._fs # it should be in in range ]0,hN-1[]
      bin_center = torch.round(loc).long()
      bin_remainder = bin_center.float() - loc

      for b in range(batch_size):
        if loc[b] == 0 or loc[b] > hN - 1:
          continue

        lobe_bins = torch.arange(-4, 5, device=freq.device)
        lobe_pos = lobe_bins + bin_remainder[b]
        lobe_mag = self._gen_bh_lobe(lobe_pos) * 10**(mag[b, i] / 20)
        # if plot:
        #   import matplotlib.pyplot as plt
        #   plt.plot(lobe_mag.detach().cpu().numpy())
        #   plt.title("Lobe magnitude")
        #   plt.legend()
        #   plt.show()
        b_indices = bin_center[b] + lobe_bins

        for j, bin_idx in enumerate(b_indices):
          if bin_idx < 0: # peak lobe crosses DC bin
            Y[b, -bin_idx] += lobe_mag[j] * torch.exp(-1j * phase[b, i])
          elif bin_idx > hN: # peak lobe crosses Nyquist bin
            Y[b, bin_idx] += lobe_mag[j] * torch.exp(-1j * phase[b, i])
          elif bin_idx == 0 or bin_idx == hN: # peak lobe in the limits of the spectrum
            Y[b, bin_idx] += lobe_mag[j] * torch.exp(1j * phase[b, i]) + lobe_mag[j] * torch.exp(-1j * phase[b, i])
          else: # peak lobe in positive freq range
            Y[b, bin_idx] += lobe_mag[j] * torch.exp(1j * phase[b,i])

    Y[:, hN+1:] = Y[:, 1:hN].flip(dims=[1]).conj()
    return Y

  def forward(self, parameters: torch.Tensor, sines_number_attenuation: float = 0.0):
    """
    parameters: [batch_size, 2*n_sines, n_frames]
    Returns waveform of shape [batch_size, 1, n_frames * hop_size]
    """
    B, n_params, n_frames = parameters.shape
    assert n_params == self.n_params, f"Expected {self.n_params} params, got {n_params}"

    tfreq = parameters[:, :self._n_sines, :]  # [B, n_sines, n_frames]
    tmag = parameters[:, self._n_sines:, :]  # [B, n_sines, n_frames]

    tfreq = tfreq * self._fs / 2  # scale from [0,1] to [0, fs/2]
    # Convert tmag from [0, 1] to dB scale: 0 dB (max) to -120 dB (min)
    tmag = tmag * 120 - 120


    # Attenuate sines above Nyquist
    tmag = tmag * (tfreq < self._fs / 2).float()

    if self.streaming:
        if not self._phases_initialized or self._phases.shape[0] != B:
            self._phases = torch.rand(B, self._n_sines, device=tfreq.device) * 2 * math.pi
            self._phases_initialized = True
    else:
        self._phases = torch.rand(B, self._n_sines, device=tfreq.device) * 2 * math.pi

    # Compute all phases
    curr_phase = self._phases.clone()  # [B, n_sines]
    delta_phi = math.pi * (tfreq[:, :, :-1] + tfreq[:, :, 1:]) / self._fs * self.hop_size  # [B, n_sines, n_frames - 1]
    all_phases = torch.zeros(B, self._n_sines, n_frames, device=tfreq.device)
    all_phases[:, :, 0] = curr_phase
    all_phases[:, :, 1:] = torch.cumsum(delta_phi, dim=-1) + curr_phase.unsqueeze(-1)

    # Flatten across frames
    tfreq_flat = tfreq.permute(0, 2, 1).reshape(-1, self._n_sines)
    tmag_flat = tmag.permute(0, 2, 1).reshape(-1, self._n_sines)
    phase_flat = all_phases.permute(0, 2, 1).reshape(-1, self._n_sines)

    Y_flat = self._gen_spec_sines_diff(tfreq_flat, tmag_flat, phase_flat)  # [B * n_frames, fft_size]
    yw_flat = torch.fft.fftshift(torch.fft.ifft(Y_flat, n=self.fft_size)).real  # [B * n_frames, fft_size]
    yw_flat *= self._synthesis_window[None, :]  # [B * n_frames, fft_size]

    # Reshape to [B, n_frames, fft_size] for scatter-based overlap-add
    yw = yw_flat.view(B, n_frames, self.fft_size)
    ysize = (n_frames - 1) * self.hop_size + self.fft_size
    y = torch.zeros(B, 1, ysize, device=tfreq.device)

    frame_offsets = torch.arange(n_frames, device=tfreq.device) * self.hop_size  # [n_frames]
    # frame_offsets = torch.arange(n_frames, device=tfreq.device) * self.hop_size - self.fft_size // 2
    frame_offsets = frame_offsets.view(1, -1, 1)  # [1, n_frames, 1]
    sample_indices = frame_offsets + torch.arange(self.fft_size, device=tfreq.device).view(1, 1, -1)  # [1, n_frames, fft_size]

    # Expand to batch
    sample_indices = sample_indices.expand(B, -1, -1).reshape(B, -1)  # [B, n_frames * fft_size]
    yw_values = yw.reshape(B, -1)  # [B, n_frames * fft_size]

    y.scatter_add_(2, sample_indices.unsqueeze(1), yw_values.unsqueeze(1))  # [B, 1, T]
    # TODO: this is due to the wrongly put together sample indices, the root cause should be fixed, and this should be removed
    y = torch.roll(y, shifts=y.shape[-1]//2, dims=-1)

    if self.streaming:
        self._phases = all_phases[:, :, -1].detach()

    y = y[:, :, self.fft_size // 2 : -self.fft_size // 2]  # trim
    return y

  def _gen_spec_sines_diff(self, freq: torch.Tensor, mag: torch.Tensor, phase: torch.Tensor):
    """
    Generate spectral sinusoids with Blackman-Harris lobes (differentiable version).

    Args:
      freq: torch.Tensor, frequencies of shape [batch_size, n_sines]
      mag: torch.Tensor, magnitudes of shape [batch_size, n_sines]
      phase: torch.Tensor, phases of shape [batch_size, n_sines]
    Returns:
      Y: torch.Tensor, complex spectrum of shape [batch_size, fft_size]
    """
    batch_size, n_sines = freq.shape
    hN = self.fft_size // 2
    Y = torch.zeros(batch_size, self.fft_size, dtype=torch.complex64, device=freq.device)

    loc = freq * self.fft_size / self._fs  # [B, n_sines]

    # Create validity mask (differentiable)
    valid_mask = (loc > 0) & (loc <= hN - 1)  # [B, n_sines]

    bin_center = torch.round(loc).long()  # [B, n_sines]
    bin_remainder = bin_center.float() - loc  # [B, n_sines]

    lobe_bins = torch.arange(-4, 5, device=freq.device)  # [9]
    lobe_pos = bin_remainder.unsqueeze(-1) + lobe_bins  # [B, n_sines, 9]

    lobe_mag = self._gen_bh_lobe(lobe_pos) * 10 ** (mag.unsqueeze(-1) / 20)  # [B, n_sines, 9]

    bin_indices = bin_center.unsqueeze(-1) + lobe_bins  # [B, n_sines, 9]

    e_pos = torch.exp(1j * phase.unsqueeze(-1))  # [B, n_sines, 1]
    e_neg = torch.exp(-1j * phase.unsqueeze(-1))  # [B, n_sines, 1]

    # Determine cases for each bin
    is_neg = bin_indices < 0  # [B, n_sines, 9]
    is_wrap = bin_indices > hN  # [B, n_sines, 9]
    is_edge = (bin_indices == 0) | (bin_indices == hN)  # [B, n_sines, 9]

    # Calculate write indices - handle negative wrapping
    write_idx = torch.where(is_neg, -bin_indices, bin_indices)

    # Keep indices within bounds for scatter operation
    write_idx = write_idx.clamp(0, self.fft_size - 1)

    # Calculate values for each case (all differentiable)
    val_default = lobe_mag * e_pos  # [B, n_sines, 9]
    val_neg = lobe_mag * e_neg  # [B, n_sines, 9]
    val_mirror = lobe_mag * (e_pos + e_neg)  # [B, n_sines, 9]

   # Start with default values and override based on conditions
    values = val_default.clone()
    values = torch.where(is_neg | is_wrap, val_neg, values)    # Negative frequencies or wrap-around
    values = torch.where(is_edge, val_mirror, values)          # DC or Nyquist bins get both phases

    # Apply validity mask to zero out invalid sinusoids
    valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(values)  # [B, n_sines, 9]
    values = values * valid_mask_expanded.to(values.dtype)

    # Additional mask for valid write indices
    valid_write_mask = (write_idx >= 0) & (write_idx < self.fft_size)
    values = values * valid_write_mask.to(values.dtype)

    # Scatter values into spectrum
    batch_idx = torch.arange(batch_size, device=freq.device).view(batch_size, 1, 1).expand_as(write_idx)

    flat_batch = batch_idx.reshape(-1)
    flat_idx = write_idx.reshape(-1)
    flat_vals = values.reshape(-1).to(Y.dtype)  # Ensure dtype matches Y

    Y.index_put_((flat_batch, flat_idx), flat_vals, accumulate=True)

    # Fill negative frequencies (Hermitian symmetry)
    Y[:, hN+1:] = Y[:, 1:hN].flip(dims=[1]).conj()

    return Y


  def _gen_bh_lobe(self, x: torch.Tensor):
    """
    Generate Blackman-Harris main lobe at offset x
    """
    N = self.fft_size
    f = x * math.pi * 2 / N
    df = 2 * math.pi / N
    consts = [0.35875, 0.48829, 0.14128, 0.01168]
    y = torch.zeros_like(f)
    for m in range(4):
      y += consts[m] / 2 * (self._sinc(f - df * m, N) + self._sinc(f + df * m, N))
    y = y / N / consts[0]
    return y

  def _sinc(self, x: torch.Tensor, N: int):
    x = x + 1e-18  # avoid div by zero
    return torch.sin(N * x / 2) / torch.sin(x / 2)
