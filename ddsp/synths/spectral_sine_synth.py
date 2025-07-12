import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import nn

from .synths import BaseSynth

from scipy.signal.windows import blackmanharris, triang

from matplotlib import pyplot as plt

class SpectralSineSynth(BaseSynth):
  def __init__(self, fs=44100, n_sines=500, fft_size: int = 512, streaming=False, device='cuda'):
    super().__init__()
    self._fs = fs
    self._n_sines = n_sines
    self.streaming = streaming
    self._device = device

    self.fft_size = fft_size  # FFT size
    self.hop_size = self.fft_size // 4  # Hop size

    self.register_buffer("_phases", torch.empty(0))
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
  def resampling_rate(self):
    return self.hop_size

  @property
  def n_params(self):
    return 2 * self._n_sines

  def forward(self, parameters: torch.Tensor, sines_number_attenuation: float = 0.0):
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

    # phases_memory = []
    # For each frame
    for frame_id in range(n_frames):
      curr_freq = tfreq[:, :, frame_id]  # [batch_size, n_sines]
      curr_mag = tmag[:, :, frame_id]    # [batch_size, n_sines]

      if frame_id > 0:
        # propagate phases (from last frames)
        delta_phi = (math.pi * (last_freq + curr_freq) / self._fs) * self.hop_size
        curr_phase = (curr_phase + delta_phi)

      # Generate sines in the spectrum
      # phases_memory.append(curr_phase.cpu().item())  # save phases for debugging
      Y = self._gen_spec_sines(curr_freq, curr_mag, curr_phase, plot=frame_id==3)  # [B, N]
      # if frame_id == 5:
      #   print(Y.real)

      last_freq = curr_freq # save frequenct for phase propagation
      curr_phase = curr_phase % (2 * np.pi)

      # IFFT and overlap-add
      yw = torch.fft.fftshift(torch.fft.ifft(Y, n=self.fft_size)).real # compute inverse FFT

      yw_nonwindowed = yw.clone()  # keep non-windowed output for debugging
      yw = yw * sw[None, None, :]  # [B, 1, N] # apply synthesis window

      start = frame_id * self.hop_size # compute the start index for overlap-add
      y[:, :, start:start + self.fft_size] += yw # overlap-add

      # if frame_id in [3,4,5]:
      #   print(f"Magnitude: {curr_mag}")
      #   plt.plot(Y.abs().squeeze().detach().cpu().numpy())
      #   plt.show()
      # if frame_id == 5:
        # plt.plot(Y.abs().squeeze().detach().cpu().numpy())
        # plt.title("Spectrum magnitude")
        # plt.show()
        # plt.plot(yw_nonwindowed.squeeze().detach().cpu().numpy())
        # plt.title("IFFT output (non-windowed)")
        # plt.show()
        # plt.plot(sw.squeeze().detach().cpu().numpy())
        # plt.title(f"Synthesis window. Max value at idx {sw.argmax()}, value: {sw[sw.argmax()]} ")
        # plt.grid(True)
        # plt.show()
        # yw_np = yw.squeeze().detach().cpu().numpy()
        # plt.plot(yw_np)
        # plt.title("Windowed IFFT output")
        # plt.grid(True, 'both')

        # Find first and last non-zero indices
        # nonzero_indices = np.flatnonzero(yw_np)
        # if nonzero_indices.size > 0:
        #     first_nonzero = nonzero_indices[0]
        #     last_nonzero = nonzero_indices[-1]
        #     plt.axvline(first_nonzero, color='g', linestyle='--', label=f'First non-zero: {first_nonzero}')
        #     plt.axvline(last_nonzero, color='m', linestyle='--', label=f'Last non-zero: {last_nonzero}')
        #     plt.legend()
        # plt.show()
        # plt.plot(y.cpu().numpy()[0,0, start : start+self.fft_size])
        # plt.axhline(y=1, color='r', linestyle='--')
        # plt.axhline(y=-1, color='r', linestyle='--')
        # plt.title("Overlap-add output")
        # plt.show()



    # plt.plot(phases_memory)
    # plt.title("Phases over time")
    # plt.xlabel("Frame ID")
    # plt.ylabel("Phase (radians)")
    # plt.xlim(300, 600)
    # plt.show()
    # Update phases for streaming
    if self.streaming:
      self._phases = curr_phase.detach()

    # Trim extra windows
    y = y[:, :, hN:-hN] # delete half of the first and last windows
    return y

  def _gen_spec_sines(self, freq: torch.Tensor, mag: torch.Tensor, phase: torch.Tensor, plot=False):
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

  def _sinc(self, x: torch.Tensor, N):
    x = x + 1e-18  # avoid div by zero
    return torch.sin(N * x / 2) / torch.sin(x / 2)
