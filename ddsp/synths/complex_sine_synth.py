import torch

from .synths import BaseSynth, register_synth


@register_synth
class ComplexSineSynth(BaseSynth):
  """
  A synthesiser that generates a mixture noise bands from amplitudes.

  Arguments:
    - n_filters: int, the number of filters in the filterbank
    - fs: int, the sampling rate of the input signal
    - resampling_factor: int, the internal up / down sampling factor for the signal
    - device: str, the device to use
  """

  def __init__(self, n_sines: int = 32, fs: int = 44100, resampling_factor: int = 32, device: str = 'cuda'):
    super().__init__()
    self._resampling_factor = resampling_factor
    self._device = device
    self.n_sines = n_sines


  @property
  def n_params(self):
    return 4 * self.n_sines # z_freq, z_phase, amp_start, amp_end


  @property
  def jit_name(self):
    return "ComplexSineSynth"


  def forward(self, parameters: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
    - parameters: torch.Tensor[batch_size, n_params, n_samples], the parameters of the synthesizer
    Returns:
      - signal: torch.Tensor[batch_size, 1, sig_length], the synthesized signal
    """
    k = self.n_sines
    # print("n_sines: ", self.n_sines)
    # print("parameters shape:", parameters.shape)

    z_freqs = parameters[:, :k, :] # frequencies
    z_phases = parameters[:, k:k*2, :] # phases
    amp_starts = parameters[:, k*2:k*3, :] # amp starts
    amp_ends = parameters[:, k*3:, :] # amp ends
    
    signal_length = self._resampling_factor * parameters.shape[-1]
    signal = SynthCoreFunction.apply(z_freqs, z_phases, amp_starts, amp_ends, signal_length)
    # print("batch size:", parameters.shape[0])   
    # print("signal shape:", signal.shape)

    return signal.unsqueeze(1)



class SynthCoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_freq, z_phase, amp_start, amp_end, signal_length):
        ctx.save_for_backward(z_freq, z_phase, amp_start, amp_end)
        ctx.signal_length = signal_length

        device = z_freq.device
        # z_freq: [batch_size, n_sinusoids, n_frames]
        batch_size, num_sinusoids, n_frames = z_freq.shape
        samples_per_frame = signal_length // n_frames

        # Time indices within each frame: [samples_per_frame]
        n = torch.arange(samples_per_frame, device=device, dtype=torch.float32)

        # Expand for broadcasting:
        # z_freq: [batch, n_sin, n_frames] -> [batch, 1, n_sin, n_frames]
        # n: [samples_per_frame] -> [1, samples_per_frame, 1, 1]
        z_freq_expanded = z_freq.unsqueeze(1)       # [batch, 1, n_sin, n_frames]
        z_phase_expanded = z_phase.unsqueeze(1)      # [batch, 1, n_sin, n_frames]
        n_expanded = n.view(1, samples_per_frame, 1, 1)

        # Compute z_phase * z_freq^n per frame
        # Result: [batch, samples_per_frame, n_sin, n_frames]
        z_combined = z_phase_expanded * (z_freq_expanded ** n_expanded)
        sinusoids = torch.real(z_combined)
        # print("sinusoids.shape", sinusoids.shape)

        # Apply amplitude envelope within each frame
        if amp_start is not None and amp_end is not None:
            t = torch.linspace(0, 1, samples_per_frame, device=device)
            # amp_start: [batch, n_sin, n_frames] -> [batch, 1, n_sin, n_frames]
            amp_start_expanded = amp_start.unsqueeze(1)
            amp_end_expanded = amp_end.unsqueeze(1)
            t_expanded = t.view(1, samples_per_frame, 1, 1)
            amp_env = amp_start_expanded * (1 - t_expanded) + amp_end_expanded * t_expanded
            sinusoids = sinusoids * amp_env

        # print("sinusoids after amp env shape", sinusoids.shape)

        # Sum over sinusoids: [batch, samples_per_frame, n_frames]
        sinusoids_summed = sinusoids.sum(dim=2)

        # print("sinusoids_summed shape", sinusoids_summed.shape)

        # Reshape to interleave frames into a continuous signal:
        # [batch, samples_per_frame, n_frames] -> [batch, n_frames, samples_per_frame] -> [batch, signal_length]
        signal = sinusoids_summed.permute(0, 2, 1).reshape(batch_size, signal_length)
        # print("signal shape:", signal.shape)

        return signal

    @staticmethod
    def backward(ctx, grad_output):
        z_freq, z_phase, amp_start, amp_end = ctx.saved_tensors
        signal_length = ctx.signal_length
        device = z_freq.device

        # We'll let autograd compute true grads, then normalize them.
        with torch.enable_grad():
            z_freq_ = z_freq.clone().detach().requires_grad_(True)
            z_phase_ = z_phase.clone().detach().requires_grad_(True)

            amp_start_ = None if amp_start is None else amp_start.clone().detach().requires_grad_(True)
            amp_end_ = None if amp_end is None else amp_end.clone().detach().requires_grad_(True)

            signal = SynthCoreFunction.forward(ctx, z_freq_, z_phase_, amp_start_, amp_end_, signal_length)
            grads = torch.autograd.grad(
                signal,
                (z_freq_, z_phase_, amp_start_, amp_end_),
                grad_output,
                retain_graph=False,
                allow_unused=True
            )

        grad_z_freq, grad_z_phase, grad_amp_start, grad_amp_end = grads

        # 🔥 Normalize the local gradients (only if they exist)
        if grad_z_freq is not None:
            grad_z_freq = grad_z_freq / (grad_z_freq.abs() + 1e-8)
        if grad_z_phase is not None:
            grad_z_phase = grad_z_phase / (grad_z_phase.abs() + 1e-8)

        return grad_z_freq, grad_z_phase, grad_amp_start, grad_amp_end, None