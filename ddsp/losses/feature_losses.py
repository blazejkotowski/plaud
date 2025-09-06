import torch
import torch.nn.functional as F
import math

def _stft(x, n_fft, hop, win=None):
    if win is None:
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win,
                   center=True, return_complex=True)
    return X  # [B, F, T]

def spectral_flux_loss(y, t, n_fft=512, hop=None):
    """L1 between positive spectral flux curves (sum over freq)."""
    if hop is None: hop = n_fft // 4
    Y = _stft(y, n_fft, hop).abs()
    Tgt = _stft(t, n_fft, hop).abs()
    dY   = F.relu(Y[..., 1:] - Y[..., :-1]).sum(dim=-3)   # [B, T-1]
    dTgt = F.relu(Tgt[..., 1:] - Tgt[..., :-1]).sum(dim=-3)
    # normalize per-sample to avoid loudness bias
    eps = 1e-8
    dY   = dY / (dY.amax(dim=-1, keepdim=True) + eps)
    dTgt = dTgt / (dTgt.amax(dim=-1, keepdim=True) + eps)
    return (dY - dTgt).abs().mean()

def instantaneous_frequency_loss(y, t, sr, n_fft=512, hop=None):
    """Phase-aware IF loss with expected phase advance removed."""
    if hop is None: hop = n_fft // 4
    Y = _stft(y, n_fft, hop)
    Tgt = _stft(t, n_fft, hop)
    # time-phase differences
    def phase_diff(X):
        return torch.angle(X[..., 1:] * torch.conj(X[..., :-1]))  # [-pi, pi]
    dphi_Y = phase_diff(Y)
    dphi_T = phase_diff(Tgt)
    # subtract expected linear advance per bin
    B, F, TTm1 = dphi_Y.shape
    k = torch.arange(F, device=y.device).view(1, F, 1)
    w = 2 * math.pi * hop * k / n_fft                      # expected advance
    err = torch.atan2(torch.sin(dphi_Y - w), torch.cos(dphi_Y - w)) \
        - torch.atan2(torch.sin(dphi_T - w), torch.cos(dphi_T - w))
    return err.abs().mean()

def modulation_spectrum_loss(y, t, sr, n_fft=512, hop=None,
                             mod_low_hz=5.0, mod_high_hz=50.0):
    """
    Match temporal modulation spectra of magnitude spectrograms.
    Computes 1D FFT over time of each freq-bin magnitude; compares power
    only in the modulation band [mod_low_hz, mod_high_hz].
    """
    if hop is None: hop = n_fft // 4
    Y = _stft(y, n_fft, hop).abs()      # [B,F,T]
    Tgt = _stft(t, n_fft, hop).abs()
    # temporal FFT over frames
    FY = torch.fft.rfft(Y, dim=-1)      # [B,F,M]
    FT = torch.fft.rfft(Tgt, dim=-1)
    PY = (FY.abs() ** 2)
    PT = (FT.abs() ** 2)
    # pick modulation bins corresponding to desired Hz
    frame_rate = sr / hop
    m = torch.fft.rfftfreq(Y.shape[-1], d=1.0/frame_rate).to(y.device)  # [M]
    band = (m >= mod_low_hz) & (m <= mod_high_hz)
    # L1 on log power in band (perceptual-ish)
    eps = 1e-8
    loss = (torch.log(PY[..., band] + eps) - torch.log(PT[..., band] + eps)).abs().mean()
    return loss

def onset_weighted_mrstft(y, t, sr, n_ffts=(2048,1024,512,256,128,64), hop_ratios=(0.25,)*6, onset_boost=2.0):
    """MR-STFT with per-frame weights from target spectral flux."""
    assert len(n_ffts) == len(hop_ratios)
    total = 0.0
    for n_fft, hr in zip(n_ffts, hop_ratios):
        hop = int(n_fft * hr)
        # base magnitude losses
        Y = _stft(y, n_fft, hop).abs()
        T = _stft(t, n_fft, hop).abs()
        # frame weights from target flux
        flux = F.relu(T[..., 1:] - T[..., :-1]).sum(dim=-3)  # [B,T-1]
        w = flux / (flux.amax(dim=-1, keepdim=True) + 1e-8)
        w = 1.0 + onset_boost * F.pad(w, (1,0))              # [B,T]
        # L1 on log-mag with weights
        loss = (w * (torch.log(Y+1e-6) - torch.log(T+1e-6)).abs()).mean()
        total += loss
    return total / len(n_ffts)

def highpass_wave(x):
    """Simple 1st-diff HP: emphasises very high freq/transients. x:[B,T]"""
    hp = x[..., 1:] - x[..., :-1]
    # pad to original length for convenience
    return F.pad(hp, (1,0))
