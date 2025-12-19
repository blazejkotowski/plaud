from typing import Optional
import h5py
import torch
import torch.nn.functional as F

from ddsp.audio_feature_dataset import AudioFeatureDataset
from ddsp.ddsp import DDSP


@torch.no_grad()
def export_latents(
    audio_ds: AudioFeatureDataset,
    model: DDSP,
    out_path: str,
    seq_len: int = 128,
    stride_factor: float = 0.2,
    device: Optional[str] = None,
):
    """
    Export DDSP control-rate features and latents (if available) to HDF5.

    - Writes concatenated `controls`: [num_sequences, seq_len, feature_size + latent_size]
    - Also writes separate `latents` dataset only when latent_size > 0.

    Sequences are created by windowing with the dataset's control-rate alignment.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    # Disable streaming if the encoder supports it
    try:
        if hasattr(model, 'encoder') and model.encoder is not None and hasattr(model.encoder, 'streaming'):
            model.encoder.streaming = False
    except Exception:
        pass

    audio = audio_ds._audio.to(dev)
    features = audio_ds._features.to(dev)  # audio-rate features

    # Encode in manageable chunks (e.g., 40s)
    sr = audio_ds._sampling_rate if hasattr(audio_ds, "_sampling_rate") else 44100
    resampling_factor = model.resampling_factor

    # Sequence/windowing params (fallback defaults)
    stride = max(1, int(seq_len * stride_factor))

    windows_lat = []
    windows_feat = []
    windows_controls = []

    l_chunk = sr * 40
    for i_chunk in range((audio.size(0) + l_chunk - 1) // l_chunk):
        a = audio[i_chunk * l_chunk : (i_chunk + 1) * l_chunk]
        f = features[i_chunk * l_chunk : (i_chunk + 1) * l_chunk]
        if a.numel() < resampling_factor:
            continue

        # Downsample features to control frames
        feat = f.t().unsqueeze(0)  # [1, F, Ta]
        feat = F.interpolate(
            feat,
            scale_factor=1 / resampling_factor,
            mode="linear",
            align_corners=False,
        ).squeeze(0).t()  # [T_ctl, F]

        # Latents: present if model exposes an encoder
        has_latents = getattr(model, 'encoder', None) is not None
        if has_latents:
            mu, scale = model.encoder(a.unsqueeze(0))
            if hasattr(model.encoder, 'reparametrize'):
                z, _ = model.encoder.reparametrize(mu, scale)
            elif hasattr(model, 'reparametrize'):
                z, _ = model.reparametrize(mu, scale)
            else:
                z = mu
            z = model._smooth_latents(z).squeeze(0)  # [T_ctl, Dz]
            # Ensure device matches features for concatenation
            z = z.to(feat.device)
        else:
            Dz = int(getattr(model, 'latent_size', 0))
            T_ctl = feat.size(0)
            if Dz > 0:
                z = torch.zeros(T_ctl, Dz, device=feat.device, dtype=feat.dtype)
            else:
                # Represent no-latent case with an empty last dimension
                z = torch.empty((T_ctl, 0), device=feat.device, dtype=feat.dtype)

        Tz = feat.size(0)
        n_chunks = (Tz - seq_len) // stride + 1
        for i in range(max(0, n_chunks)):
            z_win = z[i * stride : i * stride + seq_len]
            f_win = feat[i * stride : i * stride + seq_len]
            windows_lat.append(z_win)
            windows_feat.append(f_win)
            windows_controls.append(torch.cat([f_win, z_win], dim=-1))

    if not windows_lat:
        raise RuntimeError("No latent windows produced; check audio length and seq_len.")

    # Stack to tensors
    lat_stack = torch.stack(windows_lat, dim=0).cpu()
    feat_stack = torch.stack(windows_feat, dim=0).cpu()
    ctrl_stack = torch.stack(windows_controls, dim=0).cpu()

    # Write HDF5
    with h5py.File(out_path, "w") as h:
        # Always write concatenated controls
        h.create_dataset("controls", data=ctrl_stack.numpy(), compression="gzip")
        # Also provide separate latents if any
        if lat_stack.size(2) > 0:
            h.create_dataset("latents", data=lat_stack.numpy(), compression="gzip")

    return {
        "num_sequences": lat_stack.size(0),
        "seq_len": lat_stack.size(1),
        "latent_size": lat_stack.size(2),
        "feature_size": feat_stack.size(2),
        "control_size": ctrl_stack.size(2),
        "path": out_path,
    }


def build_controls_hdf5(
    audio_ds: AudioFeatureDataset,
    model: DDSP,
    out_path: str,
    seq_len: int = 128,
    stride_factor: float = 0.2,
    device: Optional[str] = None,
):
    """
    Build an HDF5 controls dataset (and latents if available) for Prior training.

    Returns a stats dict with counts and path.
    """
    return export_latents(
        audio_ds=audio_ds,
        model=model,
        out_path=out_path,
        seq_len=seq_len,
        stride_factor=stride_factor,
        device=device,
    )
