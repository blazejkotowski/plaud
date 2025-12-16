import torch
import pytest

from ddsp.ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace


def make_control_space(feature_dim: int, latent_dim: int) -> ControlSpace:
    fields = []
    if feature_dim > 0:
        fields.append(ControlField(name="features", dim=feature_dim, source="feature", extractor=None))
    if latent_dim > 0:
        fields.append(ControlField(name="latents", dim=latent_dim, source="latent", extractor=None))
    return ControlSpace(tuple(fields))


@pytest.mark.parametrize("feature_dim,latent_dim", [(0,4), (2,0), (2,2)])
def test_forward_shapes(feature_dim, latent_dim):
    B = 2
    resampling = 32
    fs = 16000
    T_ctl = (fs // 2) // resampling
    n_signal = fs // 2

    control_space = make_control_space(feature_dim, latent_dim)

    # Minimal synth config: use BendableNoiseBandSynth if available via config
    # If registry not used here, rely on existing default in DDSP synth_configs empty -> error.
    synth_configs = [{"class": "BendableNoiseBandSynth", "params": {"n_filters": 32, "fs": fs, "resampling_factor": 32}}]

    model = DDSP(
        control_space=control_space,
        synth_configs=synth_configs,
        fs=fs,
        resampling_factor=resampling,
        adversarial_loss=False,
    )
    model = model.to("cuda")

    audio = torch.randn(B, n_signal, device="cuda")
    if feature_dim > 0:
        features = torch.randn(B, T_ctl, feature_dim, device="cuda")
    else:
        # When no features, provide an empty last-dim tensor of shape [B,T,0]
        features = torch.empty(B, T_ctl, 0, device="cuda")

    y = model.forward(audio, features)

    assert y.dim() == 3
    assert y.shape[0] == B
    assert y.shape[2] <= n_signal


