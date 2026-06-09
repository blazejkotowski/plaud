import torch
import importlib
import pytest

omegaconf_spec = importlib.util.find_spec("omegaconf")
hydra_spec = importlib.util.find_spec("hydra")

from ddsp.ddsp import DDSP
if hydra_spec is not None:
    from ddsp.interfaces import build_control_space


@pytest.mark.skipif(omegaconf_spec is None or hydra_spec is None, reason="OmegaConf/Hydra not installed in env")
def test_config_hybrid_builds_model_and_forward():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/experiment_hybrid.yaml")

    fs = int(cfg.audio.fs)
    resampling = int(cfg.model.resampling_factor)
    n_signal = int(fs * float(cfg.audio.chunk_duration_s))
    T_ctl = n_signal // resampling

    control_space = build_control_space(cfg.model.control_space)

    synth_configs = []
    for s in cfg.model.synths:
        synth_configs.append({"class": s.type, "params": dict(s.params)})

    model = DDSP(
        control_space=control_space,
        synth_configs=synth_configs,
        fs=fs,
        resampling_factor=resampling,
        adversarial_loss=False,
    ).to("cuda")

    B = 2
    audio = torch.randn(B, n_signal, device="cuda")
    features = torch.randn(B, T_ctl, control_space.feature_dim, device="cuda")

    y = model.forward(audio, features)
    assert y.shape[0] == B
