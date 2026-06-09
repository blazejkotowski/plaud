import torch
from ddsp.ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace


def make_cs():
    return ControlSpace(tuple([
        ControlField(name="features", dim=2, source="feature"),
    ]))


def test_param_slicing_and_sum():
    fs = 16000
    resampling = 32
    T_ctl = (fs // 2) // resampling

    # Two synths with known param sizes
    synth_configs = [
        {"class": "BendableNoiseBandSynth", "params": {"n_filters": 16, "fs": fs, "resampling_factor": resampling}},
        {"class": "SineSynth", "params": {"n_sines": 8, "fs": fs, "resampling_factor": resampling}},
    ]

    model = DDSP(
        control_space=make_cs(),
        synth_configs=synth_configs,
        fs=fs,
        resampling_factor=resampling,
        adversarial_loss=False,
    ).to("cuda")

    total = sum(s.n_params for s in model.synths)
    assert total == model._total_synth_params

    B = 2
    params = torch.randn(B, total, T_ctl, device="cuda")
    y = model._synthesize(params)

    assert y.dim() == 3
    assert y.shape[0] == B
