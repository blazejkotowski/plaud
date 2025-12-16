import torch
from ddsp.ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace


def test_loss_is_finite_and_grad_flows():
    fs = 16000
    resampling = 32
    T_ctl = (fs // 2) // resampling
    B = 2

    cs = ControlSpace(tuple([
        ControlField(name="features", dim=2, source="feature"),
    ]))

    synth_configs = [{"class": "BendableNoiseBandSynth", "params": {"n_filters": 16, "fs": fs, "resampling_factor": resampling}}]

    model = DDSP(
        control_space=cs,
        synth_configs=synth_configs,
        fs=fs,
        resampling_factor=resampling,
        adversarial_loss=False,
    ).to("cuda")

    audio = torch.randn(B, fs // 2, device="cuda")
    features = torch.randn(B, T_ctl, 2, device="cuda")

    y = model.forward(audio, features)
    loss = model._reconstruction_loss(y.float(), audio.float())

    assert torch.isfinite(loss).item()

    # Basic gradient flow check
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
