import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from ddsp.ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace


def make_model(fs=16000, resampling=32, feature_dim=2, adv=True, g_start=0, d_start=0):
    cs = ControlSpace(tuple([
        ControlField(name="features", dim=feature_dim, source="feature"),
    ]))
    synth_configs = [{"class": "BendableNoiseBandSynth", "params": {"n_filters": 16, "fs": fs, "resampling_factor": resampling}}]
    return DDSP(
        control_space=cs,
        synth_configs=synth_configs,
        fs=fs,
        resampling_factor=resampling,
        adversarial_loss=adv,
        adv_g_start_epoch=g_start,
        adv_d_start_epoch=d_start,
    )


def make_loader(fs=16000, resampling=32, feature_dim=2, B=2):
    n_signal = fs // 2
    T_ctl = n_signal // resampling
    audio = torch.randn(B, n_signal)
    features = torch.randn(B, T_ctl, feature_dim)
    ds = TensorDataset(audio, features)
    return DataLoader(ds, batch_size=B)


def test_training_runs_with_adversarial_enabled():
    fs = 16000
    resampling = 32
    model = make_model(fs, resampling, adv=True, g_start=0, d_start=0).to("cuda")
    loader = make_loader(fs, resampling)

    trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)


def test_training_runs_without_adversarial_until_threshold():
    fs = 16000
    resampling = 32
    # Start adversarial after large epoch -> effectively off for 1 epoch run
    model = make_model(fs, resampling, adv=True, g_start=1000, d_start=1000).to("cuda")
    loader = make_loader(fs, resampling)

    trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
