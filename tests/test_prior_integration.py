import os
import tempfile
import torch
import lightning as L
from torch.utils.data import DataLoader

from ddsp.prior.latents_dataset_builder import export_latents
from ddsp.prior.dataset import PriorSequenceDataset
from ddsp.prior.prior import Prior

# Reuse tiny synthetic dataset and dummy DDSP-like model from exporter test
class TinyAudioDS:
    def __init__(self, T=44100*2, sampling_rate=44100, sequence_length=64, stride_factor=1.0):
        self._sampling_rate = sampling_rate
        self._sequence_length = sequence_length
        self._stride_factor = stride_factor
        t = torch.linspace(0, 2, T)
        audio = torch.sin(2 * torch.pi * 220 * t)
        self._audio = audio
        self._features = torch.stack([audio.abs(), audio], dim=-1)

class DummyDDSP:
    def __init__(self, latent_size=4, resampling_factor=256):
        self._resampling_factor = resampling_factor
    def to(self, dev):
        return self
    def reparametrize(self, mu, scale):
        return mu, scale
    def _smooth_latents(self, z):
        return z
    def streaming(self):
        return False
    def __call_encoder__(self, audio_batch):
        T = audio_batch.size(1) // self._resampling_factor
        mu = torch.randn(1, T, 4)
        scale = torch.rand(1, T, 4) * 0.1
        return mu, scale
    def encoder(self, audio_batch):
        return self.__call_encoder__(audio_batch)


def test_export_load_train_prior(tmp_path):
    # Export controls + latents
    ds = TinyAudioDS(T=44100, sequence_length=64)
    ddsp_model = DummyDDSP()

    out_file = tmp_path / 'controls_latents.h5'
    stats = export_latents(ds, ddsp_model, str(out_file), device='cpu')
    assert os.path.exists(out_file)

    # Load controls for training
    train_ds = PriorSequenceDataset(hdf5_path=str(out_file))
    assert train_ds.seq_len == ds._sequence_length

    # Build Prior model with control_size as latent_size
    control_size = train_ds.latent_size
    model = Prior(latent_size=control_size, embedding_dim=16, quantization_channels=32, nhead=4, num_layers=2, dropout=0.0, max_len=128, lr=1e-3, normalization_dict=None, device='cpu')

    # Train for 1 epoch
    dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    trainer = L.Trainer(max_epochs=1, accelerator='cpu', devices=1, log_every_n_steps=5)
    trainer.fit(model, dl)

    # Simple assertion: model should have logged a finite loss in training
    # We can't easily read logs from Trainer here; instead, run a single step
    batch = next(iter(dl))
    loss = model._step(batch)
    assert torch.isfinite(loss['loss'])
