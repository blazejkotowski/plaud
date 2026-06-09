import os
import torch

from ddsp.prior.dataset import PriorSequenceDataset
from ddsp.prior.lmdb_cache import ensure_prior_controls_lmdb

class TinyAudioDS:
    def __init__(self, T=44100*2, sampling_rate=44100, sequence_length=64, stride_factor=1.0):
        self._sampling_rate = sampling_rate
        self._sequence_length = sequence_length
        self._stride_factor = stride_factor
        # Simple sine audio and dummy features (loudness-like)
        t = torch.linspace(0, 2, T)
        audio = torch.sin(2 * torch.pi * 220 * t)
        self._audio = audio
        # Features at audio-rate: stack sine and its absolute as two features
        self._features = torch.stack([audio.abs(), audio], dim=-1)

class DummyDDSP:
    def __init__(self, latent_size=4, resampling_factor=256):
        self._resampling_factor = resampling_factor
        self.latent_size = latent_size
        self.resampling_factor = resampling_factor
    def to(self, dev):
        return self
    def reparametrize(self, mu, scale):
        return mu, scale
    def _smooth_latents(self, z):
        return z
    def __call__(self, *args, **kwargs):
        pass
    def streaming(self):
        return False
    def __getattr__(self, name):
        # Minimal mu/scale outputs simulating encoder(audio[None, :]) -> (mu, scale)
        if name == 'streaming':
            return False
        return super().__getattribute__(name)
    def __call_encoder__(self, audio_batch):
        T = audio_batch.size(1) // self._resampling_factor
        mu = torch.randn(1, T, 4)
        scale = torch.rand(1, T, 4) * 0.1
        return mu, scale
    def encoder(self, audio_batch):
        return self.__call_encoder__(audio_batch)


def test_exporter_creates_lmdb_with_controls(tmp_path):
    ds = TinyAudioDS(T=44100, sequence_length=64)
    model = DummyDDSP()

    out_dir = tmp_path / 'prior_cache_test.lmdb'
    stats = ensure_prior_controls_lmdb(
        audio_ds=ds,
        ddsp=model,
        out_path=str(out_dir),
        seq_len=ds._sequence_length,
        stride_factor=1.0,
        device='cpu',
    )

    assert os.path.exists(out_dir)
    assert int(stats["seq_len"]) == ds._sequence_length
    assert int(stats["num_sequences"]) > 0

    train_ds = PriorSequenceDataset(path=str(out_dir), in_memory=True)
    assert train_ds.seq_len == ds._sequence_length
    assert train_ds.num_controls > 0
