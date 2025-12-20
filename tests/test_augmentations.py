import pytest
import torch


from ddsp.augmentations import build_audio_augmentation_pipeline
from ddsp.rave_transforms import Compose


def test_build_audio_augmentation_pipeline_empty_returns_none():
    assert build_audio_augmentation_pipeline([], n_signal=1024, sampling_rate=22050) is None
    assert build_audio_augmentation_pipeline(None, n_signal=1024, sampling_rate=22050) is None


def test_build_audio_augmentation_pipeline_randommute_runs_and_mutes():
    fn = build_audio_augmentation_pipeline(
        [{"type": "RandomMute", "params": {"prob": 1.0}}],
        n_signal=1024,
        sampling_rate=22050,
    )
    assert isinstance(fn, Compose)

    x = torch.randn(1024)
    y = fn(x)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.zeros_like(x))


def test_build_audio_augmentation_pipeline_injects_n_signal_for_randomcrop():
    fn = build_audio_augmentation_pipeline(
        [{"type": "RandomCrop", "params": {}}],
        n_signal=512,
        sampling_rate=22050,
    )
    x = torch.randn(512)
    y = fn(x)
    assert y.shape == x.shape


def test_build_audio_augmentation_pipeline_unknown_type_raises():
    with pytest.raises(KeyError):
        build_audio_augmentation_pipeline(
            [{"type": "DefinitelyNotARealAugmentation", "params": {}}],
            n_signal=1024,
            sampling_rate=22050,
        )
