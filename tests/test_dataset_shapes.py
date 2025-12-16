import torch


def test_fake_dataset_shapes():
    # Fake dataset mimicking AudioFeatureDataset output contracts
    fs = 16000
    resampling = 32
    n_signal = fs // 2
    T_ctl = n_signal // resampling
    D_feat = 2
    B = 3

    audio = torch.randn(B, n_signal)
    features = torch.randn(B, T_ctl, D_feat)

    assert audio.shape == (B, n_signal)
    assert features.shape == (B, T_ctl, D_feat)
    assert T_ctl * resampling <= n_signal
