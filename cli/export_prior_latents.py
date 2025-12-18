import os
import torch
import hydra
from omegaconf import DictConfig
from typing import List

from ddsp.prior.latents_dataset_builder import build_controls_hdf5
from ddsp.audio_feature_dataset import AudioFeatureDataset
from ddsp.ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace

def _build_control_space(cfg) -> ControlSpace:
    fields: List[ControlField] = []
    for f in cfg:
      fields.append(ControlField(
        name=f.name,
        dim=int(f.dim),
        source=str(f.source),
        extractor=str(f.extractor) if 'extractor' in f else None,
        params=dict(f.params) if 'params' in f else {},
        normalization=dict(f.normalization) if 'normalization' in f else None,
      ))
    return ControlSpace(tuple(fields))


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig):
    """Export DDSP latents and features to HDF5 using the training experiment config."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Shared audio params
    fs = int(cfg.audio.fs)
    resampling_factor = int(cfg.audio.resampling_factor)
    chunk_duration_s = float(cfg.audio.chunk_duration_s)
    n_signal = int(fs * chunk_duration_s)

    # Paths and control space
    dataset_path = cfg.data.dataset_path
    control_space = _build_control_space(cfg.data.control_space)

    # Dataset (same as training)
    ds = AudioFeatureDataset(
        dataset_path=dataset_path,
        n_signal=n_signal,
        sampling_rate=fs,
        resampling_factor=resampling_factor,
        control_space=control_space,
        device=device,
    )

    # Synths from config
    synth_configs = []
    for s in cfg.model.synths:
        synth_configs.append({"class": s.type, "params": dict(s.params)})

    # Model (same core args as training)
    model = DDSP(
        control_space=control_space,
        synth_configs=synth_configs,
        fs=fs,
        resampling_factor=resampling_factor,
        latent_smoothing_kernel=int(cfg.model.latent_smoothing_kernel),
        decoder_gru_layers=int(cfg.model.decoder_gru_layers),
        learning_rate=float(cfg.model.learning_rate),
        plateau_patience=int(cfg.model.plateau_patience),
        capacity=int(cfg.model.capacity),
        losses=[dict(l) for l in getattr(cfg, 'losses', [])],
        adversarial_loss=bool(cfg.adversarial.enabled),
        adv_g_start_epoch=int(cfg.adversarial.schedule.g_start_epoch),
        adv_d_start_epoch=int(cfg.adversarial.schedule.d_start_epoch),
        adv_gen_weight=float(cfg.adversarial.weights.gen),
        adv_disc_weight=float(cfg.adversarial.weights.disc),
        adv_fm_weight=float(cfg.adversarial.weights.fm),
        device=device,
    ).to(device)

    # Output path (allow Hydra +prior_export.out_path override, else use default)
    out_path = (
        cfg.get('prior_export', {}).get('out_path')
        if 'prior_export' in cfg else 'latents.h5'
    )

    stats = build_controls_hdf5(ds, model, out_path=out_path, device=device)
    print(f"Exported latents to {stats['path']} with {stats['num_sequences']} sequences")

if __name__ == "__main__":
    main()
