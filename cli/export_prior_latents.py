import os
import torch
import hydra
from omegaconf import DictConfig

from ddsp.prior.exporter import export_latents
from ddsp.audio_feature_dataset import AudioFeatureDataset
from ddsp.ddsp import DDSP

@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig):
    """Export DDSP latents to HDF5 using the training experiment config.

    Expects the experiment config to build `AudioFeatureDataset` and `DDSP` similarly
    to training, or you can override paths and parameters via CLI.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Minimal dataset/model instantiation from cfg (adjust to your config structure)
    ds = AudioFeatureDataset(
        audio_path=cfg.audio.get('path', None),
        sampling_rate=cfg.audio.get('sampling_rate', 44100),
        control_space=cfg.control_space,
        sequence_length=cfg.model.get('sequence_length', 256),
        stride_factor=cfg.model.get('stride_factor', 1.0),
        device=device,
    )

    model = DDSP(
        control_space=cfg.control_space,
        encoder_cfg=cfg.model.get('encoder', {}),
        decoder_cfg=cfg.model.get('decoder', {}),
        synth_cfg=cfg.model.get('synth', {}),
        losses=cfg.get('losses', []),
        adversarial=cfg.get('adversarial', {}),
    ).to(device)

    out_path = cfg.get('prior_export', {}).get('out_path', 'latents.h5')

    stats = export_latents(ds, model, out_path=out_path, device=device)
    print(f"Exported latents to {stats['path']} with {stats['num_sequences']} sequences")

if __name__ == "__main__":
    main()
