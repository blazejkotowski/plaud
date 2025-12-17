import h5py
import torch
from torch.utils.data import Dataset
from typing import Optional

class PriorSequenceDataset(Dataset):
    """
    Loads control sequences for Prior training.

    - If `hdf5_path` provided, prefers dataset `controls` with shape [num_sequences, seq_len, control_size].
        Falls back to `latents` if `controls` is missing.
    - If path missing, generates synthetic random sequences in [-1, 1].
    """
    def __init__(
        self,
        hdf5_path: Optional[str] = None,
        num_sequences: int = 128,
        seq_len: int = 256,
        latent_size: int = 8,
    ):
        self.hdf5_path = hdf5_path
        self.synthetic = hdf5_path is None
        self.latent_size = latent_size

        if self.synthetic:
            self.num_sequences = num_sequences
            self.seq_len = seq_len
        else:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'controls' in f:
                    ds = f['controls']
                else:
                    ds = f['latents']
                self.num_sequences = ds.shape[0]
                self.seq_len = ds.shape[1]
                self.latent_size = ds.shape[2]

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self.synthetic:
            x = torch.randn(self.seq_len, self.latent_size).tanh()
            return x
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'controls' in f:
                x = torch.tensor(f['controls'][idx], dtype=torch.float32)
            else:
                x = torch.tensor(f['latents'][idx], dtype=torch.float32)
        return x
