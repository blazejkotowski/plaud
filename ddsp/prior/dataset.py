import h5py
import torch
from torch.utils.data import Dataset
from typing import Optional

class PriorSequenceDataset(Dataset):
    """Loads control sequences for Prior training."""

    def __init__(
        self,
        hdf5_path: Optional[str] = None,
        num_sequences: int = 128,
        seq_len: int = 256,
        latent_size: int = 8,
        in_memory: bool = True,
    ):
        self.hdf5_path = hdf5_path
        self.synthetic = hdf5_path is None
        self.latent_size = latent_size
        self.in_memory = in_memory
        self._dataset_name: Optional[str] = None
        self._h5 = None
        self._dataset = None
        self._data: Optional[torch.Tensor] = None

        if self.synthetic:
            self.num_sequences = num_sequences
            self.seq_len = seq_len
        else:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'controls' in f:
                    ds = f['controls']
                    self._dataset_name = 'controls'
                else:
                    ds = f['latents']
                    self._dataset_name = 'latents'
                self.num_sequences = ds.shape[0]
                self.seq_len = ds.shape[1]
                self.latent_size = ds.shape[2]
                if self.in_memory:
                    np_data = ds[:]
                    if np_data.dtype != 'float32':
                        np_data = np_data.astype('float32')
                    self._data = torch.from_numpy(np_data)
                    try:
                        self._data.share_memory_()
                    except RuntimeError:
                        pass

    def _ensure_dataset(self):
        if self.synthetic or self._dataset is not None or self._data is not None:
            return
        self._h5 = h5py.File(self.hdf5_path, 'r')
        dataset_name = self._dataset_name or ('controls' if 'controls' in self._h5 else 'latents')
        self._dataset_name = dataset_name
        self._dataset = self._h5[dataset_name]

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            finally:
                self._h5 = None
                self._dataset = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        # h5py objects cannot be pickled; recreate per worker
        state['_h5'] = None
        state['_dataset'] = None
        return state

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self.synthetic:
            x = torch.randn(self.seq_len, self.latent_size).tanh()
            return x
        if self._data is not None:
            return self._data[idx]
        self._ensure_dataset()
        x = torch.as_tensor(self._dataset[idx], dtype=torch.float32)
        return x
