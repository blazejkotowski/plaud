from __future__ import annotations

import pickle
from typing import Optional

import lmdb
import torch
from torch.utils.data import Dataset

class PriorSequenceDataset(Dataset):
    """Loads control sequences for Prior training.

    Preferred format is LMDB (directory ending with .lmdb).
    """

    def __init__(
        self,
        path: Optional[str] = None,
        in_memory: bool = True,
    ):
        if path is None:
            raise ValueError("PriorSequenceDataset requires 'path'")

        self.path = str(path)
        self.in_memory = bool(in_memory)

        self._data: Optional[torch.Tensor] = None

        # LMDB state
        self._env = None
        self._meta: Optional[dict] = None

        self._init_lmdb()

    def _init_lmdb(self) -> None:
        env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, subdir=True)
        with env.begin() as txn:
            meta_buf = txn.get(b"metadata")
            if meta_buf is None:
                raise RuntimeError(f"LMDB prior cache missing metadata: {self.path}")
            meta = pickle.loads(meta_buf)

        self._meta = meta
        self.num_sequences = int(meta["num_sequences"])
        self.seq_len = int(meta["seq_len"])
        self._num_controls = int(meta["num_controls"])

        if self.in_memory:
            # materialize to a single tensor (fastest training);
            # format is float32 bytes, shape [seq_len, D]
            data = torch.empty((self.num_sequences, self.seq_len, self._num_controls), dtype=torch.float32)
            with env.begin() as txn:
                for i in range(self.num_sequences):
                    buf = txn.get(f"controls:{i:08d}".encode())
                    if buf is None:
                        raise RuntimeError(f"Missing key controls:{i:08d} in {self.path}")
                    arr = torch.frombuffer(buf, dtype=torch.float32)
                    data[i] = arr.view(self.seq_len, self._num_controls)
            self._data = data
            try:
                self._data.share_memory_()
            except RuntimeError:
                pass
            env.close()
        else:
            self._env = env

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            finally:
                self._env = None

        # no HDF5 resources

    @property
    def num_controls(self):
        if self._data is not None:
            return int(self._data.shape[2])
        if self._meta is not None:
            return int(self._meta["num_controls"])
        raise RuntimeError("PriorSequenceDataset internal state invalid")

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        # lmdb/h5py objects cannot be pickled; recreate per worker
        state['_env'] = None
        return state

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self._data is not None:
            return self._data[idx]
        if self._meta is not None:
            if self._env is None:
                self._env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, subdir=True)
            with self._env.begin() as txn:
                buf = txn.get(f"controls:{idx:08d}".encode())
                if buf is None:
                    raise IndexError(idx)
                x = torch.frombuffer(buf, dtype=torch.float32).view(self.seq_len, self.num_controls)
            return x

        raise RuntimeError("PriorSequenceDataset internal state invalid")


class PriorTokenSequenceDataset(Dataset):
    """Loads discrete token sequences for PriorDiscrete training from LMDB."""

    def __init__(
        self,
        path: Optional[str] = None,
        in_memory: bool = True,
        dtype: torch.dtype = torch.int16,
    ):
        if path is None:
            raise ValueError("PriorTokenSequenceDataset requires 'path'")

        self.path = str(path)
        self.in_memory = bool(in_memory)
        self._dtype = dtype

        self._data: Optional[torch.Tensor] = None
        self._env = None
        self._meta: Optional[dict] = None

        self._init_lmdb()

    def _init_lmdb(self) -> None:
        env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, subdir=True)
        with env.begin() as txn:
            meta_buf = txn.get(b"metadata")
            if meta_buf is None:
                raise RuntimeError(f"LMDB token cache missing metadata: {self.path}")
            meta = pickle.loads(meta_buf)

        self._meta = meta
        self.num_sequences = int(meta["num_sequences"])
        self.seq_len = int(meta["seq_len"])
        self._num_codebooks = int(meta["num_codebooks"])
        self._codebook_size = int(meta["codebook_size"])

        if self.in_memory:
            data = torch.empty((self.num_sequences, self.seq_len, self._num_codebooks), dtype=torch.int16)
            with env.begin() as txn:
                for i in range(self.num_sequences):
                    buf = txn.get(f"tokens:{i:08d}".encode())
                    if buf is None:
                        raise RuntimeError(f"Missing key tokens:{i:08d} in {self.path}")
                    arr = torch.frombuffer(buf, dtype=self._dtype)
                    data[i] = arr.view(self.seq_len, self._num_codebooks).to(torch.int16)
            self._data = data
            try:
                self._data.share_memory_()
            except RuntimeError:
                pass
            env.close()
        else:
            self._env = env

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            finally:
                self._env = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_env'] = None
        return state

    @property
    def num_codebooks(self) -> int:
        return int(self._num_codebooks)

    @property
    def codebook_size(self) -> int:
        return int(self._codebook_size)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self._data is not None:
            return self._data[idx].to(torch.long)
        if self._meta is not None:
            if self._env is None:
                self._env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, subdir=True)
            with self._env.begin() as txn:
                buf = txn.get(f"tokens:{idx:08d}".encode())
                if buf is None:
                    raise IndexError(idx)
                x = torch.frombuffer(buf, dtype=self._dtype).view(self.seq_len, self._num_codebooks)
            return x.to(torch.long)

        raise RuntimeError("PriorTokenSequenceDataset internal state invalid")
