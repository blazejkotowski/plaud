from __future__ import annotations

import math
from typing import Any, Dict, Optional

import lightning as L
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import cross_entropy

from ddsp.prior.prior import FixedPositionalEncoding


class PriorDiscrete(L.LightningModule):
    """Causal Transformer prior over discrete VQ token sequences.

    Tokens are shaped [B, S, N] where N=num_codebooks and each token is in [0..codebook_size-1].
    """

    def __init__(
        self,
        *,
        num_codebooks: int,
        codebook_size: int,
        embedding_dim: int = 32,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        lr: float = 1e-4,
        device: str = "cuda",
    ):
        super().__init__()
        self.save_hyperparameters()

        self._num_codebooks = int(num_codebooks)
        self._codebook_size = int(codebook_size)
        self._embedding_dim = int(embedding_dim)
        self._d_model = int(self._embedding_dim * self._num_codebooks)
        self._lr = float(lr)
        self._max_len = int(max_len)

        # A learned START token per codebook (id == codebook_size) gives generation
        # an in-distribution cold start instead of a zero/random primer.
        self._start_id = self._codebook_size
        self._vocab_per_codebook = self._codebook_size + 1  # +1 for START

        # One shared embedding table with per-codebook offsets.
        vocab_size = self._vocab_per_codebook * self._num_codebooks
        self._embedding = nn.Embedding(vocab_size, self._embedding_dim)
        self.register_buffer(
            "_codebook_offsets",
            torch.arange(self._num_codebooks, dtype=torch.long) * self._vocab_per_codebook,
            persistent=False,
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=self._d_model,
            nhead=nhead,
            dim_feedforward=int(dim_feedforward),
            dropout=dropout,
        )
        self._encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._positional_encoding = FixedPositionalEncoding(
            embedding_dim=self._embedding_dim,
            max_len=self._max_len,
            dropout=dropout,
            device=device,
        )
        self.register_buffer(
            "_causal_mask_full",
            torch.triu(torch.ones(self._max_len, self._max_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

        self._activation = nn.ReLU()
        self._fc = nn.Linear(self._d_model, self._num_codebooks * self._codebook_size)

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def start_token_id(self) -> int:
        return self._start_id

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Return logits for next-token prediction.

        Args:
            x_tokens: [B, S, N] int/long
        Returns:
            logits: [B, S, N, K] float
        """
        if x_tokens.dtype != torch.long:
            x_tokens = x_tokens.long()

        b, s, n = x_tokens.shape
        if n != self._num_codebooks:
            raise ValueError(f"Expected num_codebooks={self._num_codebooks}, got {n}")
        if s > self._max_len:
            x_tokens = x_tokens[:, -self._max_len :, :]
            s = x_tokens.shape[1]

        # [S, B, N]
        x = x_tokens.permute(1, 0, 2)

        offsets = self._codebook_offsets.view(1, 1, -1)
        idx = (x + offsets).clamp(min=0, max=self._vocab_per_codebook * self._num_codebooks - 1)

        # [S, B, N, E]
        embed = self._embedding(idx)
        embed = self._positional_encoding(embed)

        # [S, B, D]
        pos = embed.reshape(s, b, self._d_model)

        causal_mask = self._causal_mask_full[:s, :s]
        enc = self._encoder(pos, mask=causal_mask) * math.sqrt(self._d_model)

        # [B, S, D]
        enc = enc.permute(1, 0, 2)
        enc = self._activation(enc)

        fc = self._fc(enc)
        logits = fc.view(b, s, self._num_codebooks, self._codebook_size)
        return logits

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        out = self._step(batch)
        self.log("loss", out["loss"], prog_bar=True)
        self.log("acc", out["acc"], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return out

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        out = self._step(batch)
        self.log("val_loss", out["loss"], prog_bar=True)
        self.log("val_acc", out["acc"], prog_bar=True)
        return out

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            threshold=1e-4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def _step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        # batch: [B, S, N]
        if batch.dtype != torch.long:
            batch = batch.long()

        b, s, n = batch.shape
        # Prepend a START token so the model learns P(first token | START).
        start = torch.full((b, 1, n), self._start_id, dtype=torch.long, device=batch.device)
        seq = torch.cat([start, batch], dim=1)  # [B, S+1, N]
        x = seq[:, :-1, :]  # [B, S, N], begins with START
        y = seq[:, 1:, :]   # [B, S, N], the real tokens

        logits = self(x)  # [B, S, N, K]
        y_hat = torch.argmax(logits, dim=-1)
        acc = (y_hat == y).float().mean()

        k = self._codebook_size
        ce = cross_entropy(
            logits.permute(0, 3, 1, 2).reshape(b, k, -1),
            y.reshape(b, -1),
            reduction="none",
        ).nanmean()

        return {"loss": ce, "acc": acc}
