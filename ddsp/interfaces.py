"""
Core contracts and schemas for the modular DDSP framework.

This module defines the minimal interfaces (Protocols) that components must
implement to plug into the system, plus the declarative ControlSpace schema
that ties dataset preparation to model I/O.

These are intentionally lightweight and free of framework coupling so the rest
of the codebase can depend on stable contracts instead of concrete classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

import torch


# ---------------------------
# Control space declarations
# ---------------------------

@dataclass(frozen=True)
class ControlField:
    """A single control dimension group in the control space.

    Attributes:
        name: Identifier (e.g., "loudness", "centroid", "latents").
        dim: Number of channels for this field.
        source: "feature" or "latent". Determines dataset vs model origin.
        extractor: Optional feature extractor type name (for feature fields).
        params: Constructor parameters for the extractor (if any).
        normalization: Optional normalization spec (e.g., mean/std or min/max).
    """

    name: str
    dim: int
    source: str  # "feature" | "latent"
    extractor: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    normalization: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class ControlSpace:
    """Declarative control space used by both datasets and model I/O.

    The dataset builders consult this schema to know which feature extractors
    to run and at which rates. The model consumes the concatenated control
    vector ordered by these fields.
    """

    fields: Tuple[ControlField, ...]

    @property
    def total_dim(self) -> int:
        return sum(f.dim for f in self.fields)

    @property
    def feature_dim(self) -> int:
        return sum(f.dim for f in self.fields if f.source == "feature")

    @property
    def latent_dim(self) -> int:
        return sum(f.dim for f in self.fields if f.source == "latent")

    def names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.fields)


def build_control_space(fields_cfg: Iterable[Any]) -> ControlSpace:
    """Build a :class:`ControlSpace` from config-like field specs.

    Supports Hydra/OmegaConf nodes, plain dicts, or simple objects with attributes.
    Expected keys/attrs per field: name, dim, source, optional extractor, params, normalization.
    """

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        get_fn = getattr(obj, "get", None)
        if callable(get_fn):
            try:
                return get_fn(key, default)
            except TypeError:
                # Some config objects have a different get() signature
                pass
        return getattr(obj, key, default)

    fields: List[ControlField] = []
    for f in fields_cfg:
        name = str(_get(f, "name"))
        dim = int(_get(f, "dim"))
        source = str(_get(f, "source"))
        extractor = _get(f, "extractor", None)
        extractor = str(extractor) if extractor is not None else None

        params = _get(f, "params", None)
        params_dict = dict(params) if params is not None else {}

        normalization = _get(f, "normalization", None)
        normalization_dict = dict(normalization) if normalization is not None else None

        fields.append(
            ControlField(
                name=name,
                dim=dim,
                source=source,
                extractor=extractor,
                params=params_dict,
                normalization=normalization_dict,
            )
        )

    return ControlSpace(tuple(fields))


# ---------------------------
# Component Protocols
# ---------------------------

class FeatureExtractor(Protocol):
    """Extracts per-frame features from audio.

    Implementations should be stateless or purely functional w.r.t. inputs.

    Shapes:
        audio: [B, 1, T_audio]
        return: [B, T_feat, C]
    """

    @property
    def name(self) -> str:  # canonical registry name
        ...

    @property
    def output_dim(self) -> int:
        ...

    def __call__(self, audio: torch.Tensor, fs: int) -> torch.Tensor:
        """Return audio-rate features aligned to the input audio length.

        Shapes:
            audio: [T_audio] or [B, T_audio]
            return: [T_audio, C] or [B, T_audio, C]
        """
        ...


class Encoder(Protocol):
    """Maps control features to a latent representation.

    Shapes:
        x: [B, T_feat, D_feat]
        return: [B, T_lat, D_lat]
    """

    @property
    def latent_size(self) -> int:
        ...

    @property
    def rate_hz(self) -> float:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class Decoder(Protocol):
    """Maps controls + latents to per-synth parameter sequences.

    Implementations may produce a single concatenated parameter tensor and/or a
    dict keyed by synth name. The contract below models the concatenated form.

    Shapes:
        controls: [B, T_ctl, D_ctl]
        latents:  [B, T_lat, D_lat]
        return:   [B, T_dec, D_params]
    """

    @property
    def param_size(self) -> int:
        ...

    def forward(self, controls: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        ...


class SynthBlock(Protocol):
    """A differentiable synthesizer block consuming its parameter slice.

    A DDSP model will split the concatenated parameter vector according to
    `param_size` for each synth and sum their audio outputs.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def param_size(self) -> int:
        ...

    @property
    def rate_hz(self) -> float:
        ...

    def forward(self, params: torch.Tensor, fs: int) -> torch.Tensor:
        """Synthesize audio from params.

        Shapes:
            params: [B, T_syn, D_params_for_block]
            return: [B, 1, T_audio]
        """
        ...


class Loss(Protocol):
    """A loss component returning a scalar (or dict of scalars)."""

    @property
    def name(self) -> str:
        ...

    def __call__(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        ...


class AdversarialModule(Protocol):
    """Encapsulates discriminator(s) and GAN losses with scheduling."""

    def generator_loss(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        global_step: int,
        epoch: int,
    ) -> torch.Tensor:
        ...

    def discriminator_loss(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        global_step: int,
        epoch: int,
    ) -> torch.Tensor:
        ...


class Exporter(Protocol):
    """Exports a trained model into a deployable artifact (e.g., TorchScript)."""

    def export(
        self,
        model: torch.nn.Module,
        config: Mapping[str, Any],
        checkpoint_path: Optional[str],
        device: Optional[torch.device] = None,
    ) -> Any:
        ...


__all__ = [
    "ControlField",
    "ControlSpace",
    "build_control_space",
    "FeatureExtractor",
    "Encoder",
    "Decoder",
    "SynthBlock",
    "Loss",
    "AdversarialModule",
    "Exporter",
]
