"""
Simple registries for pluggable components (synths, losses, feature extractors, etc.).

Usage:
    from .registry import SYNTHS

    @SYNTHS.register("NoiseBand")
    class NoiseBandSynth(...):
        ...

    synth = SYNTHS.create("NoiseBand", **params)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, TypeVar


T = TypeVar("T")


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: Optional[str] = None):
        def _decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            key = name or getattr(obj, "__name__", None)
            if not key:
                raise ValueError("Registry registration requires a name")
            lname = str(key)
            if lname in self._items:
                raise KeyError(f"{self._name} registry already has key '{lname}'")
            self._items[lname] = obj
            return obj

        return _decorator

    def add(self, name: str, obj: Callable[..., Any]) -> None:
        if name in self._items:
            raise KeyError(f"{self._name} registry already has key '{name}'")
        self._items[name] = obj

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as e:
            options = ", ".join(sorted(self._items.keys())) or "<empty>"
            raise KeyError(f"{self._name} registry has no key '{name}'. Options: {options}") from e

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        ctor = self.get(name)
        return ctor(*args, **kwargs)

    def available(self) -> Iterable[str]:
        return sorted(self._items.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


# Category registries
SYNTHS = Registry("synths")
FEATURE_EXTRACTORS = Registry("feature_extractors")
LOSSES = Registry("losses")
DISCRIMINATORS = Registry("discriminators")
ENCODERS = Registry("encoders")
DECODERS = Registry("decoders")
AUGMENTATIONS = Registry("augmentations")
PRIORS = Registry("priors")
EXPORTERS = Registry("exporters")


def build_from_config(reg: Registry, cfg: Dict[str, Any]) -> Any:
    """Create an instance from a config dict {type: str, params: {...}}."""
    typ = cfg.get("type")
    params = cfg.get("params", {})
    if not typ:
        raise ValueError("Config must contain 'type'")
    return reg.create(typ, **params)


__all__ = [
    "Registry",
    "SYNTHS",
    "FEATURE_EXTRACTORS",
    "LOSSES",
    "DISCRIMINATORS",
    "ENCODERS",
    "DECODERS",
    "AUGMENTATIONS",
    "PRIORS",
    "EXPORTERS",
    "build_from_config",
]
