from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from ddsp.rave_transforms import (
    Compose,
    RandomCompress,
    RandomCrop,
    RandomGain,
    RandomMute,
    RandomPitch,
)


_AUGMENTATION_TYPES: dict[str, type] = {
    "RandomMute": RandomMute,
    "RandomGain": RandomGain,
    "RandomCompress": RandomCompress,
    "RandomPitch": RandomPitch,
    "RandomCrop": RandomCrop,
}


def build_audio_augmentation_pipeline(
    augmentations_cfg: Any,
    *,
    n_signal: int,
    sampling_rate: int,
) -> Optional[Callable[[Any], Any]]:
    """Build an audio augmentation pipeline from config.

    Expected config format:
      data:
        augmentations:
          - type: RandomMute
            params: {prob: 0.2}

    Returns None if no augmentations are configured.
    """

    if not augmentations_cfg:
        return None

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        get_fn = getattr(obj, "get", None)
        if callable(get_fn):
            try:
                return get_fn(key, default)
            except TypeError:
                pass
        return getattr(obj, key, default)

    transforms = []
    for spec in augmentations_cfg:
        typ = _get(spec, "type", None) or _get(spec, "name", None)
        if not typ:
            raise ValueError("Augmentation entry must have 'type' (or 'name')")
        typ = str(typ)

        params = _get(spec, "params", None) or {}
        params = dict(params)

        ctor = _AUGMENTATION_TYPES.get(typ)
        if ctor is None:
            options = ", ".join(sorted(_AUGMENTATION_TYPES.keys()))
            raise KeyError(f"Unknown augmentation '{typ}'. Options: {options}")

        # Fill common defaults from runtime context
        if typ in {"RandomPitch", "RandomCrop"} and "n_signal" not in params:
            params["n_signal"] = int(n_signal)
        if typ in {"RandomCompress"} and "sr" not in params:
            params["sr"] = int(sampling_rate)

        transforms.append(ctor(**params))

    return Compose(transforms)
