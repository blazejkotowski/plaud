import warnings

# This module is deprecated. Use `ddsp.prior.latents_dataset_builder` instead.
warnings.warn(
    "ddsp.prior.exporter is deprecated; import from ddsp.prior.latents_dataset_builder",
    DeprecationWarning,
)

from ddsp.prior.latents_dataset_builder import export_latents  # re-export for backward compatibility

__all__ = ["export_latents"]
