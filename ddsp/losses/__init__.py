# from .m2l_loss import M2LLoss
# from .clap_loss import CLAPLoss
from .attribute_regularization_loss import AttributeRegularizationLoss
from .sliced_wasserstein_loss import SlicedWassersteinLoss
from .multi_scale_sliced_wasserstein_loss import MultiScaleSlicedWassersteinLoss

# Local registration of losses into the global registry
from ddsp.registry import LOSSES

LOSSES.add("AttributeRegularizationLoss", AttributeRegularizationLoss)
LOSSES.add("SlicedWassersteinLoss", SlicedWassersteinLoss)
LOSSES.add("MultiScaleSlicedWassersteinLoss", MultiScaleSlicedWassersteinLoss)
