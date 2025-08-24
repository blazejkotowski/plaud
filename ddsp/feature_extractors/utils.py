import torch
import torch.nn.functional as F
import math

# def normalize_feature(feature: torch.Tensor) -> torch.Tensor:
#   """Normalizes the feature to the range [0, 1]."""
#   q_low = feature.cpu().quantile(0.1, dim=-1)
#   q_high = feature.cpu().quantile(0.9, dim=-1)
#   feature = torch.clamp(feature, q_low, q_high)
#   return (feature - feature.min()) / (feature.max() - feature.min())

def _percentile_kth(x: torch.Tensor, q: float, dim: int = -1):
  # q in [0, 100]
  n = x.size(dim)
  # nearest-rank index in [1, n]
  k = max(1, min(n, int(math.ceil(q / 100.0 * n))))
  # kthvalue returns values with dim reduced by 1
  v = x.kthvalue(k, dim=dim).values
  return v.unsqueeze(dim)  # match keepdim=True

def normalize_feature(feature: torch.Tensor, dim: int = -1, low: float = 5.0, high: float = 95.0) -> torch.Tensor:
    """
    Normalizes the feature to the range [0, 1] based on low and high percentiles.
    Args:
      - feature: torch.Tensor, the feature tensor to normalize
      - dim: int, the dimension along which to normalize
      - low: float, the low percentile to use for normalization
      - high: float, the high percentile to use for normalization
    Returns:
      - torch.Tensor, the normalized feature tensor
    """
    lo = _percentile_kth(feature, low,  dim=dim)
    hi = _percentile_kth(feature, high, dim=dim)
    clamped = torch.clamp(feature, lo, hi)
    return (clamped - lo) / (hi - lo + 1e-8)


# def smoothen_feature(feature: torch.Tensor, window_size: int = 256+1) -> torch.Tensor:
#   """Smoothens the feature using a simple moving average."""
#   return F.avg_pool1d(feature, kernel_size=window_size, stride=1, padding=window_size // 2)


def smoothen_feature(x: torch.Tensor, window_size: int = 257) -> torch.Tensor:
    # x: [1, T]  (causal moving average)
    w = x.new_full((1, 1, window_size), 1.0 / window_size)
    y = F.conv1d(F.pad(x.unsqueeze(1), (window_size - 1, 0)), w)
    return y.squeeze(1)  # [1, T]


def postprocess_feature(feature: torch.Tensor,) -> torch.Tensor:
  """Processes the feature by normalizing and smoothening it."""
  feature = smoothen_feature(feature)
  feature = normalize_feature(feature)
  return feature
