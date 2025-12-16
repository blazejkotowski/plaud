from .base_extractor import BaseExtractor
from .librosa_feature_extractor import LibrosaFeatureExtractor

# Local registration of feature extractors into the global registry
from ddsp.registry import FEATURE_EXTRACTORS

FEATURE_EXTRACTORS.add("LibrosaFeatureExtractor", LibrosaFeatureExtractor)
