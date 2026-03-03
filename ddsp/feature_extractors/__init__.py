from .base_extractor import BaseExtractor
from .librosa_feature_extractor import LibrosaFeatureExtractor
from .crepe_pitch_extractor import CrepePitchExtractor
from .pyin_pitch_extractor import PYINPitchExtractor

# Local registration of feature extractors into the global registry
from ddsp.registry import FEATURE_EXTRACTORS

FEATURE_EXTRACTORS.add("LibrosaFeatureExtractor", LibrosaFeatureExtractor)
FEATURE_EXTRACTORS.add("CrepePitchExtractor", CrepePitchExtractor)
FEATURE_EXTRACTORS.add("PYINPitchExtractor", PYINPitchExtractor)
