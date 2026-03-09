from .synths import BaseSynth, SubbandSineSynth, SineSynth, NoiseBandSynth, HarmonicSynth, BendableNoiseBandSynth
from .spectral_sine_synth import SpectralSineSynth
from .complex_sine_synth import ComplexSineSynth
from .cumprod_sine_synth import CumprodSineSynth
from .simple_sine_synth import SimpleSineSynth

# Local registration of synths into the global registry
from ddsp.registry import SYNTHS

SYNTHS.add("SineSynth", SineSynth)
SYNTHS.add("NoiseBandSynth", NoiseBandSynth)
SYNTHS.add("BendableNoiseBandSynth", BendableNoiseBandSynth)
SYNTHS.add("SubbandSineSynth", SubbandSineSynth)
SYNTHS.add("HarmonicSynth", HarmonicSynth)
SYNTHS.add("SpectralSineSynth", SpectralSineSynth)
SYNTHS.add("ComplexSineSynth", ComplexSineSynth)
SYNTHS.add("CumprodSineSynth", CumprodSineSynth)
SYNTHS.add("SimpleSineSynth", SimpleSineSynth)
