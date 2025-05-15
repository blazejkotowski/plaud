import torch
import librosa
import essentia.standard as ess
import torch.nn.functional as F

class FeatureExtractor:
  N_FEATURES = 2

  def _init_(self):
    """
    Initializes the FeatureExtractor class.
    """
    pass

  def __call__(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Extracts features from the audio input.
    Arguments:
      - audio: torch.Tensor[batch_size, n_samples], the audio input
    Returns:
      - features: torch.Tensor[batch_size, n_samples], the extracted features
    """
    # roughness = self.roughness(audio)
    sharpness = self.sharpness(audio)
    spectral_centroid = self.spectral_centroid(audio)

    # Combine features into a single tensor
    features = torch.cat((sharpness, spectral_centroid), dim=1)
    return features


  def roughness(self, audio: torch.Tensor) -> torch.Tensor:
    w = ess.Windowing(type='hann')
    spectrum_func = ess.Spectrum()
    dissonance_func = ess.Dissonance()
    peaks_func = ess.SpectralPeaks()
    roughness_values = []

    for fram in ess.FrameGenerator(audio, frameSize = 2048, hopSize=512):
      windowed_frame = w(audio)
      frame_spectrum = spectrum_func(windowed_frame)
      frequencies, magnitudes = peaks_func(frame_spectrum)
      roughness = dissonance_func(frequencies, magnitudes)
      roughness_values.append(roughness)
    
    # roughness = torch.mean(torch.tensor(roughness_values), dim=0)
    roughness = F.interpolate(roughness.reshape(1, 1, -1), size=audio.shape[0], mode='linear')
    return roughness
  

  def sharpness(self, audio: torch.Tensor) -> torch.Tensor:
    stft = torch.abs(torch.tensor(librosa.stft(audio.cpu().numpy())))
    freqs = torch.tensor(librosa.fft_frequencies(sr=44100))
    weight = torch.pow(freqs/1000, 0.8)
    weighted_spectrum = stft * weight.unsqueeze(-1)
    energy = torch.sum(weighted_spectrum**2, dim=0)
    weighted_energy = torch.sum(weighted_spectrum**2, dim=0)
    sharpness_values = weighted_energy / (energy + 1e-10)
    # sharpness = torch.mean(sharpness_values, dim=0)
    sharpness = F.interpolate(sharpness_values.reshape(1, 1, -1), size=audio.shape[0], mode='linear')
    return sharpness
  

  def spectral_centroid(self, audio: torch.Tensor) -> torch.Tensor:
    sc = torch.tensor(librosa.feature.spectral_centroid(y=audio.cpu().numpy(), sr=44100))
    sc = F.interpolate(sc.reshape(1, 1, -1), size=audio.shape[0], mode='linear')
    return sc



    