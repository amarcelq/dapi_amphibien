import librosa
import numpy as np
from abc import ABC, abstractmethod

class FeatureExtractMethod(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

class MFCC(FeatureExtractMethod):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    def __call__(self, x):
        return librosa.feature.mfcc(y=x, sr=self.sample_rate)

class SpectralFeature(FeatureExtractMethod):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    def __call__(self, x):
            contrast = librosa.feature.spectral_contrast(y=x, sr=self.sample_rate)
            centroid = librosa.feature.spectral_centroid(y=x, sr=self.sample_rate)
            bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=self.sample_rate)
            rolloff = librosa.feature.spectral_rolloff(y=x, sr=self.sample_rate)
            return np.vstack([contrast, centroid, bandwidth, rolloff])

class Chroma(FeatureExtractMethod):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    def __call__(self, x):
        return librosa.feature.chroma_stft(y=x, sr=self.sample_rate)