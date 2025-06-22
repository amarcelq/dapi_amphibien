import librosa
import numpy as np
from abc import ABC, abstractmethod
import openl3

class FeatureExtractMethod(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

class MFCC(FeatureExtractMethod):
    def __init__(self, sample_rate: int, n_mels: int, mono: bool = True):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mono = mono
    
    def __call__(self, x):
        x = librosa.feature.mfcc(y=x, sr=self.sample_rate, n_mels=self.n_mels)
        # squeeze freq
        if self.mono and x.ndim == 4:
            x = x.squeeze(1)

        # flatten features
        return x.reshape(*x.shape[:-2], -1)

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
    def __init__(self, sample_rate: int, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono

    def __call__(self, x: np.array):
        x = librosa.feature.chroma_stft(y=x, sr=self.sample_rate)
        # squeeze freq
        if self.mono and x.ndim == 4:
            x = x.squeeze(1)

        # flatten features
        return x.reshape(*x.shape[:-2], -1)

class OpenL3Embedding(FeatureExtractMethod):
    def __init__(self, sample_rate: int, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono

    def __call__(self, x):
        x, _ = openl3.get_audio_embedding(x, self.sample_rate, content_type="env")
        # openl3 returns a list of np arrays
        x = np.array(x)
        return x