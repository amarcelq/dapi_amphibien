import numpy as np
import librosa
import pyloudnorm as pyln
from typing import Optional, Literal, Dict

class BasicPreprocessor:
    def __init__(self,
                sample_rate: int,
                mono: bool = False,
                normalize: Optional[Literal["range", "lufs"]] = "range",
                normalize_options: Optional[Dict] = None,
                trim: bool = True,
                trim_options: Optional[Dict] = None):
        self.mono = mono
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.normalize_options = normalize_options or {}
        self.trim = trim
        self.trim_options = trim_options or {}

    @staticmethod
    def convert_to_mono(x: np.ndarray) -> np.ndarray:
        # from stereo (sound different from left and right) to same sound from left and right speaker
        return librosa.to_mono(x) if x.ndim > 1 else x

    @staticmethod
    def normalize_range(x: np.ndarray, range_: list = [-1, 1]) -> np.ndarray:
        x_min, x_max = np.min(x), np.max(x)
        scale_min, scale_max = range_
        x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else x - x_min
        return x_norm * (scale_max - scale_min) + scale_min

    @staticmethod
    def normalize_to_lufs(x: np.ndarray, sample_rate: int, target_lufs: float = -23.0) -> np.ndarray:
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(x)
        x_normalized = pyln.normalize.loudness(x, loudness, target_lufs)
        return x_normalized

    @staticmethod
    def trim_silence(x: np.ndarray, top_db: int = 65) -> np.ndarray:
        x_trimmed, _ = librosa.effects.trim(x, top_db=top_db)
        return x_trimmed

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.mono:
            x = self.convert_to_mono(x)

        if self.normalize == "range":
            x = self.normalize_range(x, **self.normalize_options)
        elif self.normalize == "lufs":
            x = self.normalize_to_lufs(x, self.sample_rate, **self.normalize_options)

        if self.trim:
            x = self.trim_silence(x, **self.trim_options)

        return x