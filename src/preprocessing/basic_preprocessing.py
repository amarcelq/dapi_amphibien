#!/usr/bin/env python3
import numpy as np
import librosa
import pyloudnorm as pyln
from typing import Optional, Literal, Dict, Iterable

class BasicPreprocessor:
    def __init__(self,
                sample_rate: int,
                add_freq_dim: Optional[Iterable] = (-1, 1, -1),
                parts_len: Optional[int | float] = None,
                mono: bool = False,
                normalize: Optional[Literal["range", "lufs"]] = "range",
                normalize_options: Optional[Dict] = None,
                trim: bool = True,
                trim_options: Optional[Dict] = None,
                resample_rate: Optional[int] = None):

        self.mono = mono
        self.add_freq_dim = add_freq_dim
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.normalize_options = normalize_options or {}
        self.trim = trim
        self.trim_options = trim_options or {}
        self.parts_len = parts_len
        self.resample_rate = resample_rate

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
    def split(x: np.ndarray, parts_len: int | float, sample_rate: int, drop_last = True):
        segment_length = int(sample_rate * parts_len)
        x_parts = list()

        num_parts = int(len(x) // segment_length)
        for i in range(num_parts):
            start = i * segment_length
            end = start + segment_length
            x_parts.append(x[start:end])

        if not drop_last and len(x) % segment_length != 0:
            x_parts.append(x[num_parts * segment_length:])

        return np.array(x_parts)

    @staticmethod
    def trim_silence(x: np.ndarray, top_db: int = 65) -> np.ndarray:
        x_trimmed, _ = librosa.effects.trim(x, top_db=top_db)
        return x_trimmed
    
    @staticmethod
    def resample(x: np.ndarray, sample_rate: int, resample_rate: int):
        return librosa.resample(x, orig_sr=sample_rate, target_sr=resample_rate)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.mono:
            x = self.convert_to_mono(x)

        if self.normalize == "range":
            x = self.normalize_range(x, **self.normalize_options)
        elif self.normalize == "lufs":
            x = self.normalize_to_lufs(x, self.sample_rate, **self.normalize_options)

        if self.trim:
            x = self.trim_silence(x, **self.trim_options)

        if self.resample_rate:
            x = self.resample(x, self.sample_rate, self.resample_rate)

        if self.parts_len:
            x = self.split(x, parts_len=self.parts_len, sample_rate=self.sample_rate)
    
        if self.add_freq_dim:
            x = np.expand_dims(x, axis=1)

        return x
    
