#!/usr/bin/env python3
from torch.utils.data import DataLoader
# from src.data.dataset import AmphibDataset, sound_to_spectogramm, FILES_DIR
from pathlib import Path
from typing import Optional, List, Tuple
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import librosa
from abc import ABC, abstractmethod
from src import utils

# From the picutre it seems like stationary white noise
def analyse_noise(x: np.ndarray, 
                  sample_rate: int,
                  nperseg_factor: float = 0.2,
                  title: Optional[str] = None, 
                  plot=False, 
                  is_audio=True):
    # nperseq: num samples for Fast Fourier Transform to get different sequences
    # window to reduce spectral leakage: Spectral leakage occurs when sharp edges at segment boundaries introduce artificial 
    # frequencies in the Fast Fourier Transform; windowing smooths these edges to reduce the effect.
    nperseg = int(len(x) * nperseg_factor)
    f, Pxx = welch(x, fs=sample_rate, nperseg=nperseg)
    
    # Spectral Flatness
    Pxx = np.maximum(Pxx, 1e-20)
    spectral_flatness = np.exp(np.mean(np.log(Pxx))) / np.mean(Pxx)
    print(f"Spectral Flatness: {spectral_flatness:.3f}")

    if plot:
        ref = (20e-6)**2 if is_audio else 1.0  # 20 µPa² for audio, 1 V² else
        Pxx_db = 10 * np.log10(Pxx / ref)

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(f / 1000, Pxx_db)
        plt.xlabel("Frequenz (kHz)")
        plt.ylabel("PSD (dB re 20 µPa²/Hz)" if is_audio else "PSD (dB re 1 V²/Hz)")
        if title:
             plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class DenoiseMethod(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

class SpectralSubstraction(DenoiseMethod):
    def __init__(self,
        noise_signal: np.ndarray,
        n_fft: int = 2048, 
        alpha: float = 2.0, 
        beta: float = 0.01):
        """Robust spectral subtraction with proper noise handling.
        
        Args:
            noise_signal (np.array): Noise signal for estimation (if None, uses first 0.5s of x)
            n_fft (int): FFT size
            alpha (float): Over-subtraction factor (1.0-3.0)

        """
        self.noise_signal = noise_signal
        self.n_fft = n_fft
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x):
        X = librosa.stft(x, n_fft=self.n_fft)
        X_mag = np.abs(X)
        X_phase = np.angle(X)

        N = librosa.stft(self.noise_signal, n_fft=self.n_fft)
        N_mag = np.mean(np.abs(N), axis=1, keepdims=True)

        # Perform spectral subtraction with flooring
        clean_mag = np.maximum(X_mag - self.alpha * N_mag, self.beta * N_mag)
        clean_spectrum = clean_mag * np.exp(1.0j * X_phase)
        
        return librosa.istft(clean_spectrum)

class SpectralGate(DenoiseMethod):
    def __init__(self, sample_rate: int, noise_signal: Optional[np.ndarray] = None, stationary=False):
        self.noise_signal = noise_signal
        self.sample_rate = sample_rate
        self.stationary = stationary
    
    def __call__(self, x):
        return nr.reduce_noise(y=x, y_noise=self.noise_signal, sr=self.sample_rate, stationary=self.stationary)
    
class Wiener(DenoiseMethod):
    def __init__(self, noise_signal: np.ndarray, n_fft: int = 2048, noise_factor: float = 1.0):
        """
        Wiener filter for audio denoising.
        
        Args:
            noise_signal: Reference noise signal (noise-only segment)
            n_fft: FFT size (default: 2048)
            noise_factor: Over-subtraction factor (default: 1.0, higher = more aggressive noise reduction)
        """
        self.noise_signal = noise_signal
        self.n_fft = n_fft
        self.noise_factor = noise_factor
        
        # Pre-compute noise PSD (power spectral density)
        N = librosa.stft(noise_signal, n_fft=self.n_fft)
        self.mean_noise_psd = np.mean(np.abs(N)**2, axis=1, keepdims=True)  # Median over time frames

    def __call__(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filtering to input signal.
        
        Args:
            input_signal: Noisy input signal to process
            
        Returns:
            Denoised time-domain signal
        """
        # Compute STFT of input signal
        X = librosa.stft(input_signal, n_fft=self.n_fft)
        input_psd = np.abs(X)**2
        
        # Compute Wiener filter gain
        gain = input_psd / (input_psd + self.noise_factor * self.mean_noise_psd)
        gain = np.clip(gain, 0.1, 1.0)  # Clip gain to avoid artifacts
        
        # Apply gain to frequency components
        enhanced_stft = gain * X
        
        # Reconstruct time-domain signal
        return librosa.istft(enhanced_stft)

if __name__ == "__main__":
    pass
    # basic_noise_path = FILES_DIR / "basic_mic_noise_with_crickets.wav"
    # basic_noise, _ = librosa.load(basic_noise_path, sr=AmphibDataset.sample_rate)
    # analyse_noise(basic_noise, sample_rate=AmphibDataset.sample_rate, title="basic_mic_noise_with_crickets", plot=True)

    # froggy_file_path = FILES_DIR / "243B1F02648802FC_20250503_044100.WAV"
    # froggy, _ = librosa.load(froggy_file_path, sr=AmphibDataset.sample_rate)
    # analyse_noise(froggy, sample_rate=AmphibDataset.sample_rate, title="243B1F02648802FC_20250503_044100.WAV", plot=True)

    # spectral_sub = SpectralSubstraction(basic_noise, AmphibDataset.sample_rate)
    # x_spectral_sub = spectral_sub(froggy)
    # sf.write(FILES_DIR / "243B1F02648802FC_20250503_044100_spectral_sub.wav", x_spectral_sub, AmphibDataset.sample_rate)
    # analyse_noise(x_spectral_sub, sample_rate=AmphibDataset.sample_rate, title="243B1F02648802FC_20250503_044100_denoised_sub.WAV", plot=True)

    # spectral_gate = SpectralGate(sample_rate=AmphibDataset.sample_rate, stationary=False)
    # x_spectral_gate = spectral_gate(froggy)
    # spectral_gate = SpectralGate(sample_rate=AmphibDataset.sample_rate, stationary=True)
    # x_spectral_gate = spectral_gate(froggy)
    # sf.write(FILES_DIR / "243B1F02648802FC_20250503_044100_spectral_gate.wav", x_spectral_gate, AmphibDataset.sample_rate)
    # analyse_noise(x_spectral_gate, sample_rate=AmphibDataset.sample_rate, title="243B1F02648802FC_20250503_044100_denoised.WAV", plot=True)
    
    # wiener = Wiener(basic_noise, AmphibDataset.sample_rate)
    # x_wiener = wiener(froggy)
    # sf.write(FILES_DIR / "243B1F02648802FC_20250503_044100_wiener.wav", x_wiener, AmphibDataset.sample_rate)
    # analyse_noise(x_wiener, sample_rate=AmphibDataset.sample_rate, title="243B1F02648802FC_20250503_044100_denoised_sub.WAV", plot=True)

    # utils.plot_sound_waves([froggy, x_spectral_gate], in_one_ax=True)

    # path = "/media/marcel/3831-6261/"
    # dataset = AmphibDataset(path)
    # batch_size = 64
    # dataloader = DataLoader(dataset, batch_size=1)
    # for X_batch, paths_batch in dataloader:
    #         for x, path in zip(X_batch, paths_batch):
    #             x = x.numpy()
    #             filename = Path(path).stem
    #             img_path = FILES_DIR / "spectogramms" / (filename + ".png")
    #             img = sound_to_spectogramm(x,
    #                         model_optimized=False,
    #                         show=True)
    #             analyse_noise(x, sample_rate=AmphibDataset.sample_rate, title=filename, plot=True)
    #             x_denoised = denoise([(x, path)], sample_rate=AmphibDataset.sample_rate)[0][0]
    #             analyse_noise(x_denoised, sample_rate=AmphibDataset.sample_rate, title=f"{filename}_denoised", plot=True)
