#!/usr/bin/env python3
from torch.utils.data import DataLoader
from dataset import AmphibDataset, sound_to_image, FILES_DIR
from pathlib import Path
from typing import Optional, List, Tuple
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import librosa
from scipy.signal import stft, istft

# From the picutre it seems like stationary white noise
def analyse_noise(x: np.ndarray, sample_rate: int, title: Optional[str] = None, plot=False, is_audio=True):
    # nperseq: num samples for Fast Fourier Transform to get different sequences
    # window to reduce spectral leakage: Spectral leakage occurs when sharp edges at segment boundaries introduce artificial 
    # frequencies in the Fast Fourier Transform; windowing smooths these edges to reduce the effect.
    f, Pxx = welch(x, fs=sample_rate, nperseg=len(x) * 0.01)
    
    # Spectral Flatness
    spectral_flatness = np.exp(np.mean(np.log(Pxx + 1e-10))) / np.mean(Pxx)
    print(f"Spectral Flatness: {spectral_flatness:.3f}")

    if plot:
        ref = (20e-6)**2 if is_audio else 1.0  # 20 µPa² for audio, 1 V² else
        Pxx_db = 10 * np.log10(Pxx / ref + 1e-16)

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

def spectral_subtraction(x, noise, sample_rate, alpha=1.5, beta=0.1):
    alpha = 2.0 # scaling factor for substraction
    x_spectrum = np.mean(np.abs(librosa.stft(x)), axis=1)
    clean_spectrum = np.maximum()

def denoise(X: List[Tuple[np.ndarray, Path]], sample_rate: int, save_path: Optional[Path] = None):
    X_denoised = list()
    for x, path in X:
        x_denoised = nr.reduce_noise(y=x, sr=sample_rate)
        if save_path:
            new_filename = path.stem + '_denoised' + path.suffix
            sf.write(save_path / new_filename, x_denoised, subtype='PCM_16') # type: ignore
        X_denoised.append((x_denoised, path))

    return X_denoised

if __name__ == "__main__":
    basic_noise_path = FILES_DIR / "basic_mic_noise_with_crickets.wav"
    basic_noise, _ = librosa.load(basic_noise_path, sr=AmphibDataset.sample_rate)
    analyse_noise(basic_noise, sample_rate=AmphibDataset.sample_rate, title="basic_mic_noise_with_crickets", plot=True)


    good_file_path = FILES_DIR / "243B1F02648802FC_20250503_044100.WAV"
    good_file, _ = librosa.load(good_file_path, sr=AmphibDataset.sample_rate)
    analyse_noise(good_file, sample_rate=AmphibDataset.sample_rate, title="243B1F02648802FC_20250503_044100.WAV", plot=True)

#    x_denoised = spectral_subtraction(good_file, basic_noise, AmphibDataset.sample_rate)
    x_denoised = denoise([(good_file, Path(FILES_DIR / "243B1F02648802FC_20250503_044100_denoised2.wav"))], sample_rate=AmphibDataset.sample_rate)[0][0]
    sf.write(FILES_DIR / "243B1F02648802FC_20250503_044100_denoised.wav", x_denoised, AmphibDataset.sample_rate)
    analyse_noise(x_denoised, sample_rate=AmphibDataset.sample_rate, title="243B1F02648802FC_20250503_044100.WAV_denoised", plot=True)
    
    # path = "/media/marcel/3831-6261/"
    # dataset = AmphibDataset(path)
    # batch_size = 64
    # dataloader = DataLoader(dataset, batch_size=1)
    # for X_batch, paths_batch in dataloader:
    #         for x, path in zip(X_batch, paths_batch):
    #             x = x.numpy()
    #             filename = Path(path).stem
    #             img_path = FILES_DIR / "spectogramms" / (filename + ".png")
    #             img = sound_to_image(x,
    #                         model_optimized=False,
    #                         show=True)
    #             analyse_noise(x, sample_rate=AmphibDataset.sample_rate, title=filename, plot=True)
    #             x_denoised = denoise([(x, path)], sample_rate=AmphibDataset.sample_rate)[0][0]
    #             analyse_noise(x_denoised, sample_rate=AmphibDataset.sample_rate, title=f"{filename}_denoised", plot=True)
