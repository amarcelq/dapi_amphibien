#!/usr/bin/env python3
from torch.utils.data import Dataset
from pathlib import Path
import librosa
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.basic_preprocessing import BasicPreprocessor
from preprocessing.denoise import DenoiseMethod, SpectralGate 
from typing import Optional, Tuple, Sequence
import cv2
from tqdm import tqdm
import atexit
import shutil
import os
import torch
import random
from pydub import AudioSegment

FILES_DIR = Path(__file__).resolve().parent.parent / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

class AmphibDataset(Dataset):
    sample_rate: int = 192000

    def __init__(self,
                 parent_path: Optional[str] | Optional[Path] = None, 
                 basic_preprocessor: Optional[BasicPreprocessor] = None, 
                 denoiser: Optional[DenoiseMethod] = None,
                 file_paths: Optional[str | Path | Sequence[str] | Sequence[Path]] = None,
                 return_paths: bool = True):
        self.file_paths = []

        self.return_paths = return_paths

        if file_paths:
            _file_paths: Sequence[str | Path] = [file_paths] if isinstance(file_paths, (str, Path)) else file_paths
            self.file_paths = [Path(f).with_name(Path(f).stem + Path(f).suffix.lower()) for f in _file_paths]
        
        if parent_path:
            parent_path = Path(parent_path) if isinstance(parent_path, str) else parent_path
            if not parent_path.exists():
                raise FileNotFoundError(f"parent_path: {str(parent_path)} does not exist.")
            all_paths = list(parent_path.rglob("*.WAV"))
            if len(all_paths) == 0:
                return
            file_sizes = np.array([p.stat().st_size for p in all_paths])

            # drop all files which dont have the same size as the other to have a dataset of equal len arrays
            median_size = np.median(file_sizes)
            for path, size in zip(all_paths, file_sizes):
                if size == median_size:
                    self.file_paths.append(path.with_name(path.stem + path.suffix.lower()))

        self.basic_preprocessor = basic_preprocessor
        self.denoiser = denoiser
        self.cache_dir = FILES_DIR / "cache_denoised"
        self.cache_dir.mkdir(exist_ok=True)

        atexit.register(self.cleanup_cache)

    def cleanup_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        cache_path = self.cache_dir / path.name

        if cache_path.exists():
            x = np.load(cache_path.with_suffix(".npz"))["x"]
        else:
            x, _ = librosa.load(path, sr=self.sample_rate)

            if self.denoiser:
                x = self.denoiser(x)
            if self.basic_preprocessor:
                x = self.basic_preprocessor(x)
            np.savez_compressed(cache_path.with_suffix(".npz"), x=x)

        # need to use str instead of path since dataloader dont't support paths
        if self.return_paths:
            return x, str(path)
        else:
            return x

    @staticmethod
    def process_single_x(path, sample_rate, 
                         denoiser: Optional[DenoiseMethod] = None, 
                         basic_preprocessor: Optional[BasicPreprocessor] = None):
        x, _ = librosa.load(path, sr=sample_rate)

        if denoiser:
            x = denoiser(x)
        if basic_preprocessor:
            x = basic_preprocessor(x)
        
        return x

def sound_to_spectogramm(x: np.ndarray, 
                   save_folder_path: Optional[Path] = None, 
                   save_file_name: Optional[str | Path] = None, 
                   show: bool = False, 
                   model_optimized: bool = False, dpi: int = 200):

    if (save_folder_path and not save_file_name) or (not save_folder_path and save_file_name):
        raise ValueError(f"""You have to set save_folder_path set {save_folder_path} 
                         and save_file_name set {save_file_name}""")

    # Compute the spectrogram with Short-time Fourier Transform
    db_measures = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    # normalize the picture to range [0-255]
    img = (255 * (db_measures - db_measures.min()) / (db_measures.max() - db_measures.min())).astype(np.uint8)
    # colorize the specogramm
    img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
    # resize all pictures to 224x224 since infamouse models like resnet use this as input
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    # no addtional visual elements
    if show:
        if model_optimized:
                cv2.imshow("Spectrogram", img)
        elif not model_optimized:
            plt.figure(figsize=(224 / dpi, 224 / dpi), frameon=False)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

    if save_folder_path and save_file_name:
        save_folder_path.mkdir(parents=True, exist_ok=True)
        save_path = save_folder_path / (str(save_file_name))
        if model_optimized:
            cv2.imwrite(str(save_path) + ".png", img)
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.close()

    return img

def load_image(path: str | Path, output_size: Optional[Tuple[int, int]] = None):
    img = cv2.imread(str(path))
    if output_size:
        # bilinear interpolation
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)

    return img

class KaggleAnuranSoundDataset(Dataset):
    def __init__(self, parent_dir, resample_rate: int = 8000):
        self.parent_dir = parent_dir
        self.resample_rate = resample_rate

        self.cache_dir = FILES_DIR / "cache_kaggle"
        self.cache_dir.mkdir(exist_ok=True)
        atexit.register(self.cleanup_cache)

        self.file_paths = []

        for subdir, _, files in os.walk(parent_dir):
            for file in files:
                if file.endswith(".m4a"):
                    path = Path(subdir) / file
                    self.file_paths.append(path)

    def cleanup_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        cache_path = self.cache_dir / path.name

        if cache_path.exists():
            x = np.load(cache_path.with_suffix(".npz"))["x"]
        else:
            sound = AudioSegment.from_file(path, format='m4a')
            file = sound.export("hallo", format='wav')
            x, _ = librosa.load(file, sr=self.resample_rate)

            np.savez_compressed(cache_path.with_suffix(".npz"), x=x)

        return x

class MixedAudioDataset(Dataset):
    def __init__(self, kaggle_dataset, youtube_folder, target_len: int = 3, sample_rate=8000):
        self.kaggle_dataset = kaggle_dataset
        self.youtube_folder = youtube_folder
        self.sample_rate = sample_rate
        self.target_len = int(self.sample_rate * target_len) 
        self.youtube_files = [f for f in os.listdir(youtube_folder) if f.endswith('.wav')]

    def __len__(self):
        return len(self.kaggle_dataset)

    def __getitem__(self, idx):
        kaggle_x = self.kaggle_dataset[idx]
        kaggle_x = torch.tensor(kaggle_x).float().squeeze()
        if kaggle_x.shape[-1] > self.target_len:
            start = random.randint(0, kaggle_x.shape[-1] - self.target_len)
            kaggle_x = kaggle_x[start:start + self.target_len]
        else:
            pad_size = self.target_len - kaggle_x.shape[-1]
            kaggle_x = torch.nn.functional.pad(kaggle_x, (0, pad_size))

        # just add youtube_x to 80% to make the model more robust
        add_noise = random.random() < 0.8
        if add_noise:
            chosen_file = random.choice(self.youtube_files)
            yt_waveform, _ = librosa.load(os.path.join(self.youtube_folder, chosen_file), sr=self.sample_rate)
            yt_waveform = torch.tensor(yt_waveform).float().squeeze()
            if yt_waveform.shape[-1] > self.target_len:
                start = random.randint(0, yt_waveform.shape[-1] - self.target_len)
                youtube_x = yt_waveform[start:start + self.target_len]
            else:
                # little bit whitenoise instead of complete silence to make it more robust
                noise = torch.randn(self.target_len) * 0.01
                youtube_x = noise
        else:
            youtube_x = torch.zeros(self.target_len)

        kaggle_x = kaggle_x.unsqueeze(0)
        youtube_x = youtube_x.unsqueeze(0)
        mixture = kaggle_x + youtube_x

        return kaggle_x, youtube_x, mixture



if __name__ == "__main__":
    path = "/media/marcel/3831-6261/"
    basic_noise_path = FILES_DIR / "basic_mic_noise_with_crickets.wav"
    basic_noise, _ = librosa.load(basic_noise_path, sr=AmphibDataset.sample_rate)
    dataset = AmphibDataset(path, 
                            BasicPreprocessor(sample_rate=AmphibDataset.sample_rate), 
                            SpectralGate(sample_rate=AmphibDataset.sample_rate, noise_signal=basic_noise))
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for X_batch, paths_batch in tqdm(dataloader):
            for x, path in zip(X_batch, paths_batch):
                print(x)
                # img_path = FILES_DIR / "spectogramms" / (Path(path).stem + ".png")
                # img = sound_to_spectogramm(x.numpy(), 
                #             save_folder_path=FILES_DIR / "spectogramms",
                #             save_file_name=Path(path).stem,
                #             model_optimized=True)
