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
import tempfile
import subprocess

FILES_DIR = Path(__file__).resolve().parent.parent / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

class AmphibDataset(Dataset):
    """
    Loads and preprocesses local WAV files of amphibian sounds, applies optional denoising and caching for efficiency.
    """
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
    """
    Loads frog/toad audio files from the Kaggle anuran dataset, converts from .m4a to WAV format with caching.

    """
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
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
                subprocess.run(['ffmpeg', '-y', '-i', path, tmpfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                x, _ = librosa.load(tmpfile.name, sr=self.resample_rate)
                x = librosa.util.normalize(x)

            np.savez_compressed(cache_path.with_suffix(".npz"), x=x)

        return torch.Tensor(x)

class YouTubeNoiseDataset(Dataset):
    """
    Loads long ambient noise segments from YouTube sound recordings, extracts random fixed-length samples for background noise modeling.
    """
    def __init__(self, youtube_folder, target_len=3, sample_rate=8000, generate_n_samples=1500):
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * target_len)
        self.data = []
        max_segment_len = 10 * 60 * sample_rate # just load 10 minutes to make it more efficent
        self.generate_n_samples = generate_n_samples
        for f in os.listdir(youtube_folder):
            if not f.endswith('.wav'):
                continue
            path = os.path.join(youtube_folder, f)
            duration = librosa.get_duration(path=path)
            total_samples = int(duration * sample_rate)

            if total_samples > max_segment_len * 2:
                # Load first 10 minutes
                start_segment, _ = librosa.load(path, sr=sample_rate, offset=0, duration=10*60)
                start_segment = torch.tensor(start_segment).float()
                # Load last 10 minutes
                offset_last = duration - 10*60
                end_segment, _ = librosa.load(path, sr=sample_rate, offset=offset_last, duration=10*60)
                end_segment = torch.tensor(end_segment).float()
                self.data.append((start_segment, end_segment))
            else:
                waveform, _ = librosa.load(path, sr=sample_rate)
                waveform = librosa.util.normalize(waveform)
                waveform = torch.tensor(waveform).float()
                self.data.append((waveform,))

    def __len__(self):
        return self.generate_n_samples

    def __getitem__(self, _):
        """Returns a random sample from """
        segments = random.choice(self.data) # choose random file
        segment = random.choice(segments)  # chose either starting or ending 10 minutes
        if segment.shape[-1] > self.target_len:
            start = random.randint(0, segment.shape[-1] - self.target_len)
            sample = segment[start:start+self.target_len]
        else:
            # pads with almost silence if the sequence is shorter than the target size
            pad_value = 10 ** (-35 / 20)        
            pad_size = self.target_len - segment.shape[-1]
            sample = torch.nn.functional.pad(segment, (pad_value, pad_size))
        return sample

class MixedAudioDataset(Dataset):
    """
    Generates mixtures by combining frog sounds from Kaggle dataset with YouTube background noises for training source separation.
    """
    def __init__(self, kaggle_dataset, youtube_dataset: YouTubeNoiseDataset, target_len: int = 3, sample_rate=8000, add_prob: float = 0.8):
        self.kaggle_dataset = kaggle_dataset
        self.youtube_dataset = youtube_dataset
        self.sample_rate = sample_rate
        self.target_len = int(self.sample_rate * target_len)
        self.add_prob = add_prob

    def __len__(self):
        return len(self.kaggle_dataset)

    def __getitem__(self, idx):
        kaggle_x = torch.tensor(self.kaggle_dataset[idx]).float().squeeze()
        if kaggle_x.shape[-1] > self.target_len:
            start = random.randint(0, kaggle_x.shape[-1] - self.target_len)
            kaggle_x = kaggle_x[start:start + self.target_len]
        else:
            pad_size = self.target_len - kaggle_x.shape[-1]
            kaggle_x = torch.nn.functional.pad(kaggle_x, (0, pad_size))

        if random.random() < self.add_prob:
            youtube_x = self.youtube_dataset[random.randint(0, len(self.youtube_dataset) - 1)].unsqueeze(0)
        else:
            youtube_x = torch.randn(1, self.target_len) * 0.01

        kaggle_x = kaggle_x.unsqueeze(0)
        mixture = kaggle_x + youtube_x

        return kaggle_x, youtube_x, mixture

if __name__ == "__main__":
    path = FILES_DIR / "frog_sounds"
    basic_noise_path = FILES_DIR / "basic_noise.wav"
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
