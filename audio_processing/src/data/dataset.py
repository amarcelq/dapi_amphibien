#!/usr/bin/env python3
from torch.utils.data import Dataset
from pathlib import Path
import librosa
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.basic_preprocessing import BasicPreprocessor
from preprocessing.denoise import DenoiseMethod, SpectralGate 
from typing import Optional, List, Tuple, Sequence
import cv2
from tqdm import tqdm
import soundfile as sf
import atexit
import shutil

FILES_DIR = Path(__file__).resolve().parent.parent / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

class AmphibDataset(Dataset):
    sample_rate: int = 192000

    def __init__(self,
                 parent_path_str: Optional[str] = None, 
                 basic_preprocessor: Optional[BasicPreprocessor] = None, 
                 denoiser: Optional[DenoiseMethod] = None,
                 file_paths: Optional[str | Path | Sequence[str] | Sequence[Path]] = None,
                 return_paths: bool = True):
        self.file_paths = []

        self.return_paths = return_paths

        if file_paths:
            if isinstance(file_paths, (str, Path)):
                file_paths = [file_paths]
            self.file_paths = [Path(f).with_name(Path(f).stem + Path(f).suffix.lower()) for f in file_paths]
        
        if parent_path_str:
            parent_path = Path(parent_path_str)
            if not parent_path.exists():
                raise FileNotFoundError(f"parent_path: {parent_path_str} does not exist.")
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
