#!/usr/bin/env python3
from torch.utils.data import Dataset
from pathlib import Path
import librosa
from torch.utils.data import DataLoader
import noisereduce as nr
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import cv2

OUTPUT_DIR = Path("src") / "data" / "files" / "denoised"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class FrogDataset(Dataset):
    def __init__(self, parent_path: str):
        all_paths = list(Path(parent_path).rglob("*.WAV"))
        if len(all_paths) == 0:
            return
        file_sizes = np.array([p.stat().st_size for p in all_paths])

        self.file_paths = list()

        # drop all files which dont have the same size as the other to have a dataset of equal len arrays
        median_size = np.median(file_sizes)
        for path, size in zip(all_paths, file_sizes):
            if size == median_size:
                self.file_paths.append(path.with_name(path.stem + path.suffix.lower()))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        x, sample_rate = librosa.load(path)

        return x, sample_rate, str(path)

def denoise(X: List[Tuple[np.ndarray, int, str]], save_path: Optional[Path] = None):
    X_denoised = list()
    for x, sample_rate, path in X:
        x_denoised = nr.reduce_noise(y=x, sr=sample_rate)
        if save_path:
            origial_path = Path(path)
            new_filename = origial_path.stem + '_denoised' + origial_path.suffix
            sf.write(save_path / new_filename, x_denoised, sample_rate, subtype='PCM_16') # type: ignore
        X_denoised.append((x_denoised, sample_rate, path))

    return X_denoised

def sound_to_image(x: np.ndarray, save_folder_path: Path, save_file_name: str | Path, show: bool = False, model_optimized: bool = False, dpi: int = 200):
    # Compute the spectrogram with Short-time Fourier Transform
    db_measures = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    # normalize the picture to range [0-255]
    img = (255 * (db_measures - db_measures.min()) / (db_measures.max() - db_measures.min())).astype(np.uint8)
    # colorize the specogramm
    #img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
    # resize all pictures to 224x224 since infamouse models like resnet use this as input
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    save_folder_path.mkdir(parents=True, exist_ok=True)
    save_path = save_folder_path / (str(save_file_name))
    # no addtional visual elements
    if model_optimized:
        cv2.imwrite(str(save_path) + ".png", img)
        if show:
            cv2.imshow("Spectrogram", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        plt.figure(figsize=(224 / dpi, 224 / dpi), frameon=False)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        img = load_image(save_path)
        if show:
            plt.show()
        plt.close()
    
    return img

def load_image(path: str | Path, output_size: Optional[Tuple[int, int]] = None):
    img = cv2.imread(str(path))
    if output_size:
        # bilinear interpolation
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)

    return img

if __name__ == "__main__":
    dataset = FrogDataset("/media/marcel/3831-6261/")
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=denoise)
    for X_batch in dataloader:
        for x, sr, p in X_batch:
            img_path = OUTPUT_DIR / "spectogramms" / (Path(p).stem + ".png")
            img = sound_to_image(x, 
                           save_folder_path=OUTPUT_DIR / "spectogramms",
                           save_file_name=Path(p).stem,
                           model_optimized=True)
