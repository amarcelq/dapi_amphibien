#!/usr/bin/env python3
from torch.utils.data import Dataset
from pathlib import Path
import librosa
from torch.utils.data import DataLoader
import noisereduce as nr
import soundfile as sf
import numpy as np
from typing import Optional, List, Tuple

OUTPUT_DIR = Path("src") / "data" / "denoised"
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

if __name__ == "__main__":
    dataset = FrogDataset("/media/marcel/3831-6261/")
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: denoise(b, save_path=OUTPUT_DIR))
    for X_batch in dataloader:
        print(X_batch)
