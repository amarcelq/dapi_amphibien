#!/usr/bin/env python3
from torch.utils.data import Dataset
from pathlib import Path
import librosa
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import cv2

FILES_DIR = Path(__file__).resolve().parent / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

class AmphibDataset(Dataset):
    sample_rate: int = 192000

    def __init__(self, parent_path_str: str):
        self.file_paths = list()
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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        x, _ = librosa.load(path, sr=self.sample_rate)

        return x, str(path)

def sound_to_image(x: np.ndarray, 
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
    dataset = AmphibDataset(path)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size)#, collate_fn=denoise)
    for X_batch, paths_batch in dataloader:
            for x, path in zip(X_batch, paths_batch):
                img_path = FILES_DIR / "spectogramms" / (Path(path).stem + ".png")
                img = sound_to_image(x.numpy(), 
                            save_folder_path=FILES_DIR / "spectogramms",
                            save_file_name=Path(path).stem,
                            model_optimized=True)
