#!/usr/bin/env python3
from data.dataset import AmphibDataset, FILES_DIR, KaggleAnuranSoundDataset, MixedAudioDataset
from preprocessing.denoise import SpectralGate, DenoiseMethod
from preprocessing.basic_preprocessing import BasicPreprocessor
from torch.utils.data import DataLoader
from sound_seperation.clustering import BGMM, ClusteringMethod
from sound_seperation.feature_extraction import MFCC, SpectralFeature, Chroma, OpenL3Embedding, FeatureExtractMethod
from sound_seperation.feature_reduction import PCA, FeatureReductionMethod
from sound_seperation.sound_speration import ConvTas, SoundSperationMethod

from typing import Optional, Sequence
from tqdm import tqdm
import librosa
from collections import defaultdict
import soundfile as sf
import os
from pathlib import Path
import numpy as np
import openl3
import kagglehub
import torch

TRAIN: bool = False
SAMPLE_RATE: int = 8000

def create_datasets(data_path: Sequence[str] | Sequence[Path], denoiser: Optional[DenoiseMethod] = None, basic_preprocessor: Optional[BasicPreprocessor] = None):
    if isinstance(data_path, str) or isinstance(data_path, Path):
        dataset = AmphibDataset(parent_path=data_path, basic_preprocessor=basic_preprocessor, denoiser=denoiser)
    else:
        dataset = AmphibDataset(file_paths=data_path, basic_preprocessor=basic_preprocessor, denoiser=denoiser)

    kaggle_path = kagglehub.dataset_download("mehmetbayin/anuran-sound-frogs-or-toads-dataset")
    kaggle_dataset = KaggleAnuranSoundDataset(kaggle_path, AmphibDataset.sample_rate)
    mixture_dataset = MixedAudioDataset(kaggle_dataset, FILES_DIR / "youtube_sounds", target_len=5, sample_rate=AmphibDataset.sample_rate)

    return dataset, kaggle_dataset, mixture_dataset

def get_frog_cluster(x: torch.Tensor):
    # TODO: Combine and Mean some frog sounds and lay them in the middle of both clusters and take the one with lesser distance
    pass

def predict_cluster(x: np.ndarray,
                    clusterer: ClusteringMethod,
                    sound_sperator: Optional[SoundSperationMethod] = None,
                    feature_extractior: Optional[FeatureExtractMethod] = None,
                    feature_reductor: Optional[FeatureReductionMethod] = None,
                    ) -> dict:
    
    features: np.ndarray | torch.Tensor = x
    if sound_sperator:
        _seperated = sound_sperator.pred(features)
        # TODO: Check if _spererated is really from same range as orignial x
        # TODO: Implement this function!
        x = _seperated[0, 0, :44000].reshape(-1, 1)#get_frog_cluster(_seperated)
        features = x

    if feature_extractior:
        features = feature_extractior(features)

    if feature_reductor:
        features = feature_reductor(features)

    labels = clusterer(features)

    unique_labels = set(labels)
    clustered: dict = {label: [] for label in unique_labels}
    for idx, label in enumerate(labels):
        clustered[label].append(x[idx])

    return clustered

def save_cluster_to_file(clustered: dict, output_dir: Path | str, file_name: str | Path):
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    for cluster, data in clustered.items():
        output_path = output_dir  /  (file_name + f"_{cluster}" + ".wav")
        sf.write(output_path, data, SAMPLE_RATE)

def main(x_path: Optional[Path | str] = None) -> None:
    basic_noise_path = FILES_DIR / "basic_mic_noise_with_crickets.wav"
    basic_noise, _ = librosa.load(basic_noise_path, sr=AmphibDataset.sample_rate)
    spectral_gate = SpectralGate(AmphibDataset.sample_rate, noise_signal=basic_noise, stationary=True)
    basic_preprocessor = BasicPreprocessor(AmphibDataset.sample_rate, parts_len=8, add_freq_dim=None, resample_rate=SAMPLE_RATE)
    AmphibDataset.sample_rate = SAMPLE_RATE
    sound_seperator = ConvTas(num_sources=2, sample_rate=SAMPLE_RATE)
    # TODO: Finish Implementation OpenL3Embedding
    #feature_extractior = OpenL3Embedding()
    feature_reductor = PCA(n_dims=2)
    clusterer = BGMM(n_clusters=5)

    if not x_path:
        dataset, kaggle_dataset, mixture_dataset = create_datasets(data_path="/media/marcel/3831-6261", denoiser=spectral_gate, basic_preprocessor=basic_preprocessor)
        if TRAIN:
            # Post trains the sound seperator model
            batch_size = 4
            train_loader = DataLoader(mixture_dataset, batch_size=batch_size)
            sound_seperator.train(train_loader)
    else:
        if x_path:
            x, _ = librosa.load(x_path, sr=AmphibDataset.sample_rate)
            x = torch.Tensor(x).unsqueeze(0).unsqueeze(1)
            x_path = Path(x_path) if isinstance(x_path, str) else x_path
            output_dir = x_path.stem
            clustered = predict_cluster(x, clusterer, sound_seperator)
            save_cluster_to_file(clustered, output_dir, x_path.stem)
        else:
            output_base_dir = FILES_DIR / "clustered"
            dataloader = DataLoader(dataset)
            for x, path in tqdm(dataloader):
                path = Path(path[0])
                output_dir = output_base_dir / path.stem
                clustered = predict_cluster(x, clusterer, sound_seperator)
                save_cluster_to_file(clustered, output_dir, x_path.stem)

if __name__ == "__main__":
    main("/media/marcel/3831-6261/20250503/243B1F02648802FC_20250503_035500.WAV")
