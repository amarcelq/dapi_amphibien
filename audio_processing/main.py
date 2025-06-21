from fastapi import FastAPI
from pydantic import BaseModel
import requests
import asyncio
import audio_processing.src.main as audio

app = FastAPI()

@app.get("/healthy")
async def healthy():
    return {"message": "Alive"}


class StartProcessRequest(BaseModel):
    path: str

@app.post("/start_process")
async def start_process(request: StartProcessRequest):
    # example sending progress
    session_key = request.session_key
    in_file_path = request.path
    request.post("web:8000/internal/progress/update/",json={"session_key":session_key,"progress":{"status":"running","name":"Loading File","description":"Loading the uploaded file"}})
    asyncio.create_task(process(session_key,in_file_path))
    return {"message": f"Process started for path: {in_file_path}"}



def process(session_key,in_file_path):
    audio
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
import kagglehub
import torch
from torch.utils.data import random_split

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
                    feature_extractor: Optional[FeatureExtractMethod] = None,
                    feature_reductor: Optional[FeatureReductionMethod] = None,
                    ) -> dict:
    
    features: np.ndarray | torch.Tensor = x
    if sound_sperator:
        _seperated = sound_sperator.pred(features)
        # TODO: Implement this function!
        x = _seperated[0, 1, :]#.reshape(-1, 1)        
        sf.write(FILES_DIR / "clustered" / "hallo.wav", x, SAMPLE_RATE)
        features = x

    if feature_extractor:
        features = feature_extractor(features)

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
    feature_extractor = OpenL3Embedding(sample_rate=SAMPLE_RATE)
    feature_reductor = PCA(n_dims=2)
    clusterer = BGMM(n_clusters=5)

    if TRAIN:
        dataset, kaggle_dataset, mixture_dataset = create_datasets(data_path="/media/marcel/3831-6261", denoiser=spectral_gate, basic_preprocessor=basic_preprocessor)
        # Post trains the sound seperator model
        batch_size = 4
        mixture_dataset, _ = random_split(mixture_dataset, [300, len(mixture_dataset) - 300])
        train_loader = DataLoader(mixture_dataset, batch_size=batch_size)
        sound_seperator.train(train_loader)

    else:
        if x_path:
            x, _ = librosa.load(x_path, sr=AmphibDataset.sample_rate)
            x = torch.Tensor(x).unsqueeze(0).unsqueeze(1)
            x_path = Path(x_path) if isinstance(x_path, str) else x_path
            output_dir = FILES_DIR / "clustered" #x_path.stem
            clustered = predict_cluster(x, clusterer, feature_extractor, feature_reductor, sound_seperator)
            save_cluster_to_file(clustered, output_dir, x_path.stem)
        else:
            if not TRAIN:
                dataset, kaggle_dataset, mixture_dataset = create_datasets(data_path="/media/marcel/3831-6261", denoiser=spectral_gate, basic_preprocessor=basic_preprocessor)
            output_base_dir = FILES_DIR / "clustered"
            dataloader = DataLoader(dataset)
            for x, path in tqdm(dataloader):
                path = Path(path[0])
                output_dir = output_base_dir / path.stem
                clustered = predict_cluster(x, clusterer, sound_seperator)
                save_cluster_to_file(clustered, output_dir, x_path.stem)

if __name__ == "__main__":
    main("/media/marcel/3831-6261/20250503/243B1F02648802FC_20250503_035500.WAV")
