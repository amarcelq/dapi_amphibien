#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from data.dataset import AmphibDataset, FILES_DIR, KaggleAnuranSoundDataset, MixedAudioDataset
from preprocessing.denoise import SpectralGate, DenoiseMethod
from preprocessing.basic_preprocessing import BasicPreprocessor
from torch.utils.data import DataLoader
from sound_seperation.clustering import BGMM, ClusteringMethod
from sound_seperation.feature_extraction import MFCC, SpectralFeature, Chroma, OpenL3Embedding, FeatureExtractMethod
from sound_seperation.feature_reduction import PCA, FeatureReductionMethod
from sound_seperation.sound_speration import ConvTas, SoundSperationMethod

from typing import Optional, Sequence, Any
from tqdm import tqdm
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
import kagglehub
import torch
from torch.utils.data import random_split
import os
import requests

TRAIN: bool = False
SAMPLE_RATE: int = 8000
SILENCE_THRESHOLD = 60

app = FastAPI()

@app.get("/healthy")
async def healthy():
    return {"message": "Alive"}

class StartProcessRequest(BaseModel):
    path: str
    session_key: str

def create_post_content(session_key: str, name: str, description: str, status: str = "running") -> tuple[str, dict[str, Any]]:
    return "web:8000/internal/progress/update/", {"session_key": session_key, 
                                                  "progress":{"status": status,
                                                              "name": {name},
                                                              "description": {description}}}

@app.post("/start_process")
async def start_process(request: StartProcessRequest):
    requests.post("web:8000/internal/progress/update/",
                  json={"session_key": request.session_key,
                        "progress":{"status":"running","name":"Loading File","description":"Loading the uploaded file"}})
    asyncio.create_task(process(request))
    
    return {"message": f"Process started for path: {request.path}"}

def process(request: StartProcessRequest):
    main(request.path, request.session_key)

def create_datasets(data_path: Sequence[str] | Sequence[Path], denoiser: Optional[DenoiseMethod] = None, basic_preprocessor: Optional[BasicPreprocessor] = None):
    if isinstance(data_path, (str, Path)):
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

def predict_cluster(x: np.ndarray | torch.Tensor,
                    clusterer: ClusteringMethod,
                    sound_seperator: Optional[SoundSperationMethod] = None,
                    denoiser: Optional[DenoiseMethod] = None,
                    feature_extractor: Optional[FeatureExtractMethod] = None,
                    feature_reductor: Optional[FeatureReductionMethod] = None,
                    session_key: Optional[str] = None
                    ) -> dict:
    
    
    features: torch.Tensor | np.ndarray = torch.Tensor(x) if isinstance(x, np.ndarray) else x
    if sound_seperator:
        if session_key:
            requests.post(create_post_content(session_key=session_key, 
                                name=f"Seperate Sources with {sound_seperator.__class__.__name__}",
                                description="Seperate Sources"))
        features = features.reshape(1, 1, -1)
        _seperated = sound_seperator.pred(features)
        print(_seperated)
        # TODO: Implement this function!
        #get_frog_cluster()
        x = _seperated[0, 1, :].numpy()#.reshape(-1, 1)
        print(x.shape)
        features = x

    if denoiser:
        if session_key:
            requests.post(create_post_content(session_key=session_key, 
                                name=f"Denoise Seperated Sources with {denoiser.__class__.__name__}",
                                description="Denoise seperated Sources"))    
        print(x)
        x = denoiser(x)
        print(x)
        features = x

    if feature_extractor:
        if session_key:
            requests.post(create_post_content(session_key=session_key, 
                                name=f"Extract Features with {feature_extractor.__class__.__name__}",
                                description="Extract relevant features from sperated Sources"))   
        features = feature_extractor(features)

    if feature_reductor:
        if session_key:
            requests.post(create_post_content(session_key=session_key, 
                                name=f"Reduce Features with {feature_reductor.__class__.__name__}",
                                description="Reduce Features"))   
        features = feature_reductor(features)

    if session_key:
        requests.post(create_post_content(session_key=session_key, 
                            name=f"Create Clusters with {clusterer.__class__.__name__}",
                            description="Create Clusters"))  
    labels = clusterer(features.reshape(-1, 1))

    unique_labels = set(labels)
    clustered: dict = {label: [] for label in unique_labels}
    for idx, label in enumerate(labels):
        splitted = librosa.effects.split(x[idx], top_db=SILENCE_THRESHOLD)
        clustered[label].append(splitted)

    return clustered

def save_cluster_to_file(clustered: dict, output_dir: Path | str, file_name: str | Path):
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    for cluster, data in clustered.items():
        for occurence, splitted_data in enumerate(data):
            output_file = os.path.join(output_dir, f"{file_name}_{occurence}_{cluster}.wav")
            sf.write(output_file, splitted_data, SAMPLE_RATE)

def main(x_path: Optional[Path | str] = None, session_key: Optional[str] = None) -> None:
    #basic_noise_path = FILES_DIR / "basic_mic_noise_with_crickets.wav"
    #basic_noise, _ = librosa.load(basic_noise_path, sr=AmphibDataset.sample_rate)
    #noise_signal=basic_noise
    denoiser = SpectralGate(AmphibDataset.sample_rate, stationary=True)
    basic_preprocessor = BasicPreprocessor(AmphibDataset.sample_rate, parts_len=8, add_freq_dim=None, resample_rate=SAMPLE_RATE)
    AmphibDataset.sample_rate = SAMPLE_RATE
    sound_seperator = ConvTas(num_sources=2, sample_rate=SAMPLE_RATE)
    feature_extractor = OpenL3Embedding(sample_rate=SAMPLE_RATE)
    feature_reductor = PCA(n_dims=2)
    clusterer = BGMM(n_clusters=10)

    if TRAIN:
        dataset, kaggle_dataset, mixture_dataset = create_datasets(data_path="/media/marcel/3831-6261", denoiser=denoiser, basic_preprocessor=basic_preprocessor)
        # Post trains the sound seperator model
        batch_size = 4
        mixture_dataset, _ = random_split(mixture_dataset, [300, len(mixture_dataset) - 300])
        train_loader = DataLoader(mixture_dataset, batch_size=batch_size)
        sound_seperator.train(train_loader)

    else:
        if x_path:
            if session_key:
                requests.post(create_post_content(session_key=session_key, 
                                    name=f"Loading File with {SAMPLE_RATE}",
                                    description="Loading File")) 

            x, _ = librosa.load(x_path, sr=SAMPLE_RATE)
            x_path = Path(x_path) if isinstance(x_path, str) else x_path
            output_dir = FILES_DIR / "clustered" / x_path.stem
            clustered = predict_cluster(x, 
                                        clusterer = clusterer, 
                                        sound_seperator = sound_seperator, 
                                        feature_extractor = feature_extractor, 
                                        feature_reductor = feature_reductor,
                                        denoiser = denoiser,
                                        session_key = session_key)
            save_cluster_to_file(clustered, output_dir, x_path.stem)
        else:
            if not TRAIN:
                dataset, kaggle_dataset, mixture_dataset = create_datasets(data_path="/media/marcel/3831-6261", denoiser=spectral_gate, basic_preprocessor=basic_preprocessor)
            output_base_dir = FILES_DIR / "clustered"
            dataloader = DataLoader(dataset)
            for x, path in tqdm(dataloader):
                path = Path(path[0])
                output_dir = output_base_dir / path.stem
                clustered = predict_cluster(x, 
                                            clusterer = clusterer, 
                                            sound_seperator = sound_seperator, 
                                            feature_extractor = feature_extractor, 
                                            feature_reductor = feature_reductor)
                save_cluster_to_file(clustered, output_dir, path.stem)
    return

if __name__ == "__main__":
    main("/media/marcel/3831-6261/20250503/243B1F02648802FC_20250503_035500.WAV")
