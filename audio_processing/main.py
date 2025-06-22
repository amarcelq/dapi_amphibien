#!/usr/bin/env python3
from fastapi import FastAPI, BackgroundTasks
import tensorflow as tf
from pydantic import BaseModel
import asyncio
from data.dataset import AmphibDataset, FILES_DIR, KaggleAnuranSoundDataset, MixedAudioDataset, YouTubeNoiseDataset
from data.utils import download_mixture_audio
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
from scipy.spatial.distance import euclidean
from collections import defaultdict

TRAIN: bool = False
SAMPLE_RATE: int = 8000
SILENCE_THRESHOLD = 35
FROG_MEAN_PATH = FILES_DIR / "frog_mean.npy"
FROG_MEAN = np.load(FROG_MEAN_PATH) if FROG_MEAN_PATH.exists() else None
NON_FROG_MEAN_PATH = FILES_DIR / "no_frog_mean.npy"
NON_FROG_MEAN = np.load(NON_FROG_MEAN_PATH) if NON_FROG_MEAN_PATH.exists() else None
TARGET_LEN = 5

app = FastAPI()

@app.get("/healthy")
async def healthy():
    return {"message": "Alive"}

class StartProcessRequest(BaseModel):
    path: str
    session_key: str

def create_post_content(session_key: str, name: str, description: str, status: str = "running") -> tuple[str, dict[str, Any]]:
    return "http://web:8000/internal/progress/update/", {"session_key": session_key, 
                                                              "progress":{"status": status,
                                                              "name": name,
                                                              "description": description}}

def post_content(session_key: str, name: str, description: str, status: str = "running")->None:
    url, body = create_post_content(session_key=session_key, name=name, description=description, status=status)
    print(body)
    requests.post(url=url, json=body)  

@app.post("/start_process")
async def start_process(request: StartProcessRequest):
    requests.post("http://web:8000/internal/progress/update/",
                  json={"session_key": request.session_key,
                        "progress":{"status":"running","name":"Loading File","description":"Loading the uploaded file"}})
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, process, request)
    # asyncio.create_task(process(request))
    
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

    youtube_sounds_path = FILES_DIR / "youtube_sounds"
    if not youtube_sounds_path.exists():
        download_mixture_audio()

    youtube_dataset = YouTubeNoiseDataset(
        youtube_folder=youtube_sounds_path,
        target_len=TARGET_LEN,
        sample_rate=AmphibDataset.sample_rate,
        generate_n_samples=len(kaggle_dataset)
    )

    mixture_dataset = MixedAudioDataset(
        kaggle_dataset=kaggle_dataset,
        youtube_dataset=youtube_dataset,
        target_len=TARGET_LEN,
        sample_rate=AmphibDataset.sample_rate
    )

    return dataset, kaggle_dataset, youtube_dataset, mixture_dataset

def mean_dataset(dataset, sample_rate: int, target_len: int = 5, use_n_samples: Optional[int] = None) -> np.ndarray:
    target_len = int(sample_rate * target_len)
    total_sum = torch.zeros(target_len)

    use_n_samples = use_n_samples if use_n_samples is not None else len(dataset)

    for i in range(use_n_samples):
        x = dataset[i]
        x = x[:target_len]
        pad_width = target_len - x.shape[0]
        if pad_width > 0:
            pad_value = 10 ** (-SILENCE_THRESHOLD / 20)            
            x = torch.nn.functional.pad(x, (0, pad_width), value=pad_value)
        total_sum += x

    mean = total_sum / use_n_samples
    return mean.numpy()

def is_frog_cluster(x: torch.Tensor | np.ndarray,
                    frog_mean: np.ndarray,
                    non_frog_mean: np.ndarray,
                    sample_rate: int,
                    target_len: int) -> bool:
    
    target_len = int(target_len * sample_rate)
    x_np = x.numpy() if isinstance(x, torch.Tensor) else x
    frog_votes = 0
    non_frog_votes = 0

    for i in range(0, len(x_np) - target_len + 1, target_len):
        chunk = x_np[i:i + target_len]
        dist_frog = euclidean(chunk, frog_mean)
        dist_non_frog = euclidean(chunk, non_frog_mean)
        if dist_frog < dist_non_frog:
            frog_votes += 1
        else:
            non_frog_votes += 1

    return frog_votes > non_frog_votes

def predict_cluster(x: np.ndarray | torch.Tensor,
                    clusterer: ClusteringMethod,
                    frog_mean: np.ndarray,
                    non_frog_mean: np.ndarray,
                    sound_seperator: SoundSperationMethod,
                    denoiser: Optional[DenoiseMethod] = None,
                    feature_extractor: Optional[FeatureExtractMethod] = None,
                    feature_reductor: Optional[FeatureReductionMethod] = None,
                    session_key: Optional[str] = None,
                    ) -> dict:
    
    
    features: torch.Tensor | np.ndarray = torch.Tensor(x) if isinstance(x, np.ndarray) else x
    if session_key:
        post_content(session_key=session_key, 
                            name=f"Seperate Sources with {sound_seperator.__class__.__name__}",
                            description="Seperate Sources")
        
    frogs: list[np.ndarray] = list()
    stack = [features]
    i = 1

    while stack:
        if session_key:
            post_content(session_key=session_key, 
                                name=f"Sound Seperation with {sound_seperator.__class__.__name__}",
                                description=f"Sound Seperation Iteration: {i}")
        current = stack.pop()
        current: torch.Tensor | np.ndarray = torch.Tensor(current) if isinstance(x, np.ndarray) else current
        current = current.reshape(1, 1, -1)
        _separated = sound_seperator.pred(current)

        source_1 = _separated[0, 0, :]
        source_2 = _separated[0, 1, :]
        for source in (source_1, source_2):
            if is_frog_cluster(source, 
                               frog_mean, 
                               non_frog_mean, 
                               sample_rate=SAMPLE_RATE, 
                               target_len=TARGET_LEN) and (len(frogs) == 0 or not np.allclose(frogs[-1], source, atol=1e-2)):
                    if denoiser:
                        source = denoiser(source)
                    frogs.append(source)
                    stack.append(source)
        if i == 8:
            break
        i += 1
    # TODO: Maybe fix it!
    # if denoiser:
    #     if session_key:
    #         post_content(session_key=session_key, 
    #                             name=f"Denoise Seperated Sources with {denoiser.__class__.__name__}",
    #                             description="Denoise seperated Sources")
    #     x = denoiser(x)
    #     features = x

    # if feature_extractor:
    #     if session_key:
    #         post_content(session_key=session_key, 
    #                             name=f"Extract Features with {feature_extractor.__class__.__name__}",
    #                             description="Extract relevant features from sperated Sources")
    #     features = feature_extractor(features)

    # if feature_reductor:
    #     if session_key:
    #         post_content(session_key=session_key, 
    #                             name=f"Reduce Features with {feature_reductor.__class__.__name__}",
    #                             description="Reduce Features")   
    #     features = feature_reductor(features)

    # if session_key:
    #     post_content(session_key=session_key, 
    #                         name=f"Create Clusters with {clusterer.__class__.__name__}",
    #                         description="Create Clusters")  
    # labels = clusterer(features.reshape(-1, 1))

    # unique_labels = set(labels)
    # clustered: dict = {label: [] for label in unique_labels}
    splitted_frogs = list()
    for frog in frogs:
        indizes = librosa.effects.split(frog, top_db=SILENCE_THRESHOLD)
        splitted = list()
        for i in indizes:
            s, e = i
            splitted.append(frog[s:e])
        splitted_frogs.append(splitted)

    return splitted_frogs

def save_cluster_to_file(splitted_frogs: list, output_dir: Path | str, file_name: Optional[str | Path] = None):
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    for i, splitted_frog in enumerate(splitted_frogs):
        splitted_frog_dir = output_dir / str(i)
        splitted_frog_dir.mkdir(parents=True, exist_ok=True)
        for i, occurence in enumerate(splitted_frog):
            sf.write(splitted_frog_dir / f"{i}.wav", occurence, SAMPLE_RATE, format="wav")

    # for cluster, data in clustered.items():
    #     for occurence, splitted_data in enumerate(data):
    #         output_file = os.path.join(output_dir, f"{file_name}_{occurence}_{cluster}.wav")
    #         sf.write(output_file, splitted_data, SAMPLE_RATE)

def main(x_path: Optional[Path | str] = None, session_key: Optional[str] = None) -> None:
    AmphibDataset.sample_rate = SAMPLE_RATE
    basic_noise_path = FILES_DIR / "basic_noise.wav"
    basic_noise, _ = librosa.load(basic_noise_path, sr=SAMPLE_RATE)
    denoiser = SpectralGate(sample_rate=SAMPLE_RATE, stationary=True, noise_signal=basic_noise)
    basic_preprocessor = BasicPreprocessor(sample_rate=SAMPLE_RATE, parts_len=8, add_freq_dim=None, resample_rate=SAMPLE_RATE)
    sound_seperator = ConvTas(num_sources=2, sample_rate=SAMPLE_RATE)
    feature_extractor = None#OpenL3Embedding(sample_rate=SAMPLE_RATE)
    feature_reductor = PCA(n_dims=2)
    clusterer = BGMM(n_clusters=10)

    if FROG_MEAN is None or NON_FROG_MEAN is None or TRAIN:
        dataset, kaggle_dataset, youtube_dataset, mixture_dataset = create_datasets(data_path=FILES_DIR / "frog_sounds", denoiser=denoiser, basic_preprocessor=basic_preprocessor)

    if TRAIN:
        # Post trains the sound seperator model
        batch_size = 4
        mixture_dataset, _ = random_split(mixture_dataset, [300, len(mixture_dataset) - 300])
        train_loader = DataLoader(mixture_dataset, batch_size=batch_size)
        sound_seperator.train(train_loader)

    else:
        if FROG_MEAN is None:
            #frog_mean = mean_dataset(kaggle_dataset, SAMPLE_RATE, target_len=TARGET_LEN)
            frog_mean, _ = librosa.load(FILES_DIR / "frog_mean.wav", sr=SAMPLE_RATE, duration=TARGET_LEN)
            frog_mean = denoiser(frog_mean)
            np.save(FROG_MEAN_PATH, frog_mean)
        if NON_FROG_MEAN is None:
            non_frog_mean = mean_dataset(youtube_dataset, SAMPLE_RATE, target_len=TARGET_LEN)
            np.save(NON_FROG_MEAN_PATH, non_frog_mean)


        if x_path:    
            if session_key:
                post_content(session_key=session_key, 
                                    name=f"Loading File with {SAMPLE_RATE}",
                                    description="Loading File")

            x, _ = librosa.load(x_path, sr=SAMPLE_RATE)
            x_path = Path(x_path) if isinstance(x_path, str) else x_path
            output_dir = FILES_DIR / "clustered" / x_path.stem
            clustered = predict_cluster(x, 
                                        clusterer = clusterer, 
                                        sound_seperator = sound_seperator, 
                                        feature_extractor = feature_extractor, 
                                        feature_reductor = feature_reductor,
                                        denoiser = denoiser,
                                        session_key = session_key,
                                        frog_mean = FROG_MEAN if FROG_MEAN is not None else frog_mean,
                                        non_frog_mean = NON_FROG_MEAN if NON_FROG_MEAN is not None else non_frog_mean)
            save_cluster_to_file(clustered, output_dir, x_path.stem)
        else:
            output_base_dir = FILES_DIR / "clustered"
            if (FROG_MEAN is None and NON_FROG_MEAN is None) or TRAIN:
                dataset = AmphibDataset(parent_path=FILES_DIR / "frog_sounds", denoiser=denoiser, basic_preprocessor=basic_preprocessor)
            dataloader = DataLoader(dataset)
            for x, path in tqdm(dataloader):
                path = Path(path[0])
                output_dir = output_base_dir / path.stem
                clustered = predict_cluster(x, 
                                            clusterer = clusterer,
                                            sound_seperator = sound_seperator, 
                                            feature_extractor = feature_extractor, 
                                            feature_reductor = feature_reductor,
                                            frog_mean = FROG_MEAN if FROG_MEAN is not None else frog_mean,
                                            non_frog_mean = NON_FROG_MEAN if NON_FROG_MEAN is not None else non_frog_mean)
                save_cluster_to_file(clustered, output_dir, path.stem)
    return

if __name__ == "__main__":
    main(FILES_DIR / "frog_sounds" / "243B1F02648802FC_20250503_035500.WAV")
