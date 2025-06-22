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
from dataclasses import dataclass
from scipy.spatial.distance import cosine
from collections import deque

TRAIN: bool = False
WEB_USE: bool = True
SAMPLE_RATE: int = 8000
SILENCE_THRESHOLD: int = 35
FROG_MEAN_PATH: Path = FILES_DIR / "frog_mean.npy"
FROG_MEAN: Optional[np.ndarray] = np.load(FROG_MEAN_PATH) if FROG_MEAN_PATH.exists() else None
NON_FROG_MEAN_PATH: Path = FILES_DIR / "no_frog_mean.npy"
NON_FROG_MEAN: Optional[np.ndarray] = np.load(NON_FROG_MEAN_PATH) if NON_FROG_MEAN_PATH.exists() else None
TARGET_LEN: int = 5

app = FastAPI()

@app.get("/healthy")
async def healthy():
    return {"message": "Alive"}

class StartProcessRequest(BaseModel):
    path: str
    session_key: str

@dataclass
class AudioSegment:
    data: np.ndarray
    start_ms: float
    duration_ms: float

def create_post_content(session_key: str, name: str, description: str, status: str = "running") -> tuple[str, dict[str, Any]]:
    return "http://web:8000/internal/progress/update/", {"session_key": session_key, 
                                                              "progress":{"status": status,
                                                              "name": name,
                                                              "description": description}}

def post_content(session_key: str, name: str, description: str, status: str = "running")->None:
    url, body = create_post_content(session_key=session_key, name=name, description=description, status=status)
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

def generate_web_sample(id: int | str, snippets: Sequence):
    id = str(id)
    sample = {
                "id": id,
                "name": f"#{id}",
                "snippets": [{"url": str(file_path), "start": start_ms, "duration": duration_ms} for file_path, start_ms, duration_ms in snippets]
            }

    return sample

def create_web_return(samples: Sequence, in_path: str):
    res = {
            "main_audio": {"name": "Original", "url": str(in_path)}, # the main, original uploaded audio track from the user
            "alternative_audio": [{"name": "Denoised", "url": ""}], # alternative tracks that were created during processing (e.g. a denoised Version). They should all have the same length and SampleRate
            "samples": samples
        }
    
    return res

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

def separate_frog_sources(
    features: np.ndarray | torch.Tensor,
    sound_seperator,
    frog_mean,
    non_frog_mean,
    sample_rate: int,
    target_len: int,
    denoiser=None,
    session_key=None,
    max_depth: int = 3
) -> list[np.ndarray]:
    frogs: list[np.ndarray] = []
    queue: deque[tuple[np.ndarray | torch.Tensor, int]] = deque([(features, 0)])
    max_total_nodes = (2**(max_depth + 1) - 1)
    nodes_procceded = 0

    while queue:
        current, depth = queue.popleft()
        
        if depth >= max_depth:
            continue 
        
        if WEB_USE:
            post_content(
                session_key=session_key,
                name=f"Separate Sources with {sound_seperator.__class__.__name__}",
                description=f"Separate Sources {nodes_procceded}/{max_total_nodes} Nodes procceeded."
            )
        else:
            print(f"{nodes_procceded}/{max_total_nodes} Nodes procceeded.")

        if isinstance(current, np.ndarray):
            current = torch.Tensor(current)
        current = current.reshape(1, 1, -1)

        separated = sound_seperator.pred(current).squeeze(0)
        sources = [separated[0, :], separated[1, :]]

        added_children = 0

        for source in sources:
            if is_frog_cluster(source, frog_mean, non_frog_mean, sample_rate=sample_rate, target_len=target_len):
                source_np = source.numpy() if isinstance(source, torch.Tensor) else source
                if denoiser:
                    source_np = denoiser(source_np)
                queue.append((source_np, depth + 1))
                added_children += 1
            nodes_procceded += 1

        if added_children == 0 or depth + 1 >= max_depth:
            frogs.extend([
                BasicPreprocessor.normalize_to_lufs(source.numpy() if isinstance(source, torch.Tensor) else source, sample_rate=sample_rate)
                for source in sources
                if is_frog_cluster(source, frog_mean, non_frog_mean, sample_rate, target_len)
            ])

    return frogs

def split_frogs(frogs: list[Any], sample_rate: int, silence_threshold: float) -> list[list[AudioSegment]]:
    splitted_frogs = []
    for frog in frogs:
        segments = []
        indices = librosa.effects.split(frog, top_db=silence_threshold)
        for start, end in indices:
            segment = AudioSegment(
                data=frog[start:end],
                start_ms=(start / sample_rate) * 1000,
                duration_ms=((end - start) / sample_rate) * 1000
            )
            segments.append(segment)
        splitted_frogs.append(segments)
    return splitted_frogs

def predict_cluster(x: np.ndarray | torch.Tensor,
                    clusterer: ClusteringMethod,
                    sample_rate: int,
                    silence_threshhold: int,
                    frog_mean: np.ndarray,
                    non_frog_mean: np.ndarray,
                    sound_seperator: SoundSperationMethod,
                    denoiser: Optional[DenoiseMethod] = None,
                    feature_extractor: Optional[FeatureExtractMethod] = None,
                    feature_reductor: Optional[FeatureReductionMethod] = None,
                    session_key: Optional[str] = None,
                    ) -> list:
    
    if WEB_USE and session_key is None:
        raise ValueError("Session key must be provided when WEB_USE is enabled.")

    x_tensor = torch.Tensor(x) if isinstance(x, np.ndarray) else x

    frogs = separate_frog_sources(
        features=x_tensor,
        sound_seperator=sound_seperator,
        frog_mean=frog_mean,
        non_frog_mean=non_frog_mean,
        sample_rate=sample_rate,
        target_len=TARGET_LEN,
        denoiser=denoiser,
        session_key=session_key
    )

    # If there would be more compute available it would be possible to train the seperator to divide into "non frog" and "frog"
    # and after 

    # if denoiser:
    #     if WEB_USE:
    #         post_content(session_key=session_key, 
    #                      name=f"Denoise Seperated Sources with {denoiser.__class__.__name__}",
    #                        description="Denoise seperated Sources")
    #     x = denoiser(x)
    #     features = x

    # if feature_extractor:
    #     if WEB_USE:
    #           post_content(session_key=session_key, 
    #                             name=f"Extract Features with {feature_extractor.__class__.__name__}",
    #                             description="Extract relevant features from sperated Sources")
    #     features = feature_extractor(features)

    # if feature_reductor:
    #     if WEB_USE:
    #         post_content(session_key=session_key, 
    #                             name=f"Reduce Features with {feature_reductor.__class__.__name__}",
    #                             description="Reduce Features")   
    #     features = feature_reductor(features)

    # if WEB_USE:
    #     post_content(session_key=session_key, 
    #                         name=f"Create Clusters with {clusterer.__class__.__name__}",
    #                         description="Create Clusters")  
    # labels = clusterer(features.reshape(-1, 1))

    # unique_labels = set(labels)
    # clustered: dict = {label: [] for label in unique_labels}
    return split_frogs(frogs, sample_rate, silence_threshhold)

def save_cluster_to_file(
    splitted_frogs: list,
    output_dir: Path | str,
    session_key: Optional[str] = None,
    in_path: Optional[Path] = None
) -> None:
    output_dir = Path(output_dir)
    if WEB_USE:
        if not in_path:
            raise ValueError("in_path is not permitted if WEB_USE is set.")
        samples = list()

    for cluster_idx, frog_cluster in enumerate(splitted_frogs):
        cluster_dir = output_dir / str(cluster_idx)
        cluster_dir.mkdir(parents=True, exist_ok=True)
        if WEB_USE:
            snippets = list()

        for segment_idx, segment in enumerate(frog_cluster):
            path = cluster_dir / f"{segment_idx}.wav"
            sf.write(path, segment.data, SAMPLE_RATE, format="wav")
            if WEB_USE:
                snippets.append((path.absolute(), segment.start_ms, segment.duration_ms))

        if WEB_USE:
            samples.append(generate_web_sample(cluster_idx, snippets))

    if WEB_USE:
        requests.post(
            "http://web:8000/internal/progress/finish/",
            json={
                "session_key": session_key,
                "result": create_web_return(samples, str(in_path.absolute()))
            }
        )
        post_content(session_key, "Done", "Finished processing", "done")

def main(x_path: Optional[Path | str] = None, session_key: Optional[str] = None) -> None:
    if WEB_USE and session_key is None:
        raise ValueError("Session key have to be != None if WEB_USAGE.")

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
            if WEB_USE:
                post_content(session_key=session_key, 
                            name=f"Loading File with {SAMPLE_RATE}",
                            description="Loading File")

            x, _ = librosa.load(x_path, sr=SAMPLE_RATE)
            x_path = Path(x_path) if isinstance(x_path, str) else x_path
            if WEB_USE:
                output_dir = x_path.parent
            else:
                output_dir = FILES_DIR / "clustered" / x_path.stem
            clustered = predict_cluster(x, 
                                        clusterer = clusterer,
                                        sample_rate=SAMPLE_RATE,
                                        silence_threshhold=SILENCE_THRESHOLD,
                                        sound_seperator = sound_seperator, 
                                        feature_extractor = feature_extractor, 
                                        feature_reductor = feature_reductor,
                                        denoiser = denoiser,
                                        session_key = session_key,
                                        frog_mean = FROG_MEAN if FROG_MEAN is not None else frog_mean,
                                        non_frog_mean = NON_FROG_MEAN if NON_FROG_MEAN is not None else non_frog_mean)
            save_cluster_to_file(clustered, output_dir, session_key, x_path)
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
                                            sample_rate=SAMPLE_RATE,
                                            silence_threshhold=SILENCE_THRESHOLD,
                                            sound_seperator = sound_seperator, 
                                            feature_extractor = feature_extractor, 
                                            feature_reductor = feature_reductor,
                                            frog_mean = FROG_MEAN if FROG_MEAN is not None else frog_mean,
                                            non_frog_mean = NON_FROG_MEAN if NON_FROG_MEAN is not None else non_frog_mean)
                save_cluster_to_file(clustered, output_dir, session_key)
    return

if __name__ == "__main__":
    main(FILES_DIR / "frog_sounds" / "243B1F02648802FC_20250503_035500.WAV")
