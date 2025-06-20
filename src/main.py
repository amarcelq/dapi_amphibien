#!/usr/bin/env python3
from data.dataset import AmphibDataset, FILES_DIR
from preprocessing.denoise import SpectralGate
from preprocessing.basic_preprocessing import BasicPreprocessor
from torch.utils.data import DataLoader
from sound_seperation.clustering import BGMM
from sound_seperation.feature_extraction import MFCC, SpectralFeature, Chroma
from sound_seperation.feature_reduction import PCA
from sound_seperation.sound_speration import MixIt, ConvTas

from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt
from collections import defaultdict
import soundfile as sf
import os
from pathlib import Path
import numpy as np
import openl3
import torch
def main():

    data_path = "/media/marcel/3831-6261/20250430"
    basic_noise_path = FILES_DIR / "basic_mic_noise_with_crickets.wav"
    basic_noise, _ = librosa.load(basic_noise_path, sr=AmphibDataset.sample_rate)
    spectral_gate = SpectralGate(AmphibDataset.sample_rate, noise_signal=basic_noise, stationary=True)
    basic_preprocessor = BasicPreprocessor(AmphibDataset.sample_rate, add_freq_dim=None, resample_rate=8000)
    # TODO: shift it to BasicPreprocessor
    AmphibDataset.sample_rate = 8000
    sound_seperator = ConvTas(num_sources=2, sample_rate=AmphibDataset.sample_rate)

    feature_extraction = Chroma(AmphibDataset.sample_rate)
    feature_reduction = PCA(n_dims=2)
    bgmm = BGMM(n_clusters=5)

    dataset = AmphibDataset(parent_path_str=data_path, basic_preprocessor=basic_preprocessor, denoiser=spectral_gate)
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    cluster_data = defaultdict(list)

    for X, path in tqdm(dataloader):
        path = Path(path[0])
        #mixture_waveform = torch.sum(x, axis=0)

        X = torch.Tensor(X).unsqueeze(1)
        X = sound_seperator.pred(X)
        print(X.shape)
        x = [X[i].numpy() for i in range(X.shape[0])]
        X, _ = openl3.get_audio_embedding(x, AmphibDataset.sample_rate, content_type="env")
        X = np.array(X).shape

        for i, data in enumerate(dataset):
            x = np.array(x).mean(axis=1)
            #x = feature_reduction(x)
            y = bgmm(x)

            for cluster, x in zip(y, X.squeeze(0)):
                cluster_data[cluster].extend(x)

            for cluster, data in cluster_data.items():
                output_dir = FILES_DIR / "clustered" / path.stem
                os.makedirs(output_dir, exist_ok=True)
                sf.write(output_dir / f"{cluster}.wav", data, AmphibDataset.sample_rate)

if __name__ == "__main__":
    main()
