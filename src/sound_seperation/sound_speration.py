# wave-u-net
# bird-MixIT
# Demucs
# convtastnet https://docs.pytorch.org/audio/2.3.0/generated/torchaudio.models.ConvTasNet.html#torchaudio.models.ConvTasNet
from abc import ABC, abstractmethod
from demucs.pretrained import get_model
from demucs.apply import apply_model
from data.dataset import FILES_DIR
import torch
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from sklearn.decomposition import FastICA
from sound_seperation.models.MixIT import model
from sound_seperation.models.MixIT.train import data_io
import os
import numpy as np
from torchaudio.models import ConvTasNet
import torchaudio

WEIGHTS_DIR = FILES_DIR / "weights"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tf.disable_v2_behavior()

class SoundSperationMethod(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def train(self, x):
        pass

    @abstractmethod
    def pred(self, x):
        pass

class ICA(SoundSperationMethod):
    def __init__(self, **params):
        self.ica = FastICA(**params)

    def __call__(self, x):
        x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        sources = self.ica.fit_transform(x.T).T
        sources = torch.tensor(sources, dtype=torch.float32)
        return sources

    def pred(self, x):
        return self.__call__(x)

class MixIt(SoundSperationMethod):
    def __init__(self, model_dir, sample_rate: int):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.model_dir = model_dir

            self.hparams = model.get_model_hparams()
            self.hparams.sr = sample_rate

            self.roomsim_params = {
                # TODO: Change it to make is correct (how many sequences should be combined)
                'num_sources': len(self.hparams.signal_names),
                'num_receivers': 1,
                'num_samples': int(self.hparams.sr),
            }

            self.feature_spec = data_io.get_roomsim_spec(**self.roomsim_params)
            self.inference_spec = data_io.get_inference_spec()

            self.params = {
                'feature_spec': self.feature_spec,
                'inference_spec': self.inference_spec,
                'hparams': self.hparams,
                'io_params': {
                    'parallel_readers': 1,
                    'num_samples': int(self.hparams.sr)
                },
                'model_dir': self.model_dir,
                'train_batch_size': 6,
                'eval_batch_size': 6,
                'train_steps': 20000000,
                'eval_suffix': 'validation',
                'eval_examples': 800,
                'save_checkpoints_secs': 600,
                'save_summary_steps': 1000,
                'keep_checkpoint_every_n_hours': 4,
                'write_inference_graph': True,
                'randomize_training': True,
            }
        
            self.estimator = None

    def __call__(self, x):
        pass

    def _input_fn(self, params):
        with self.graph.as_default():
            X_dataset = params['input_data']
            randomize_order = params.get('randomize_order', False)

            max_combined_sources = self.roomsim_params['num_sources']
            num_mics = self.roomsim_params['num_receivers']
            num_samples = self.roomsim_params['num_samples']

            n_prefetch = self.params["io_params"]["parallel_readers"] 

            def add_freq_dim(chunk):
                return tf.cond(
                    tf.equal(tf.rank(chunk), 3),
                    lambda: chunk,
                    lambda: tf.expand_dims(chunk, axis=1) # (max_combined_sources, 1, num_samples)
                )

            X_dataset = X_dataset.map(add_freq_dim)

            # Build mixture and sources waveforms.
            def combine_mixture_and_sources(waveforms):

                #waveforms = tf.reshape(waveforms, (max_combined_sources, num_mics, num_samples))
                # waveforms is shape (max_combined_sources, num_mics, num_samples).
                mixture_waveform = tf.reduce_sum(waveforms, axis=0)
                source_waveforms = tf.reshape(waveforms,
                                            (max_combined_sources, num_mics, num_samples))
                return {'receiver_audio': mixture_waveform,
                        'source_images': source_waveforms}
            
            X_dataset = X_dataset.batch(max_combined_sources).map(combine_mixture_and_sources)
            if randomize_order:
                X_dataset = X_dataset.shuffle(buffer_size=50)

            X_dataset = X_dataset.prefetch(n_prefetch)

            iterator = X_dataset.make_one_shot_iterator()
            return iterator.get_next()

    def _train_input_fn(self, X_dataset):
        with self.graph.as_default():
            train_params = self.params.copy()
            train_params['input_data'] = X_dataset
            train_params['batch_size'] = self.params['train_batch_size']
            if self.params['randomize_training']:
                train_params['randomize_order'] = True
            return self._input_fn(train_params)

    def _estimator_model_fn(self, features, labels, mode, params):
        with self.graph.as_default():
            spec = model.model_fn(features, labels, mode, params)
            return spec

    def train(self, X):        
        with self.graph.as_default():
            # combine multiple audio "files" to one mixture and use the not combined "files" as y data to give the model the chance to learn to seperate
            self.params["num_samples"] = X.shape[1]
            self.params["io_params"]["num_samples"] = X.shape[1]

            X = tf.data.Dataset.from_tensor_slices(X)

            run_config = tf_estimator.RunConfig(
                model_dir=self.params['model_dir'],
                save_summary_steps=self.params['save_summary_steps'],
                save_checkpoints_secs=self.params['save_checkpoints_secs'],
                keep_checkpoint_every_n_hours=self.params['keep_checkpoint_every_n_hours'])

            self.estimator = tf_estimator.Estimator(
                model_fn=self._estimator_model_fn,
                params=self.params,
                config=run_config)

            self.estimator.train(input_fn=lambda: self._train_input_fn(X),
                                steps=self.params['train_steps'])

    def pred(self, x):
        raise NotImplementedError

class ConvTas(SoundSperationMethod):
    def __init__(self, num_sources = 2, sample_rate: int = 8000):
        self.pre_trained_weights = WEIGHTS_DIR / "conv_tasnet_base_libri2mix.pt"

        self.model = ConvTasNet(
            num_sources=num_sources,
            enc_kernel_size=16,
            enc_num_feats=512,
            msk_kernel_size=3,
            msk_num_feats=128,
            msk_num_hidden_feats=512,
            msk_num_layers=8,
            msk_num_stacks=3,
            msk_activate="relu",
        )

        state_dict = torch.load(self.pre_trained_weights)
        self.model.load_state_dict(state_dict)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train(self, x):
        raise NotImplementedError

    def pred(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

class FICA(SoundSperationMethod):
    def __init__(self, num_sources=3):
        self.ica = FastICA(num_sources=3, whiten="arbitrary-variance")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.ica.fit_transform(x)     

class Demucs(SoundSperationMethod):
    def __init__(self, model_name='htdemucs', device=None, **params):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(name=model_name).to(self.device)
        self.model.eval()
        self.params = params

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            sources = apply_model(self.model, x.unsqueeze(0), split=True, overlap=0.25)[0]
        return {name: source.cpu() for name, source in zip(self.model.sources, sources)}

    def train(self):
        pass

if __name__ == "__main__":
    pass
