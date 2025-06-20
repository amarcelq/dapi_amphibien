# wave-u-net
# bird-MixIT
# Demucs
# convtastnet https://docs.pytorch.org/audio/2.3.0/generated/torchaudio.models.ConvTasNet.html#torchaudio.models.ConvTasNet
from abc import ABC, abstractmethod
from data.dataset import FILES_DIR
import torch
from sklearn.decomposition import FastICA
from torchaudio.models import ConvTasNet
from tqdm import tqdm
from asteroid.losses import PITLossWrapper
from asteroid.losses import pairwise_neg_sisdr

WEIGHTS_DIR = FILES_DIR / "weights"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class ConvTas(SoundSperationMethod):
    def __init__(self, num_sources = 2, sample_rate: int = 8000, pretrained_weights_path=WEIGHTS_DIR / "conv_tasnet_base_libri2mix.pt"):#"frog_convtas.pt"):
        self.pre_trained_weights = pretrained_weights_path

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

    def train(self, train_loader, epochs=20):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='perm_avg')
        self.model.train()
        for i in range(epochs):
            for kaggle_x, youtube_x, mixture in tqdm(train_loader, f"Train ConvTas Epoch: {i}"):
                optimizer.zero_grad()
                est_sources = self.model(mixture)
                # calc two times the loss and take the lesser one otherwise the order of stack will bias the result
                sources = torch.stack([kaggle_x, youtube_x], dim=1).squeeze(2)                
                loss1 = loss_func(est_sources, sources)
                sources = torch.stack([youtube_x, kaggle_x], dim=1).squeeze(2)                
                loss2 = loss_func(est_sources, sources)
                loss = torch.min(loss1, loss2)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), (WEIGHTS_DIR / "frog_convtas.pt"))

    def pred(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

class FICA(SoundSperationMethod):
    def __init__(self, num_sources=3):
        self.ica = FastICA(num_sources=3, whiten="arbitrary-variance")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.ica.fit_transform(x)     

if __name__ == "__main__":
    pass
