#!/usr/bin/env python3
import numpy as np
from typing import Sequence, Optional
import matplotlib.pyplot as plt

def plot_sound_waves(X: np.ndarray | Sequence[np.ndarray], 
                     in_one_ax: bool = False,
                     subplot_titles: Optional[Sequence[str]] = None,
                     sample_rate: int = 192000) -> None:
    if isinstance(X, np.ndarray) and X.ndim == 2:
        X = [X]

    n_rows = 1 if in_one_ax else len(X)

    if subplot_titles is not None and len(subplot_titles) != len(X):
        raise ValueError(f"Len subplot_titles {len(subplot_titles)} have to be same as Num axes {n_rows}")

    fig, ax = plt.subplots(nrows=n_rows, figsize=(20, 3 * n_rows))

    for i, x in enumerate(X):
        time = np.arange(len(x)) / sample_rate
        current_ax = ax[i] if n_rows > 1 else ax
        current_ax.plot(time, x)
        current_ax.set_xlabel("Time (s)")
        current_ax.set_ylabel("Amplitude")
        if subplot_titles:
            current_ax.set_title(subplot_titles[i])

    plt.tight_layout()
    plt.show()