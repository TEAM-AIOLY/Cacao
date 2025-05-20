import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter,detrend
import distinctipy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.lines import Line2D
import random
from sklearn.manifold import TSNE
import umap

random.seed(42)
data_path='./data/datasets/3rd_data_sampling_and_microbial_data/data 3rd sampling 450-950nm.csv'
data = pd.read_csv(data_path, sep=',', header=0)

wv =(np.array(data.keys()[1:]).astype(float))
spectral_data = (np.array(data.iloc[:, 1:]))

Treatments = np.array([f"T{i}" for i in range(1, 16)])  # Generate T1 to T15

microbe_path = './data/datasets/3rd_data_sampling_and_microbial_data/MIX MICROBE dis.csv'
data_microbe = pd.read_csv(microbe_path, sep=',', header=0, lineterminator='\n', skip_blank_lines=True)
Y = np.array(data_microbe.iloc[0:17, 1:])
y_micro = Y[:, 1:]
y_ferti = Y[:, 0].astype(float)
label_names = ['B1', 'B2', 'B3', 'B4']


# X = np.log1p(spectral_data) 

# Settings for the smooth derivatives using a Savitsky-Golay filter
w = 37 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree
d=1 ## Sav.Gol derive  order

X = savgol_filter(spectral_data, window_length=w, polyorder=p,deriv=d, axis=1)
X= detrend(X, axis=1)


tsne = TSNE(n_components=3, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

default_colors = distinctipy.get_colors(15)
ferti_norm = (y_ferti - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
sizes = 30 + 300 * ferti_norm ** 1.5  # emphasize effect


unique_ferti = np.unique(y_ferti)
ferti_legend_elements = []
for val in unique_ferti:
    norm_val = (val - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
    size = 30 + 300 * norm_val ** 1.5
    ferti_legend_elements.append(Line2D([0], [0], marker='o', color='gray', label=f'Ferti: {val:.1f}',
        markerfacecolor='gray', markersize=np.sqrt(size), markeredgecolor='black', linewidth=0))

pc_pairs = [(0, 1), (1, 2), (0, 2)]

for i, j in pc_pairs:
    plt.figure(figsize=(8, 6))
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        for x, y, label, ferti_size in zip(X_tsne[idx, i], X_tsne[idx, j], Treatments[idx], sizes[idx]):
            plt.scatter(x, y, color=default_colors[k], s=ferti_size, edgecolor='black', alpha=0.85)
            plt.text(x + 0.1, y, label, fontsize=9)
    plt.title("t-SNE on Cacao Data with Fertilization Size")
    plt.xlabel(f"t-SNE {i + 1}")
    plt.ylabel("t-SNE {j + 1}")
    plt.grid()
    plt.legend(handles=ferti_legend_elements, title="Fertilization Size",
            bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
    plt.tight_layout()
    plt.show()
