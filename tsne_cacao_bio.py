import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import distinctipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import random
from sklearn.manifold import TSNE
import umap

random.seed(42)

# === Load and preprocess data ===
data_path = './data/datasets/3rd_data_sampling_and_microbial_data/y_bio.csv'

    
    
data = pd.read_csv(data_path, sep=';', header=0, lineterminator='\n', skip_blank_lines=True)
Treatments = np.array([f"T{i}" for i in range(1, 16)])  # Generate T1 to T15

headers = ['Girth_final','Girth_increment','height_final','height_increment','Nb_leaves','roor_length',
           'Fresh_weight_leaves','Fresh_weight_stem','Fresh_weight_root','Total_fresh_weight',
           'Dry_weight_leaves','Dry_weight_stem','Dry_weight_root','Total_dry_weight']

X = data.iloc[1:17, 1:]
X = X.map(lambda x: str(x).replace(',', '.'))
X = np.array(X).astype(float)
X = StandardScaler().fit_transform(X)

microbe_path = './data/datasets/3rd_data_sampling_and_microbial_data/MIX MICROBE dis.csv'
data_microbe = pd.read_csv(microbe_path, sep=',', header=0, lineterminator='\n', skip_blank_lines=True)
Y = np.array(data_microbe.iloc[0:17, 1:])
y_micro = Y[:, 1:]
y_ferti = Y[:, 0].astype(float)
label_names = ['B1', 'B2', 'B3', 'B4']


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



for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]

    for pc_x, pc_y in pc_pairs:
        plt.figure(figsize=(8, 6))
        for i in range(X_tsne.shape[0]):
            plt.scatter(X_tsne[i, pc_x], X_tsne[i, pc_y], color=colors[i],
                        s=sizes[i], edgecolor='black', alpha=0.85)
            plt.text(X_tsne[i, pc_x] + 0.1, X_tsne[i, pc_y], f'T{i+1}', fontsize=7)
        plt.title(f"t-SNE on Cacao Data with Microbe Presence: {label_name}")
        plt.xlabel(f"t-SNE {pc_x + 1}")
        plt.ylabel(f"t-SNE {pc_y + 1}")
        plt.grid(True)
        plt.legend(handles=ferti_legend_elements, title="Fertilization Size",
                   bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1)
        plt.tight_layout()
        plt.show()
        
triplets = [(0, 1, 2)]
for pc1, pc2, pc3 in triplets:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        ax.scatter(X_tsne[idx, pc1], X_tsne[idx, pc2], X_tsne[idx, pc3],
                   label=treatment, color=default_colors[k], s=sizes[idx], edgecolor='black', alpha=0.85)
        for x, y, z, label in zip(X_tsne[idx, pc1], X_tsne[idx, pc2], X_tsne[idx, pc3], Treatments[idx]):
            ax.text(x, y, z + 0.2, label, fontsize=9)
    ax.set_xlabel(f"t-SNE {pc1 + 1}")
    ax.set_ylabel(f"t-SNE {pc2 + 1}")
    ax.set_zlabel(f"t-SNE {pc3 + 1}")
    ax.set_title(f'3D PCA Plot: PC{pc1+1} vs PC{pc2+1} vs PC{pc3+1}')
    ferti_legend = ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                             bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
    ax.add_artist(ferti_legend)
    plt.tight_layout()
    plt.show(block=False)
    
    
       
for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]

    for pc1, pc2, pc3 in triplets:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(X_tsne.shape[0]):
            ax.scatter(X_tsne[i, pc1], X_tsne[i, pc2], X_tsne[i, pc3],
                       color=colors[i], s=sizes[i], edgecolor='black', alpha=0.85)
            ax.text(X_tsne[i, pc1], X_tsne[i, pc2], X_tsne[i, pc3] + 0.2, f'T{i+1}', fontsize=8)
            ax.set_xlabel(f"t-SNE {pc1 + 1}")
            ax.set_ylabel(f"t-SNE {pc2 + 1}")
        ferti_legend = ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                                 bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
        ax.add_artist(ferti_legend)
        plt.tight_layout()
        plt.show(block=False)
        


# UMAP embedding
umap_model = umap.UMAP(n_components=3, n_neighbors=5, random_state=42)
X_umap = umap_model.fit_transform(X)

# --- 2D UMAP plots with fertilization size ---
for i, j in pc_pairs:
    plt.figure(figsize=(8, 6))
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        for x, y, label, ferti_size in zip(X_umap[idx, i], X_umap[idx, j], Treatments[idx], sizes[idx]):
            plt.scatter(x, y, color=default_colors[k], s=ferti_size, edgecolor='black', alpha=0.85)
            plt.text(x + 0.1, y, label, fontsize=9)
    plt.title("UMAP on Cacao Data with Fertilization Size")
    plt.xlabel(f"UMAP {i + 1}")
    plt.ylabel(f"UMAP {j + 1}")
    plt.grid()
    plt.legend(handles=ferti_legend_elements, title="Fertilization Size",
               bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
    plt.tight_layout()
    plt.show()

# --- 2D UMAP plots with microbe presence ---
for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]

    for pc_x, pc_y in pc_pairs:
        plt.figure(figsize=(8, 6))
        for i in range(X_umap.shape[0]):
            plt.scatter(X_umap[i, pc_x], X_umap[i, pc_y], color=colors[i],
                        s=sizes[i], edgecolor='black', alpha=0.85)
            plt.text(X_umap[i, pc_x] + 0.1, X_umap[i, pc_y], f'T{i+1}', fontsize=7)
        plt.title(f"UMAP on Cacao Data with Microbe Presence: {label_name}")
        plt.xlabel(f"UMAP {pc_x + 1}")
        plt.ylabel(f"UMAP {pc_y + 1}")
        plt.grid(True)
        plt.legend(handles=ferti_legend_elements, title="Fertilization Size",
                   bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1)
        plt.tight_layout()
        plt.show()

# --- 3D UMAP plots with fertilization size ---
for pc1, pc2, pc3 in triplets:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        ax.scatter(X_umap[idx, pc1], X_umap[idx, pc2], X_umap[idx, pc3],
                   label=treatment, color=default_colors[k], s=sizes[idx], edgecolor='black', alpha=0.85)
        for x, y, z, label in zip(X_umap[idx, pc1], X_umap[idx, pc2], X_umap[idx, pc3], Treatments[idx]):
            ax.text(x, y, z + 0.2, label, fontsize=9)
    ax.set_xlabel(f"UMAP {pc1 + 1}")
    ax.set_ylabel(f"UMAP {pc2 + 1}")
    ax.set_zlabel(f"UMAP {pc3 + 1}")
    ax.set_title(f'3D UMAP Plot: UMAP{pc1+1} vs UMAP{pc2+1} vs UMAP{pc3+1}')
    ferti_legend = ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                             bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
    ax.add_artist(ferti_legend)
    plt.tight_layout()
    plt.show(block=False)

# --- 3D UMAP plots with microbe presence ---
for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]

    for pc1, pc2, pc3 in triplets:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(X_umap.shape[0]):
            ax.scatter(X_umap[i, pc1], X_umap[i, pc2], X_umap[i, pc3],
                       color=colors[i], s=sizes[i], edgecolor='black', alpha=0.85)
            ax.text(X_umap[i, pc1], X_umap[i, pc2], X_umap[i, pc3] + 0.2, f'T{i+1}', fontsize=8)
            ax.set_xlabel(f"UMAP {pc1 + 1}")
            ax.set_ylabel(f"UMAP {pc2 + 1}")
        ferti_legend = ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                                 bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
        ax.add_artist(ferti_legend)
        plt.tight_layout()
        plt.show(block=False)