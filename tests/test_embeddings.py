import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch.nn.functional import cosine_similarity

import jsonlines
from tqdm import tqdm

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
ROOT_DIR = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
DATA_DIR = Path(os.getenv('DATA_DIR'))
WORK_DIR = Path(os.getenv('WORK_DIR'))
sys.path.append(ROOT_DIR.as_posix())

from utils.model_heplers import load_model_and_tokenizer


def draw_pca_cumulative_variance_plot(cumulative_variance, n_components, model_name):
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(1, n_components + 1)
    plt.plot(x_axis, cumulative_variance, marker='o', linestyle='-', color='blue', label='Cumulative Variance')
    plt.bar(x_axis, cumulative_variance, alpha=0.3, color='gray', label='Individual Variance')

    plt.xlabel('Number of Principal Components (D)', fontsize=12)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), [f'{i*100:.0f}%' for i in np.arange(0, 1.1, 0.1)])
    plt.xticks(np.concatenate(([1], np.arange(10, n_components + 1, 10))))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    fig_dir = ROOT_DIR / "figures/embeddings"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path = fig_dir / f"{model_name}_cumulative_variance.pdf"
    plt.savefig(save_path)
    print(f"\nFigure saved to {save_path}")
    
def check_embedding_vectors(model_name, tokens=["happy", "sad", "up", "down"], use_standarization=False, discard_principal_component=False):
    model, tokenizer = load_model_and_tokenizer(model_name)
    model_name = model_name.split("/")[-1]

    if use_standarization and discard_principal_component:
        raise ValueError("use_standarization and discard_principal_component cannot be True at the same time.")
    
    embeddings = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    print(embeddings.shape)  # [151936, 4096]

    embeddings_2d = None
    if discard_principal_component:
        mu = np.mean(embeddings, axis=0)
        embeddings_centered = embeddings - mu

        # fixed
        # D = 100
        # pca = PCA(n_components=D)
        # embeddings_projected = pca.fit_transform(embeddings_centered)
        # reconstructed_from_D = pca.inverse_transform(embeddings_projected)
        # embeddings_2d = embeddings_centered - reconstructed_from_D

        # adaptively select D
        # n_components = 1000
        # pca = PCA(n_components=n_components)
        # pca.fit(embeddings_centered)
        # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        # # draw_pca_cumulative_variance_plot(cumulative_variance, n_components, model_name)
        # met_threshold_mask = cumulative_variance >= 0.15
        # D_index = np.argmax(met_threshold_mask)
        # D = D_index + 1
        # print(f"D: {D}")

        # d/100
        D = int(embeddings.shape[1] / 100)
        print(f"D: {D}")
        pca = PCA(n_components=D)
        embeddings_projected = pca.fit_transform(embeddings_centered)
        reconstructed_from_D = pca.inverse_transform(embeddings_projected)
        embeddings_2d = embeddings_centered - reconstructed_from_D
    elif use_standarization:
        pass
    else:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

    vectors = {}
    vector_names = []
    for token in tokens:
        token_id = tokenizer(token, add_special_tokens=False).input_ids[0]
        vectors[token] = embeddings_2d[token_id]
        vector_names.append(token)

    # plot the vectors
    plt.figure(figsize=(10, 8))
    for token in tokens:
        plt.plot([0, vectors[token][0]], [0, vectors[token][1]], label=token)
        plt.annotate(token, (vectors[token][0], vectors[token][1]))
    # "happy-sad"
    start_x_hs = vectors["sad"][0]
    start_y_hs = vectors["sad"][1]
    dx_hs = vectors["happy"][0] - vectors["sad"][0]
    dy_hs = vectors["happy"][1] - vectors["sad"][1]
    # "up-down"
    start_x_ud = vectors["down"][0]
    start_y_ud = vectors["down"][1]
    dx_ud = vectors["up"][0] - vectors["down"][0]
    dy_ud = vectors["up"][1] - vectors["down"][1]
    # plot "happy-sad" and "up-down" vectors
    plt.quiver([start_x_hs, start_x_ud], [start_y_hs, start_y_ud], [dx_hs, dx_ud], [dy_hs, dy_ud],
               color=['purple', 'brown'], angles='xy', scale_units='xy', scale=1)
    plt.text(vectors["happy"][0], vectors["happy"][1], 'happy-sad', 
             color='purple', fontsize=12, ha='left', va='bottom')
    plt.text(vectors["up"][0], vectors["up"][1], 'up-down', 
             color='brown', fontsize=12, ha='left', va='bottom')
    
    plt.legend()
    fig_dir = ROOT_DIR / "figures/embeddings"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{model_name}.pdf"
    if use_standarization:
        fig_path = fig_dir / f"{model_name}_standarized.pdf"
    if discard_principal_component:
        fig_path = fig_dir / f"{model_name}_discard_principal_component.pdf"
    plt.savefig(fig_path)

def check_embedding_antonyms(model_name, token_pairs=[("happy", "sad"), ("up", "down")]):
    model, tokenizer = load_model_and_tokenizer(model_name)
    model_name = model_name.split("/")[-1]

    embeddings = model.get_input_embeddings().weight
    print(embeddings.shape)  # [151936, 4096]
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings.detach().cpu().float().numpy())

    vectors = {}
    vector_names = []
    for token_pair in token_pairs:
        for token in token_pair:
            token_id = tokenizer(token, add_special_tokens=False).input_ids[0]
            vectors[token] = embeddings_2d[token_id]
            vector_names.append(token)
    # plot the vectors
    plt.figure(figsize=(10, 8))
    for token in vector_names:
        plt.plot([0, vectors[token][0]], [0, vectors[token][1]], label=token)
        plt.annotate(token, (vectors[token][0], vectors[token][1]))
    # plot antonyms
    for token_pair in token_pairs:
        plt.quiver([vectors[token_pair[1]][0]], [vectors[token_pair[1]][1]], 
                 [vectors[token_pair[0]][0] - vectors[token_pair[1]][0]], 
                 [vectors[token_pair[0]][1] - vectors[token_pair[1]][1]],
                 color='red', linestyle='--', angles='xy', scale_units='xy', scale=1)
        plt.annotate(f"{token_pair[0]}-{token_pair[1]}", 
                     (vectors[token_pair[0]][0], vectors[token_pair[1]][1]))

    plt.legend()
    fig_dir = ROOT_DIR / "figures/embeddings"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{model_name}_antonyms.pdf")

if __name__ == "__main__":
    pass

    model_name = "Qwen/Qwen3-8B"  # "Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-2-7B-hf", 
    check_embedding_vectors(model_name, discard_principal_component=False)
    # check_embedding_antonyms(model_name, token_pairs=[("front", "back"), ("tomorrow", "yesterday"), ("forward", "backward")])
    # check_embedding_antonyms(model_name, token_pairs=[("king", "queen"), ("man", "woman")])