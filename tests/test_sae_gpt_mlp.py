import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
import re
import jsonlines

from datasets import load_dataset  
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt, tokenize_and_concatenate
from sae_lens import SAE, HookedSAETransformer

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(root_dir.as_posix())

from utils.plot_helpers import format_plotly_figure


torch.set_grad_enabled(False)


# model = HookedSAETransformer.from_pretrained("gpt2-small", device = device)
# print(model.cfg)

# About SAEs for GPT-2 Small: https://jbloomaus.github.io/SAELens/latest/sae_table/
# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release = "gpt2-small-mlp-tm",
#     sae_id = "blocks.7.hook_mlp_out",
#     device = device
# )
# print(cfg_dict)  # d_in: 768, d_sae: 24576,
# print(f"Sparsity: {sparsity}")  # None

# feature_directions = sae.W_dec
# print("sae.W_dec:", feature_directions.shape)  # [24576, 768]
# breakpoint()


def get_dashboard_html(sae_release = "gpt2-small", sae_id="7-res-jb", feature_idx=0):
    # https://www.neuronpedia.org/gpt2-small/7-res-jb/3121?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300
    # https://neuronpedia.org/gpt2-small/7-mlp-tm/3121?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return html_template.format(sae_release, sae_id, feature_idx)


prompt_happy = [
    "I'll be on a vacation tommorow and I'm so happy.",
    "My mom brings home a new puppy and I'm so happy.",
    "I'm so glad I got the job I wanted.",
    "I feel so happy when I'm with my friends.",
    "I'm so happy I got the promotion I wanted.",
]
prompt_sad = [
    "I'm so sad I lost my wallet.",
    "I broke up with my boyfriend and I'm so sad.",
    "I am so sad now that I want to cry.",
    "I faied my exam and I'm so upset.",
    "She is upset by the terrible news"
]
prompt_excited = [
    "I'm so excited to see my friends.",
    "The concert of my favorite band is tonight and I'm so excited.",
    "I am so excited to go to the party tonight.",
    "I'm so excited to go on a date with my crush.",
    "My parents are coming to visit me and I'm so excited."
]
prompt_up = [
    "The bird flew up into the sky.",
    "He looked up at the stars.",
    "She climbed up the mountain.",
    "The balloon floated up into the air.",
    "The rocket blasted up into space.",
]
prompt_down = [
    "The flag flew down into the ground.",
    "He looked down at the ground.",
    "The leaves fell down onto the ground.",
    "A bottle of water fell down onto the ground."
    "The elevator is going down.",
]
# prompt_happy += prompt_excited

name_to_keywords = {
    "happy": [" happy", " glad"],
    "sad": [" sad", " upset"],
    "excited": [" excited"],
    "up": [" up"],
    "down": [" down"],
}

# tokens_happy = model.to_str_tokens(prompt_happy)
# for i, tokens in enumerate(tokens_happy):
#     print(f"Prompt {i}: {tokens}")

language = "en"
data_dir = root_dir / f"data/data_various_context/{language}"
concepts = ["emotion_positive", "emotion_negative", "orientation_positive", "orientation_negative"]
data = {}
for concept in concepts:
    concept_data_path = data_dir / f"{concept}.jsonl"
    with jsonlines.open(concept_data_path) as f:
        data[concept] = {
            "sentences": [],
            "target": []
        }
        for line in f:
            data[concept]["sentences"].append(line["sentence"])
            data[concept]["target"].append(line["keyword"])

batchsize=32
prompt_happy = data["emotion_positive"]["sentences"][:batchsize]
prompt_sad = data["emotion_negative"]["sentences"][:batchsize]
prompt_up = data["orientation_positive"]["sentences"][:batchsize]
prompt_down = data["orientation_negative"]["sentences"][:batchsize]

name_to_keywords = {
    "happy": [data["emotion_positive"]["target"][i] for i in range(batchsize)],
    "sad": [data["emotion_negative"]["target"][i] for i in range(batchsize)],
    "up": [data["orientation_positive"]["target"][i] for i in range(batchsize)],
    "down": [data["orientation_negative"]["target"][i] for i in range(batchsize)],
}

name_to_prompts = {
    "happy": prompt_happy,
    "sad": prompt_sad,
    "excited": prompt_excited,
    "up": prompt_up,
    "down": prompt_down,
}

# get the positions of the keywords
def get_keyword_id_old(prompts: list, keywords: list):
    keyword_idxs = []
    prompt_tokens_list = model.to_str_tokens(prompts)
    for prompt_tokens in prompt_tokens_list:
        for keyword in keywords:
            if keyword in prompt_tokens:
                keyword_idxs.append(prompt_tokens.index(keyword))
                break
    return keyword_idxs


def get_keyword_id(prompts: list, keywords: list):
    prompt_tokens_list = model.to_str_tokens(prompts)
    keyword_idxs = []
    for prompt_tokens, keyword in zip(prompt_tokens_list, keywords):
        if f" {keyword}" in prompt_tokens:
            keyword_idxs.append(prompt_tokens.index(f" {keyword}"))
        else:
            print(f"Keyword {keyword} not found in prompt {prompt_tokens}")
    return keyword_idxs


# Get the feature activations from the SAE
def analyze_sae_activations(
    model: HookedSAETransformer,
    sae_release: str | List[str],
    sae_id: str | List[str],
    prompt_name: str,
):  
    saes = []
    if isinstance(sae_release, str):
        sae_release, sae_id = [sae_release], [sae_id]
    for release, id in zip(sae_release, sae_id):
        sae, _, _ = SAE.from_pretrained(
            release = release,
            sae_id = id,
            device = device
        )
        saes.append(sae)
    prompts = name_to_prompts[prompt_name]
    _, cache = model.run_with_cache_with_saes(prompts, saes=saes)
    # for k,v in cache.items():
    #     # if "sae" in k:
    #         print(k, v.shape)  # [bsz, len, dim]
    # breakpoint()

    for release, id in zip(sae_release, sae_id):
        # get layer id from sae_id
        layer_id = int(re.findall(r'blocks\.(\d+)\.', id)[0])
        acts = [f"blocks.{layer_id}.mlp.hook_post"]
        sae_pos = f"blocks.{layer_id}.hook_mlp_out"
        sae_acts = ["hook_sae_input", "hook_sae_acts_post"]
        acts = []
        sae_acts = ["hook_sae_acts_post"]
        act_names = acts + [f"{sae_pos}.{a}" for a in sae_acts]
        keyword_idxs = get_keyword_id(prompts, name_to_keywords[prompt_name])
        for act_name in act_names:
            cache_to_study = cache[act_name]  # [bsz, len, dim]
            batch_idxs = torch.arange(len(prompts)).to(device)
            cache_filtered = cache_to_study[batch_idxs, keyword_idxs, :]  # [bsz, dim]
            cache_average = cache_filtered.mean(dim=0).squeeze()  # [dim]

            data_list = cache_average.tolist()
            x_values = list(range(len(data_list)))
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=x_values,
                        y=data_list,
                        mode='lines',
                        name="Activation values",
                        line=dict(color='green', width=2)
                    )
                ]
            )
            fig = format_plotly_figure(fig)
            img_dir = root_dir / f"figures/sae_{release.replace('-', '_')}/layer_{layer_id}"
            img_dir.mkdir(parents=True, exist_ok=True)
            save_path = img_dir / f"{prompt_name}_{act_name}.pdf"
            fig.write_image(save_path.as_posix())
            top_acts, top_feat_ids = torch.topk(cache_average, 5)
            print(f"{prompt_name} - {act_name} top features: {top_feat_ids}")  # [tensor of feature ids]

            # for val, id in zip(top_acts, top_feat_ids):
            #     print(f"Feature {id} fired {val:.2f}")
            #     html = get_dashboard_html(sae_release = "gpt2-small", sae_id="7-res-jb", feature_idx=id)
            #     # display(IFrame(html, width=1200, height=300))

    return cache

# _, cache_happy = model.run_with_cache_with_saes(prompt_happy, saes=[sae])
# for k,v in cache_happy.items():
#     if "sae" in k:
#         print(k, v.shape)  # [bsz, len, dim]
# breakpoint()
# blocks.7.hook_mlp_out.hook_sae_input torch.Size([10, 17, 768])
# blocks.7.hook_mlp_out.hook_sae_acts_pre torch.Size([10, 17, 24576])
# blocks.7.hook_mlp_out.hook_sae_acts_post torch.Size([10, 17, 24576])
# blocks.7.hook_mlp_out.hook_sae_recons torch.Size([10, 17, 768])
# blocks.7.hook_mlp_out.hook_sae_output torch.Size([10, 17, 768])

def compute_mlp_activations(
    model: HookedSAETransformer,
    sae_release: str | List[str],
    sae_id: str | List[str],
    prompt_name: str,
):  
    saes = []
    if isinstance(sae_id, str):
        sae_release, sae_id = [sae_release], [sae_id]
    for id in sae_id:
        sae, _, _ = SAE.from_pretrained(
            release = sae_release,
            sae_id = id,
            device = device
        )
        saes.append(sae)
    prompts = name_to_prompts[prompt_name]
    _, cache = model.run_with_cache_with_saes(prompts, saes=saes)
    # for k,v in cache.items():
    #     # if "sae" in k:
    #         print(k, v.shape)  # [bsz, len, dim]
    # breakpoint()

    caches_all_layers = []
    for release, id in zip(sae_release, sae_id):
        # get layer id from sae_id
        layer_id = int(re.findall(r'blocks\.(\d+)\.', id)[0])
        acts = [f"blocks.{layer_id}.mlp.hook_post"]
        sae_pos = f"blocks.{layer_id}.hook_mlp_out"
        sae_acts = []
        act_names = acts + [f"{sae_pos}.{a}" for a in sae_acts]
        keyword_idxs = get_keyword_id(prompts, name_to_keywords[prompt_name])
        for act_name in act_names:
            cache_to_study = cache[act_name]  # [bsz, len, dim]
            batch_idxs = torch.arange(len(prompts)).to(device)
            cache_filtered = cache_to_study[batch_idxs, keyword_idxs, :]  # [bsz, dim]
            cache_average = cache_filtered.mean(dim=0).squeeze()  # [dim]
            caches_all_layers.append(cache_average)
    return caches_all_layers

def calc_euclidean_distance(v1, v2):
    """
    Compute the Euclidean distance (L2 norm) between two activation vectors.
    
    Input:
    - v1 (torch.Tensor): Activation vector of shape [1, d_mlp]
    - v2 (torch.Tensor): Activation vector of shape [1, d_mlp]
    
    Returns:
    - torch.Tensor: A scalar tensor representing the Euclidean distance between v1 and v2.
    """
    # torch.dist 计算p范数距离，p=2即为欧几里得距离
    return torch.dist(v1, v2, p=2)

def calc_js_divergence(v1, v2, temperature=1.0):
    """
    Compute the Jensen-Shannon divergence (JS divergence) between two activation vectors.
    In the calculation of softmax, a temperature parameter (temperature) is introduced,
    with T > 1.0 making the distribution smoother, and T < 1.0 making the distribution sharper.
    Default T=1.0.
    
    Input:
    - v1 (torch.Tensor): Activation vector of shape [1, d_mlp]
    - v2 (torch.Tensor): Activation vector of shape [1, d_mlp]
    - temperature (float): Temperature parameter for softmax, default is 1.0
    
    Returns:
    - torch.Tensor: A scalar tensor representing the Jensen-Shannon divergence between v1 and v2.
    """

    log_p = F.log_softmax(v1 / temperature, dim=-1)
    log_q = F.log_softmax(v2 / temperature, dim=-1)
    
    p = log_p.exp()
    q = log_q.exp()
    
    m = 0.5 * (p + q)
    log_m = m.log()
    
    # 计算 KL 散度 D_KL(P || M) 和 D_KL(Q || M)
    # F.kl_div(input, target) 期望 input 是 log 概率，target 是 概率
    # 它计算的是 D_KL(target || input) 的一种形式，所以我们需要调整
    # 我们用 F.kl_div(log_m, p) 来计算 D_KL(P || M)
    # reduction='batchmean' 在这里等价于 'sum' / batch_size。
    # 因为 batch_size=1，所以 'batchmean' 和 'sum' 效果一样。
    # log_target=False 表示 target (p, q) 不是 log 概率
    kl_p_m = F.kl_div(log_m, p, reduction='batchmean', log_target=False)
    kl_q_m = F.kl_div(log_m, q, reduction='batchmean', log_target=False)
    
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd

def calc_pearson_correlation(v1, v2):
    """
    Compute the Pearson correlation coefficient between two activation vectors.
    
    Input:
    - v1 (torch.Tensor): Activation vector of shape [1, d_mlp]
    - v2 (torch.Tensor): Activation vector of shape [1, d_mlp]
    
    Returns:
    - torch.Tensor: A scalar tensor representing the Pearson correlation coefficient between v1 and v2.
                      The value ranges from -1 to 1. A value close to 1 indicates a strong positive correlation,
                      while a value close to -1 indicates a strong negative correlation. A value close to 0 indicates
                      no correlation.
    """
    # Remove the batch dimension [1, ...] to get [d_mlp]
    v1_s = v1.squeeze()
    v2_s = v2.squeeze()
    
    # Center the vectors (subtract mean)
    v1_c = v1_s - v1_s.mean()
    v2_c = v2_s - v2_s.mean()
    
    # Compute covariance (dot product)
    covariance = (v1_c * v2_c).sum()
    
    # Compute product of standard deviations
    std_dev_prod = torch.sqrt((v1_c**2).sum()) * torch.sqrt((v2_c**2).sum())
    
    # Prevent division by zero
    if std_dev_prod == 0:
        return torch.tensor(0.0)
        
    return covariance / std_dev_prod

    # # 另一种更简洁的 PyTorch 方法
    # # 将两个向量堆叠成 [2, d_mlp] 的张量
    # stacked_vecs = torch.stack([v1_s, v2_s])
    # # torch.corrcoef 会返回一个 2x2 的相关矩阵
    # corr_matrix = torch.corrcoef(stacked_vecs)
    # # 我们需要的是非对角线元素 [0, 1] 或 [1, 0]
    # return corr_matrix[0, 1]


if __name__ == "__main__":
    pass

    device = "cuda:1"
    model = HookedSAETransformer.from_pretrained("gpt2-small", device = device)

    # Analyze SAE activations for different prompts
    # sae_release = "gpt2-small-mlp-tm"
    # sae_id = "blocks.7.hook_mlp_out"
    # analyze_sae_activations(model, sae_release, sae_id, prompt_name="happy")
    # analyze_sae_activations(model, sae_release, sae_id, prompt_name="up")
    # analyze_sae_activations(model, sae_release, sae_id, prompt_name="sad")


    # Analyze MLP activations for different prompts
    sae_release = "gpt2-small-mlp-tm"
    sae_id = [f"blocks.{layer_id}.hook_mlp_out" for layer_id in range(model.cfg.n_layers)]
    caches_happy = compute_mlp_activations(model, sae_release, sae_id, prompt_name="happy")
    caches_up = compute_mlp_activations(model, sae_release, sae_id, prompt_name="up")
    caches_sad = compute_mlp_activations(model, sae_release, sae_id, prompt_name="sad")
    caches_down = compute_mlp_activations(model, sae_release, sae_id, prompt_name="down")

    caches_happy = [data.detach().cpu() for data in caches_happy]
    caches_up = [data.detach().cpu() for data in caches_up]
    caches_sad = [data.detach().cpu() for data in caches_sad]
    caches_down = [data.detach().cpu() for data in caches_down]
        
    euclidean_dists_happy_up = []
    js_divergences_happy_up = []
    pearson_correlations_happy_up = []
    euclidean_dists_sad_up = []
    js_divergences_sad_up = []
    pearson_correlations_sad_up = []
    euclidean_dists_happy_down = []
    js_divergences_happy_down = []
    pearson_correlations_happy_down = []
    euclidean_dists_sad_down = []
    js_divergences_sad_down = []
    pearson_correlations_sad_down = []

    for layer_id in range(model.cfg.n_layers):
        act_happy = caches_happy[layer_id]
        act_up = caches_up[layer_id]
        act_sad = caches_sad[layer_id]
        act_down = caches_down[layer_id]

        euclidean_dists_happy_up.append(calc_euclidean_distance(act_happy, act_up))
        js_divergences_happy_up.append(calc_js_divergence(act_happy, act_up))
        pearson_correlations_happy_up.append(calc_pearson_correlation(act_happy, act_up))

        euclidean_dists_sad_up.append(calc_euclidean_distance(act_sad, act_up))
        js_divergences_sad_up.append(calc_js_divergence(act_sad, act_up))
        pearson_correlations_sad_up.append(calc_pearson_correlation(act_sad, act_up))

        euclidean_dists_happy_down.append(calc_euclidean_distance(act_happy, act_down))
        js_divergences_happy_down.append(calc_js_divergence(act_happy, act_down))
        pearson_correlations_happy_down.append(calc_pearson_correlation(act_happy, act_down))

        euclidean_dists_sad_down.append(calc_euclidean_distance(act_sad, act_down))
        js_divergences_sad_down.append(calc_js_divergence(act_sad, act_down))
        pearson_correlations_sad_down.append(calc_pearson_correlation(act_sad, act_down))

    # plot the results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(euclidean_dists_happy_up, label="happy-up")
    plt.plot(euclidean_dists_sad_up, label="sad-up")
    plt.legend()
    plt.title("Euclidean Distance")

    plt.subplot(2, 3, 2)
    plt.plot(js_divergences_happy_up, label="happy-up")
    plt.plot(js_divergences_sad_up, label="sad-up")
    plt.legend()
    plt.title("JS Divergence")

    plt.subplot(2, 3, 3)
    plt.plot(pearson_correlations_happy_up, label="happy-up")
    plt.plot(pearson_correlations_sad_up, label="sad-up")
    plt.legend()
    plt.title("Pearson Correlation")

    plt.subplot(2, 3, 4)
    plt.plot(euclidean_dists_happy_down, label="happy-down")
    plt.plot(euclidean_dists_sad_down, label="sad-down")
    plt.legend()
    plt.title("Euclidean Distance")

    plt.subplot(2, 3, 5)
    plt.plot(js_divergences_happy_down, label="happy-down")
    plt.plot(js_divergences_sad_down, label="sad-down")
    plt.legend()
    plt.title("JS Divergence")

    plt.subplot(2, 3, 6)
    plt.plot(pearson_correlations_happy_down, label="happy-down")
    plt.plot(pearson_correlations_sad_down, label="sad-down")
    plt.legend()
    plt.title("Pearson Correlation")

    fig_dir = root_dir / "figures/mlp_activations"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{sae_release}.pdf")
