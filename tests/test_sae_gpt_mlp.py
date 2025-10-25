import os
import torch
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
import re

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

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

model = HookedSAETransformer.from_pretrained("gpt2-small", device = device)
# print(model.cfg)

# About SAEs for GPT-2 Small: https://jbloomaus.github.io/SAELens/latest/sae_table/
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-mlp-tm",
    sae_id = "blocks.7.hook_mlp_out",
    device = device
)
print(cfg_dict)  # d_in: 768, d_sae: 24576,
# print(f"Sparsity: {sparsity}")  # None

# feature_directions = sae.W_dec
# print("sae.W_dec:", feature_directions.shape)  # [24576, 768]
# breakpoint()


def get_dashboard_html(sae_release = "gpt2-small", sae_id="7-res-jb", feature_idx=0):
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
# prompt_happy += prompt_excited

name_to_prompts = {
    "happy": prompt_happy,
    "sad": prompt_sad,
    "excited": prompt_excited,
    "up": prompt_up,
}
name_to_keywords = {
    "happy": [" happy", " glad"],
    "sad": [" sad", " upset"],
    "excited": [" excited"],
    "up": [" up"],
}

# tokens_happy = model.to_str_tokens(prompt_happy)
# for i, tokens in enumerate(tokens_happy):
#     print(f"Prompt {i}: {tokens}")


# get the positions of the keywords
def get_keyword_id(prompts: list, keywords: list):
    keyword_idxs = []
    prompt_tokens_list = model.to_str_tokens(prompts)
    for prompt_tokens in prompt_tokens_list:
        for keyword in keywords:
            if keyword in prompt_tokens:
                keyword_idxs.append(prompt_tokens.index(keyword))
                break
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


# Analyze SAE activations for different prompts
sae_release = "gpt2-small-mlp-tm"
sae_id = "blocks.7.hook_mlp_out"
analyze_sae_activations(model, sae_release, sae_id, prompt_name="happy")
analyze_sae_activations(model, sae_release, sae_id, prompt_name="up")
analyze_sae_activations(model, sae_release, sae_id, prompt_name="sad")



