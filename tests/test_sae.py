import os
import torch
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import load_dataset  
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt, tokenize_and_concatenate
from sae_lens import SAE, HookedSAETransformer

from pathlib import Path


torch.set_grad_enabled(False)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

model = HookedSAETransformer.from_pretrained("gpt2-small", device = device)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb",
    sae_id = "blocks.7.hook_resid_pre",
    device = device
)

print(model.cfg, sae.cfg)


def get_dashboard_html(sae_release = "gpt2-small", sae_id="7-res-jb", feature_idx=0):
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return html_template.format(sae_release, sae_id, feature_idx)


prompt_happy = ["I'll be on a vacation tommorow and I'm so happy.",
          "My mom brings home a new puppy and I'm so happy.",
          "I'm so glad I got the job I wanted.",
          "I feel so happy when I'm with my friends.",
          "I'm so happy I got the promotion I wanted.",]
prompt_sad = ["I'm so sad I lost my wallet.",
          "I broke up with my boyfriend and I'm so sad.",
          "I am so sad now that I want to cry.",
          "I faied my exam and I'm so upset.",
          "She is upset by the terrible news"]
prompt_excited = ["I'm so excited to see my friends.",
                "The concert of my favorite band is tonight and I'm so excited.",
                "I am so excited to go to the party tonight.",
                "I'm so excited to go on a date with my crush.",
                "My parents are coming to visit me and I'm so excited."]

tokens_happy = model.to_str_tokens(prompt_happy)
for i, tokens in enumerate(tokens_happy):
    print(f"Prompt {i}: {tokens}")


# # get the positions of the keywords
def get_keyword_id(prompts: list, keywords: list):
    keyword_idxs = []
    prompt_tokens_list = model.to_str_tokens(prompts)
    for prompt_tokens in prompt_tokens_list:
        for keyword in keywords:
            if keyword in prompt_tokens:
                keyword_idxs.append(prompt_tokens.index(keyword))
                break
    return keyword_idxs

happy_idxs = get_keyword_id(prompt_happy, [" happy", " glad"])
sad_idxs = get_keyword_id(prompt_sad, [" sad", " upset"])
excited_idxs = get_keyword_id(prompt_excited, [" excited"])
print(happy_idxs)
print(sad_idxs)
print(excited_idxs)

# # Get the feature activations from the SAE
_, cache_happy = model.run_with_cache_with_saes(prompt_happy, saes=[sae])

for k,v in cache_happy.items():
    if "sae" in k:
        print(k, v.shape)

cache_to_study = cache_happy['blocks.7.hook_resid_pre.hook_sae_acts_post']
# Select the activations of the happy keywords
idx_tensor = torch.tensor(happy_idxs).unsqueeze(1).unsqueeze(2).expand(-1, 1, cache_to_study.shape[2])
print(idx_tensor.shape)
cache_happy_filtered = torch.gather(cache_to_study, 1, idx_tensor)
print(cache_to_study.shape)
cache_happy_average = cache_happy_filtered.mean(dim=0).squeeze()
print(cache_happy_average.shape)

px.line(
    cache_happy_average.tolist(),
    title="Feature activations at the keyword for 'happy'",
    labels={"index": "Feature", "value": "Activation"},
).show()
happy_acts, happy_feat_ids = torch.topk(cache_happy_average, 3)
print(happy_feat_ids)

# for val, id in zip(happy_acts, happy_feat_ids):
#     print(f"Feature {id} fired {val:.2f}")
#     html = get_dashboard_html(sae_release = "gpt2-small", sae_id="7-res-jb", feature_idx=id)
#     # display(IFrame(html, width=1200, height=300))