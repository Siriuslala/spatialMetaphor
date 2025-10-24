import os
import torch
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import load_dataset  
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt, tokenize_and_concatenate
from sae_lens import SAE, HookedSAETransformer

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))


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
    release = "gpt2-small-res-jb",
    sae_id = "blocks.7.hook_resid_pre",
    device = device
)
print(cfg_dict)  # d_in: 768, d_sae: 24576,
# print(f"Sparsity: {sparsity}")  # None


def get_dashboard_html(sae_release = "gpt2-small", sae_id="7-res-jb", feature_idx=0):
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
prompt_excited = ["I'm so excited to see my friends.",
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

prompt_happy += prompt_excited

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
up_idxs = get_keyword_id(prompt_up, [" up"])
print(happy_idxs)  # (5,)
print(sad_idxs)
print(excited_idxs)
print(up_idxs)

# # Get the feature activations from the SAE
_, cache_happy = model.run_with_cache_with_saes(prompt_happy, saes=[sae])

for k,v in cache_happy.items():
    if "sae" in k:
        print(k, v.shape)  # [bsz, len, dim]
# breakpoint()
# blocks.7.hook_resid_pre.hook_sae_input torch.Size([5, 17, 768])
# blocks.7.hook_resid_pre.hook_sae_acts_pre torch.Size([5, 17, 24576])
# blocks.7.hook_resid_pre.hook_sae_acts_post torch.Size([5, 17, 24576])
# blocks.7.hook_resid_pre.hook_sae_recons torch.Size([5, 17, 768])
# blocks.7.hook_resid_pre.hook_sae_output torch.Size([5, 17, 768])

# Select the activations of the happy keywords
cache_to_study = cache_happy['blocks.7.hook_resid_pre.hook_sae_acts_post']  # [5, 17, 24576]
batch_idxs = torch.arange(len(happy_idxs)).to(device)
cache_happy_filtered = cache_to_study[batch_idxs, happy_idxs, :]  # [5, 24576]
cache_happy_average = cache_happy_filtered.mean(dim=0).squeeze()
print(cache_happy_average.shape)

# fig = px.line(
#     cache_happy_average.tolist(),
#     # title="Feature activations at the keyword for 'happy'",
#     labels={"index": "SAE feature IDs", "value": "Activation values"},
#     color_discrete_sequence=["green"]
# )

data_list = cache_happy_average.tolist()
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
fig.update_layout(
    # title="Feature activations at the keyword for 'happy'",
    xaxis_title="SAE feature IDs",
    yaxis_title="Activation values",
    margin=dict(
        l=5,  # 左边距 (Left)
        r=5,  # 右边距 (Right)
        b=5,  # 下边距 (Bottom)
        t=5,  # 上边距 (Top)
        pad=4  # 内部填充 (Padding)
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
)
fig.update_xaxes(showline=False, showgrid=True, gridcolor='lightgrey', gridwidth=1, zeroline=True, zerolinewidth=1, zerolinecolor='black', autorange=True, anchor='y', side='bottom', ticks='outside', ticklen=5, ticklabelposition='outside')
fig.update_yaxes(showline=False, showgrid=True, gridcolor='lightgrey', gridwidth=1, zeroline=True, zerolinewidth=1, zerolinecolor='black', autorange=True, anchor='x', side='left', ticks='outside', ticklen=5, ticklabelposition='outside')

img_dir = root_dir / "figures"
save_path = img_dir / "sae_feats-gpt2_layer_7_res_pre-happy.pdf"
fig.write_image(save_path.as_posix())
happy_acts, happy_feat_ids = torch.topk(cache_happy_average, 5)
print(happy_feat_ids)  # [2392,  9840, 21753]

# for val, id in zip(happy_acts, happy_feat_ids):
#     print(f"Feature {id} fired {val:.2f}")
#     html = get_dashboard_html(sae_release = "gpt2-small", sae_id="7-res-jb", feature_idx=id)
#     # display(IFrame(html, width=1200, height=300))


# Same for "up"
_, cache_up = model.run_with_cache_with_saes(prompt_up, saes=[sae])
for k,v in cache_up.items():
    if "sae" in k:
        print(k, v.shape)  # [bsz, len, dim]
# Select the activations of the "up" keywords
cache_to_study = cache_up['blocks.7.hook_resid_pre.hook_sae_acts_post']  # [5, 9, 24576]
batch_idxs = torch.arange(len(prompt_up)).to(device)
up_idxs = get_keyword_id(prompt_up, [" up"])
cache_up_filtered = cache_to_study[batch_idxs, up_idxs, :]  # [5, 24576]
cache_up_average = cache_up_filtered.mean(dim=0).squeeze()
print(cache_up_average.shape)
data_list = cache_up_average.tolist()
x_values = list(range(len(data_list)))
fig = go.Figure(
    data=[
        go.Scatter(
            x=x_values,
            y=data_list,
            mode='lines',
            name="Activation values",
            line=dict(color='blue', width=2)
        )
    ]
)
fig.update_layout(
    # title="Feature activations at the keyword for 'up'",
    xaxis_title="SAE feature IDs",
    yaxis_title="Activation values",
    margin=dict(
        l=5,  # 左边距 (Left)
        r=5,  # 右边距 (Right)
        b=5,  # 下边距 (Bottom)
        t=5,  # 上边距 (Top)
        pad=4  # 内部填充 (Padding)
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
)
fig.update_xaxes(showline=False, showgrid=True, gridcolor='lightgrey', gridwidth=1, zeroline=True, zerolinewidth=1, zerolinecolor='black', autorange=True, anchor='y', side='bottom', ticks='outside', ticklen=5, ticklabelposition='outside')
fig.update_yaxes(showline=False, showgrid=True, gridcolor='lightgrey', gridwidth=1, zeroline=True, zerolinewidth=1, zerolinecolor='black', autorange=True, anchor='x', side='left', ticks='outside', ticklen=5, ticklabelposition='outside')
img_dir = root_dir / "figures"
save_path = img_dir / "sae_feats-gpt2_layer_7_res_pre-up.pdf"
fig.write_image(save_path.as_posix())
up_acts, up_feat_ids = torch.topk(cache_up_average, 5)
print(up_feat_ids)  # [13779,  9823, 19956]


