import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
# import plotly.express as px # REMOVED
import matplotlib.pyplot as plt # ADDED
from torch.utils.data import DataLoader

from typing import List, Union, Optional, Callable
from functools import partial
import copy
import itertools
import json

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

import transformer_lens.patching as patching

# ==========================================
# Matplotlib Visualization Helper
# ==========================================
def plot_heatmap(data, title, xlabel, ylabel, x_labels=None, y_labels=None):
    """
    Helper function to plot heatmaps using matplotlib.
    Data should be a 2D numpy array or torch tensor.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    # Use RdBu colormap: Red for Positive, Blue for Negative. 
    # In Noising: Negative (Blue) means the component was important (performance dropped).
    max_val = np.max(np.abs(data))
    plt.imshow(data, cmap='RdBu', aspect='auto', vmin=-max_val, vmax=max_val, origin='lower')
    plt.colorbar(label="Attribution Effect")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if x_labels is not None:
        # Prevent overcrowding if too many labels
        step = max(1, len(x_labels) // 20)
        plt.xticks(np.arange(0, len(x_labels), step), x_labels[::step], rotation=45)
        
    if y_labels is not None:
        step = max(1, len(y_labels) // 20)
        plt.yticks(np.arange(0, len(y_labels), step), y_labels[::step])
        
    plt.tight_layout()
    plt.show()

# ==========================================
# Setup Model and Data
# ==========================================

model = HookedTransformer.from_pretrained("gpt2-small")
model.set_use_attn_result(True)

prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

clean_tokens = model.to_tokens(prompts)
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]

answer_token_indices = torch.tensor(
    [
        [model.to_single_token(answers[i][j]) for j in range(2)]
        for i in range(len(answers))
    ],
    device=model.cfg.device,
)

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()

CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff

def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )

filter_not_qkv_input = lambda name: "_input" not in name

def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )

# ==========================================
# Calculating Gradients
# ==========================================

print("Calculating Clean Gradients (Noising Mode)...")

clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, clean_tokens, ioi_metric
)

corrupted_value, corrupted_cache, _ = get_cache_fwd_and_bwd(
    model, corrupted_tokens, ioi_metric
)

print("Clean Value (Should be 1.0):", clean_value)
print("Corrupted Value (Should be 0.0):", corrupted_value)


# ==========================================
# Patching Functions
# ==========================================

def attr_patch_layer_out(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    clean_grad_cache: ActivationCache, # ### CHANGE: Argument name changed
):
    clean_layer_out, labels = clean_cache.decompose_resid(-1, return_labels=True)
    corrupted_layer_out = corrupted_cache.decompose_resid(-1, return_labels=False)
    
    clean_grad_layer_out = clean_grad_cache.decompose_resid(
        -1, return_labels=False
    )
    
    # add noise: (corrupted - clean) * clean_grad
    layer_out_attr = einops.reduce(
        clean_grad_layer_out * (corrupted_layer_out - clean_layer_out),
        "component batch pos d_model -> component pos",
        "sum",
    )
    return layer_out_attr, labels

# Execute
layer_out_attr, layer_out_labels = attr_patch_layer_out(
    clean_cache, corrupted_cache, clean_grad_cache # Pass clean_grad
)

# Visualize
plot_heatmap(
    layer_out_attr, 
    title="Layer Output Attribution (Noising/Knockout)\nBlue (Negative) = Important Component", 
    xlabel="Position", 
    ylabel="Component",
    y_labels=layer_out_labels
)


# --- Head Output Patching ---

HEAD_NAMES = [
    f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]

def attr_patch_head_out(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    clean_grad_cache: ActivationCache, # ### CHANGE
):
    labels = HEAD_NAMES

    clean_head_out = clean_cache.stack_head_results(-1, return_labels=False)
    corrupted_head_out = corrupted_cache.stack_head_results(-1, return_labels=False)
    
    clean_grad_head_out = clean_grad_cache.stack_head_results(
        -1, return_labels=False
    )
    
    # (corrupted - clean) * clean_grad
    head_out_attr = einops.reduce(
        clean_grad_head_out * (corrupted_head_out - clean_head_out),
        "component batch pos d_model -> component pos",
        "sum",
    )
    return head_out_attr, labels


head_out_attr, head_out_labels = attr_patch_head_out(
    clean_cache, corrupted_cache, clean_grad_cache
)

# Visualize Head Output
plot_heatmap(
    head_out_attr,
    title="Head Output Attribution (Noising)\nBlue (Negative) = Important Head",
    xlabel="Position",
    ylabel="Head (Flattened)",
    # y_labels=head_out_labels # Too many labels, let's skip or sample
)

# Sum over positions
sum_head_out_attr = einops.reduce(
    head_out_attr,
    "(layer head) pos -> layer head",
    "sum",
    layer=model.cfg.n_layers,
    head=model.cfg.n_heads,
)

plot_heatmap(
    sum_head_out_attr,
    title="Head Output Attribution Summed (Noising)\nBlue (Negative) = Important",
    xlabel="Head Index",
    ylabel="Layer"
)

# --- Head Vector (Q, K, V) Patching ---

from typing_extensions import Literal

def stack_head_vector_from_cache(
    cache, activation_name: Literal["q", "k", "v", "z"]
):
    stacked_head_vectors = torch.stack(
        [cache[activation_name, l] for l in range(model.cfg.n_layers)], dim=0
    )
    stacked_head_vectors = einops.rearrange(
        stacked_head_vectors,
        "layer batch pos head_index d_head -> (layer head_index) batch pos d_head",
    )
    return stacked_head_vectors


def attr_patch_head_vector(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    clean_grad_cache: ActivationCache, # ### CHANGE
    activation_name: Literal["q", "k", "v", "z"],
):
    labels = HEAD_NAMES

    clean_head_vector = stack_head_vector_from_cache(clean_cache, activation_name)
    corrupted_head_vector = stack_head_vector_from_cache(
        corrupted_cache, activation_name
    )
    
    clean_grad_head_vector = stack_head_vector_from_cache(
        clean_grad_cache, activation_name
    )
    
    # (corrupted - clean) * clean_grad
    head_vector_attr = einops.reduce(
        clean_grad_head_vector * (corrupted_head_vector - clean_head_vector),
        "component batch pos d_head -> component pos",
        "sum",
    )
    return head_vector_attr, labels


head_vector_attr_dict = {}
for activation_name, activation_name_full in [
    ("k", "Key"),
    ("q", "Query"),
    ("v", "Value"),
    ("z", "Mixed Value"),
]:
    head_vector_attr_dict[activation_name], head_vector_labels = attr_patch_head_vector(
        clean_cache, corrupted_cache, clean_grad_cache, activation_name
    )
    
    # Visualize Sum over Pos
    sum_head_vector_attr = einops.reduce(
        head_vector_attr_dict[activation_name],
        "(layer head) pos -> layer head",
        "sum",
        layer=model.cfg.n_layers,
        head=model.cfg.n_heads,
    )
    
    plot_heatmap(
        sum_head_vector_attr,
        title=f"{activation_name_full} AtP Summed (Noising)\nBlue = Important",
        xlabel="Head Index",
        ylabel="Layer"
    )