import jsonlines
import numpy as np
import einops
import textwrap
from typing import Literal
import plotly.express as px
from functools import partial
import dataclasses
import gc
import pandas as pd

import torch
import torch.nn as nn
import safetensors
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download
import sae_lens
from sae_lens import SAE, HookedSAETransformer, SAEConfig  # pip install -U sae-lens

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
ROOT_DIR = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
DATA_DIR = Path(os.getenv('DATA_DIR'))
WORK_DIR = Path(os.getenv('WORK_DIR'))
sys.path.append(ROOT_DIR.as_posix())

from tools.sae.test_gemma_scope_2 import load_sae_gemma, load_transcoder_gemma
from utils.model_heplers import format_prompt_gemma, get_target_token_position
from tools.hook import (
    gather_acts_hook,
    gather_residual_activations,
    gather_transcoder_activations,
)
from utils.plot_helpers import (
    generate_token_activation_map,
    generate_multi_token_activation_maps
)

# monkey patch
from utils.monkey_patch.patch_sae_lens import patched_gemma_3_sae_huggingface_loader
sae_lens.loading.pretrained_sae_loaders.NAMED_PRETRAINED_SAE_LOADERS["gemma_3"] = patched_gemma_3_sae_huggingface_loader


def get_prompts(
    theme: Literal["emotion", "space"],
    target: Literal["happy", "up"],
) -> list[str]:

    if theme == "emotion":
        if target == "happy":
            prompts = [
                "He just won the game and is feeling happy."
            ]
            target_tokens = ["happy"]
        else:
            prompts = [
                "He just lost the game and is feeling sad."
            ]
            target_tokens = ["sad"]
    elif theme == "space":
        if target == "up":
            prompts = [
                "The eagle is in the sky, so the child looked up.",
                "To get to the rooftop, the man walked up the stairs.",
                "The movie was excellent, so he gave it a thumbs up."
            ]
            target_tokens = ["up", "up", "up"]
        else:
            prompts = [
                "The mouse is on the floor, so the child looked down.",
                "To get to the basement, the man walked down the stairs.",
                "The movie was terrible, so he gave it a thumbs down."
            ]
            target_tokens = ["down", "down", "down"]
    else:
        raise ValueError(f"Unknown theme: {theme}")
    
    return prompts, target_tokens

def find_gemma_sae_features(
    model_name,
    theme: Literal["emotion", "space"],
    target: Literal["happy", "up"],
):
    pass

def find_gemma_transcoder_features(
    model_name,
    theme: Literal["emotion", "space"],
    target: Literal["happy", "up"],
    topk: int = 10,
    device: str = "cuda",
    **transcoder_kwargs,
):
    """
    Choice 1: use transcoder at 4 layers, with l0 medium
    Choice 2: use transcoder at all layers, 16k, with l0 large
    """
    # load models
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # get inputs
    prompts, target_tokens = get_prompts(theme, target)
    target_token_positions = []
    for i in range(len(prompts)):
        tokens = tokenizer.tokenize(prompts[i], add_special_tokens=True)
        target_token_pos = get_target_token_position(tokens, target_tokens[i])
        target_token_positions.append(target_token_pos)
        print(f"target token: {target_tokens[i]} at position {target_token_pos}")
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=True).to(device)
    # breakpoint()

    # forward and gather activations
    layer_ids = transcoder_kwargs["layer"]
    cache = gather_transcoder_activations(model, layer_ids, inputs)

    # get features for each layer
    res_dir = ROOT_DIR / "figures/transcoder_gemma"
    if target is not None and target != "":
        scene = f"{theme}_{target}"
    else:
        scene = theme
    res_path = res_dir / f"{scene}.jsonl"

    for layer_id in layer_ids:

        # load transcoder
        tc_kwargs = transcoder_kwargs.copy()
        tc_kwargs["layer"] = layer_id
        transcoder = load_transcoder_gemma(
            device=device,
            **tc_kwargs,
        )

        # get features
        tc_input = cache[f"transcoder_input_layer{layer_id}"]
        features = transcoder.encode(tc_input)  # (batch_size, seq_len, d_model)
        
        ## get features fire at the target token position
        target_token_acts = features[torch.arange(len(prompts)), target_token_positions, :]  # (batch_size, d_model)
        print(f"target token acts shape: {target_token_acts.shape}")
        top_feat_acts_target_token, top_feat_ids_target_token = torch.topk(target_token_acts, topk, dim=-1)
        for i in range(len(prompts)):
            print(f"top {topk} features for prompt {i} at the target token position: {top_feat_ids_target_token[i].tolist()}")

        ## get features that activate the strongest when averaged over all tokens in the sequence
        top_feat_acts, top_feat_ids = features.mean(1).topk(topk)  # (batch_size, topk)
        for i in range(len(prompts)):
            print(f"top {topk} features for prompt {i}: {top_feat_ids[i].tolist()}")

        with jsonlines.open(res_path, "a") as f:
            f.write({
                "tc_meta": tc_kwargs,
                "prompts": prompts,
                "target_tokens": target_tokens,
                "top_feats_at_target_token": top_feat_ids_target_token.tolist(),
                "top_feats_at_all_tokens": top_feat_ids.tolist(),
            })
        
    

if __name__ == "__main__":

    if True:
        model_name = "google/gemma-3-1b-pt"
        transcoder_kwargs = {
            "repo_id": "google/gemma-scope-2-1b-pt",  # "google/gemma-scope-2-1b-pt", "google/gemma-scope-2-1b-it"
            "transcoder_pos": "transcoder",
            "release": "gemma-scope-2-1b-pt-transcoders",
            "layer": [7, 13, 17, 22],
            "width": "16k",  # 65k, 262k
            "L0": "medium",
        }
        find_gemma_transcoder_features(
            model_name,
            theme="space",
            target="up",
            topk=10,
            device="cuda:5",
            **transcoder_kwargs,
        )