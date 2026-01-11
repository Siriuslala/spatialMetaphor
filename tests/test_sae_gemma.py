from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download
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

from sae_lens import SAE  # pip install -U sae-lens


def test_gemma(device="cuda"):
    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-pt",
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

    # test prompt
    prompt = "The law of conservation of energy states that energy cannot be created or destroyed, only transformed."
    # Note that this implicitly adds a special "Beginning of Sequence" or <bos> token to the start
    tokens = tokenizer.tokenize(prompt)
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    print(inputs)
    print(tokens)

    # generate text
    outputs = model.generate(input_ids=inputs, max_new_tokens=50)
    output_str = tokenizer.decode(outputs[0][inputs.shape[1]:])
    print(textwrap.fill(output_str))

def test_gemma_it(device="cuda"):
    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    
    def format_prompt(user_prompt: str) -> str:
        return f"""<start_of_turn>user
        {user_prompt}<end_of_turn>
        <start_of_turn>model
        """

    user_prompt = "What is your name?"
    inputs = tokenizer.encode(format_prompt(user_prompt), return_tensors="pt", add_special_tokens=True).to(device)

    outputs = model.generate(input_ids=inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs.shape[1]:]))

def test_sae_gemma(device="cuda"):
    """
        sae_lens/pretrained_saes.yaml
    """
    REPO_ID = "google/gemma-scope-2-1b-pt"  # HF repo_id
    sae_pos = "resid_post"  # HF folder name
    release = "gemma-scope-2-1b-pt-res"
    LAYER = 22  # options are {7, 13, 17, 22}
    WIDTH = "65k"   # options are {16k, 65k, 262k, 1m}
    L0 = "medium"  # options are {small, medium, big}
    sae_id = f"layer_{LAYER}_width_{WIDTH}_l0_{L0}"

    # maybe downloaded
    path_to_params = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{sae_pos}/{sae_id}/params.safetensors",
    )
    params = safetensors.torch.load_file(path_to_params)
    params = {k.replace("w_enc", "W_enc").replace("w_dec", "W_dec"): v for k, v in params.items()}

    # load sae
    d_model, d_sae = params["W_enc"].shape
    print(f"d_model: {d_model}, d_sae: {d_sae}")
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device
    )
    # print the param keys in sae
    # print(sae.state_dict().keys())  # ['b_dec', 'W_dec', 'W_enc', 'threshold', 'b_enc']
    sae.load_state_dict(params)


if __name__ == "__main__":

    # test_gemma(device="cuda:2")
    # test_gemma_it(device="cuda:2")

    test_sae_gemma(device="cuda:2")