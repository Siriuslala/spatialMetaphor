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
from sae_lens import SAE, HookedSAETransformer  # pip install -U sae-lens


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

def load_sae_gemma(
    repo_id: str = "google/gemma-scope-2-1b-pt",
    sae_pos: str = "resid_post",
    release: str = "gemma-scope-2-1b-pt-res",
    layer: int = 22,
    width: str = "65k",
    L0: Literal["small", "medium", "big"] = "medium",
    device="cuda"
):
    """
    Load SAE. For names, please refer to `sae_lens/pretrained_saes.yaml`.
    Params:
        repo_id: HF repo_id,
        sae_pos: str = "resid_post",
        release: str = "gemma-scope-2-1b-pt-res",
        layer: int = 22,
        width: Literal["16k", "65k", "262k", "1m"] = "65k",
        L0: Literal["small", "medium", "big"] = "medium",
        device="cuda"
    """
    # load sae
    sae_id = f"layer_{layer}_width_{width}_l0_{L0}"
    # maybe downloaded
    path_to_params = hf_hub_download(
        repo_id=repo_id,
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
    # sae.load_state_dict(params)
    
    return sae

def test_sae_gemma_hookedtransformer(model_name, device="cuda", **sae_kwargs):
    # load models
    model = HookedSAETransformer.from_pretrained_no_processing(model_name, device=device)
    sae = load_sae_gemma(device=device, **sae_kwargs)

    # forward
    tokens = model.to_tokens("Hello, world!")
    logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

    # access SAE activations
    # sae_acts = cache["blocks.12.hook_resid_post.hook_sae_acts_post"]
    # print(f"SAE activations shape: {sae_acts.shape}")
    print(cache.keys())

def test_sae_gemma_pytorch(model_name, device="cuda", **sae_kwargs):
    # load models
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sae = load_sae_gemma(device=device, **sae_kwargs)

    # get inputs
    prompt = "The law of conservation of energy states that energy cannot be created or destroyed, only transformed."
    tokens = tokenizer.tokenize(prompt)
    print(tokens)
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    # inputs = tokenizer(prompt, return_tensors="pt")
    
    # prepare hooks
    def hook_fn(module, input, output, cache, key):
        # Gemma transformer blocks output a tuple; hidden states are first
        hidden_states = output[0] if isinstance(output, tuple) else output
        cache[key] = hidden_states.detach()

    cache = {}
    key = "resid_post"
    target_layer = 22
    hook = partial(hook_fn, cache=cache, key=key)
    handle = model.model.layers[target_layer].register_forward_hook(hook)

    # forward
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()

    # Use SAE on extracted activations
    target_act = cache[key]
    sae_acts = sae.encode(target_act.to(torch.float32))
    recon = sae.decode(sae_acts)
    print(f"Cache shape: {target_act.shape}")  # [1, 19, 1152]
    print(f"SAE activations shape: {sae_acts.shape}")  # [1, 19, 65536]
    print(f"Reconstruction shape: {recon.shape}")  # [1, 19, 1152]

    reconstruction_mse = torch.mean((recon[:, 1:] - target_act[:, 1:].float()) ** 2)
    target_variance = target_act[:, 1:].float().var()
    fvu = reconstruction_mse / target_variance
    print(f"Fraction of variance unexplained: {fvu:.2%}")

    l0_per_token = (sae_acts > 1).sum(-1)[0]
    print(l0_per_token.tolist())
    print(f"Average L0: {l0_per_token[1:].float().mean():.2f}")


if __name__ == "__main__":


    # test_gemma(device="cuda:2")
    # test_gemma_it(device="cuda:2")

    # test sae gemma ======================================================================
    model_name = "google/gemma-3-1b-pt"
    sae_kwargs = {
        "repo_id": "google/gemma-scope-2-1b-pt",
        "sae_pos": "resid_post",
        "release": "gemma-scope-2-1b-pt-res",
        "layer": 22,
        "width": "65k",
        "L0": "medium",
    }
    test_sae_gemma_pytorch(model_name, device="cuda:2", **sae_kwargs)
