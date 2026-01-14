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
from sae_lens import SAE, HookedSAETransformer, SAEConfig, SkipTranscoder  # pip install -U sae-lens
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')
ROOT_DIR = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent.parent))
DATA_DIR = Path(os.getenv('DATA_DIR'))
WORK_DIR = Path(os.getenv('WORK_DIR'))
sys.path.append(ROOT_DIR.as_posix())

from utils.model_heplers import format_prompt_gemma
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

    user_prompt = "What is your name?"
    inputs = tokenizer.encode(format_prompt_gemma(user_prompt), return_tensors="pt", add_special_tokens=True).to(device)

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
    # load sae params
    sae_id = f"layer_{layer}_width_{width}_l0_{L0}"
    # maybe downloaded
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=f"{sae_pos}/{sae_id}/params.safetensors",
    )
    params = safetensors.torch.load_file(path_to_params)
    params = {k.replace("w_enc", "W_enc").replace("w_dec", "W_dec"): v for k, v in params.items()}

    d_model, d_sae = params["W_enc"].shape
    print(f"d_model: {d_model}, d_sae: {d_sae}")

    # load sae
    # cfg = SAEConfig(
    #     d_in=d_model,
    #     d_sae=d_sae,
    #     activation_fn="relu",
    #     device=device
    # )
    # sae = SAE(cfg)
    # sae.load_state_dict(params)

    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    # print the param keys in sae
    # print(sae.state_dict().keys())  # ['b_dec', 'W_dec', 'W_enc', 'threshold', 'b_enc']
    # sae.load_state_dict(params)
    
    return sae

def load_transcoder_gemma(
    repo_id: str = "google/gemma-scope-2-1b-pt",
    transcoder_pos: str = "transcoder",
    release: str = "gemma-scope-2-1b-pt-transcoders",
    layer: int = 17,
    width: str = "65k",
    L0: Literal["small", "medium", "big"] = "medium",
    device="cuda"
):
    """
    Load transcoder. For names, please refer to `sae_lens/pretrained_saes.yaml`.
    For model archs, please refer to:
        https://decoderesearch.github.io/SAELens/latest/api/;
        https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf.
    """
    # load sae params
    transcoder_id = f"layer_{layer}_width_{width}_l0_{L0}_affine"
    # maybe downloaded
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=f"{transcoder_pos}/{transcoder_id}/params.safetensors",
    )
    params = safetensors.torch.load_file(path_to_params)
    params = {k.replace("w_enc", "W_enc").replace("w_dec", "W_dec").replace('affine_skip_connection', 'W_skip'): v for k, v in params.items()}
    # breakpoint()

    d_model, d_sae = params["W_enc"].shape
    print(f"d_model: {d_model}, d_sae: {d_sae}")

    transcoder = SkipTranscoder.from_pretrained(
        release=release,
        sae_id=f"{transcoder_pos}/{transcoder_id}",
        device=device,
    )
    # print(transcoder)
    # print(transcoder.state_dict().keys())  # ['b_dec', 'W_dec', 'W_enc', 'b_enc', 'threshold', 'W_skip']
    # print(transcoder.state_dict()["W_skip"])
    # breakpoint()

    return transcoder

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
    tokens = tokenizer.tokenize(prompt, add_special_tokens=True)
    print(tokens)
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    # inputs = tokenizer(prompt, return_tensors="pt")
    
    # forward and cache the residual activations
    target_layer = sae_kwargs["layer"]
    cahced_act = gather_residual_activations(model, target_layer, inputs)

    # Use SAE on extracted activations
    ## find sae features
    target_act = cahced_act
    sae_acts = sae.encode(target_act.to(torch.float32))
    recon = sae.decode(sae_acts)
    print(f"Cache shape: {target_act.shape}")  # [1, 19, 1152]
    print(f"SAE activations shape: {sae_acts.shape}")  # [1, 19, 65536]
    print(f"Reconstruction shape: {recon.shape}")  # [1, 19, 1152]

    ## find recon loss
    reconstruction_mse = torch.mean((recon[:, 1:] - target_act[:, 1:].float()) ** 2)
    target_variance = target_act[:, 1:].float().var()
    fvu = reconstruction_mse / target_variance
    print(f"Fraction of variance unexplained: {fvu:.2%}")

    l0_per_token = (sae_acts > 1).sum(-1)[0]
    print(l0_per_token.tolist())
    print(f"Average L0: {l0_per_token[1:].float().mean():.2f}")

    ## find top features
    top_activations, top_features = sae_acts.max(-1)
    print(top_features)

    top_acts, top_latents = sae_acts.squeeze().mean(0).topk(5)  # [1, 19, 65536]
    for act, idx in zip(top_acts, top_latents):
        print(f"{act:>6.1f} | {idx}")
    
    ## study single feature
    feature_idx = 6524
    activations = sae_acts[0, :, feature_idx].tolist()

    fig_dir = ROOT_DIR / "figures"
    save_dir = fig_dir / "sae_gemma"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name.split("/")[-1]
    output_path = save_dir / f"{model_name}-sae_feat_{feature_idx}-prompt_physics.pdf"
    generate_token_activation_map(tokens, activations, output_path)

def test_sae_gemma_pytorch_explore_one_feature(
    model_name, 
    device="cuda",
    target_layer=22,
    feature_idx=6524,
    **sae_kwargs, 
):
    # load models
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sae = load_sae_gemma(device=device, **sae_kwargs)

    # test on prompts
    if False:
        all_results = []
        for prompt in [
            "Gemma Scope 2 is a model release from Google DeepMind",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            "Gravity describes how massive objects attract one another",
            "A charge accelerating through an electric field experiences a force",
            "Chemical fuel stores energy in molecular bonds, which is released"
        ]:
            inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            _target_acts = gather_residual_activations(model, target_layer, inputs)

            _sae_acts = sae.encode(_target_acts.to(torch.float32))

            tokens = tokenizer.tokenize(prompt, add_special_tokens=True)
            act_values = _sae_acts[0, :, feature_idx].tolist()
            all_results.append({
                "tokens": tokens,
                "activations": act_values
            })
        fig_dir = ROOT_DIR / "figures"
        save_dir = fig_dir / "sae_gemma"
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_name.split("/")[-1]
        output_path = save_dir / f"{model_name}-sae_feat_{feature_idx}-prompts.pdf"
        generate_multi_token_activation_maps(all_results, output_path)

    if True:
        w_u = model.lm_head.weight  # shape (d_vocab, d_model)
        w_u_eff = w_u * model.model.norm.weight  # model.model.norm is applied to the output of the last decoder layer

        decoder_vector = sae.W_dec[feature_idx]  # shape (d_model,)

        top_activations, top_tokens = torch.topk(w_u_eff @ decoder_vector, k=10)

        for act, tok in zip(top_activations, top_tokens):
            print(f"{act:.4f} | {tokenizer.decode(tok)}")

def test_sae_gemma_pytorch_intervene_with_sae_features(
    model_name, 
    device="cuda",
    target_layer=22,
    feature_idx=6524,
    **sae_kwargs, 
):
    # load models
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sae = load_sae_gemma(device=device, **sae_kwargs)

    def generate_with_steering(model, sae, inputs, target_layer, feature_idx: int, coeff: float):

        def steering_hook(mod, inputs, outputs):
            output = outputs[0]
            # We have to be careful about KV caching! This logic handles different cases depending on
            # whether this is the first forward pass or a cached pass.
            if output.shape[1] == 1:
                avg_norm = torch.norm(output, dim=-1)
                output += coeff * avg_norm * sae.W_dec[feature_idx]
            else:
                # avg_norm = torch.norm(output[0, 1:], dim=-1, keepdim=True)
                # output[0, 1:] += coeff * avg_norm * sae.w_dec[feature_idx]
                avg_norm = torch.norm(output[0, -1:], dim=-1, keepdim=True)
                output[0, -1:] += coeff * avg_norm * sae.W_dec[feature_idx]

            return outputs

        handle = model.model.layers[target_layer].register_forward_hook(steering_hook)
        try:
            outputs = model.generate(input_ids=inputs, max_new_tokens=80, do_sample=False)
            output_str = tokenizer.decode(outputs[0])
        finally:
            handle.remove()

        return output_str.split("<start_of_turn>model")[1].strip()


    user_prompt = "Tell me a fun fact."
    inputs = tokenizer.encode(format_prompt_gemma(user_prompt), return_tensors="pt", add_special_tokens=True).to(device)

    print(user_prompt)
    print("======================= NO STEERING =======================")
    output_str = generate_with_steering(
        model=model,
        sae=sae,
        inputs=inputs,
        target_layer=target_layer - 8,
        feature_idx=feature_idx,
        coeff=0.0,
    )
    print(textwrap.fill(output_str))
    print("======================= STEERING =======================")
    output_str_steered = generate_with_steering(
        model=model,
        sae=sae,
        inputs=inputs,
        target_layer=target_layer - 8,
        feature_idx=feature_idx,
        coeff=0.14,
    )
    print(textwrap.fill(output_str_steered))
    
def test_sae_gemma_pytorch_intervene_with_recon(model_name, device="cuda", **sae_kwargs):
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
    def fwd_pass_with_sae_intervention(model, sae, target_layer, inputs):
        # Forward pass to get logits & hidden activations
        model_output_clean = model.forward(inputs, output_hidden_states=True)
        logits_clean = model_output_clean.logits[0]  # (len, vocab_size)
        hidden_states = model_output_clean.hidden_states[target_layer + 1][0]  # (len, d_model), [0]: get first sample

        # Get the SAE reconstruction
        recon = sae(hidden_states.to(torch.float32))  # (len, d_model)

        def intervene_on_target_act_hook(mod, inputs, outputs):
            # outputs[0]: hidden_states (bsz, len, d_model)
            # [0, 1:]: first sample, all tokens except the first one
            outputs[0][0, 1:] = recon[1:]
            return outputs

        handle = model.model.layers[target_layer].register_forward_hook(intervene_on_target_act_hook)
        try:
            model_output = model.forward(inputs)
        finally:
            handle.remove()

        # Get logits from this corrupted forward pass
        logits = model_output.logits[0]

        return logits_clean, logits
    
    def cross_entropy_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Measures avg cross entropy loss."""
        logprobs = logits[:-1].log_softmax(dim=-1)
        tokens = tokens[1:]
        correct_logprobs = logprobs[torch.arange(len(tokens)), tokens]
        return -correct_logprobs

    target_layer = 22
    logits_clean, logits_sae = fwd_pass_with_sae_intervention(model, sae, target_layer, inputs)
    loss_clean = cross_entropy_loss(logits_clean, inputs[0])
    loss_sae = cross_entropy_loss(logits_sae, inputs[0])

    print(f"Loss (clean): {loss_clean.mean():.4f}")
    print(f"Loss (corrupted): {loss_sae.mean():.4f}")
    print(f"Delta loss: {loss_sae.mean() - loss_clean.mean():.4f}")

def test_sae_transcoder_pytorch(model_name, device="cuda", **transcoder_kwargs):
    # load models
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transcoder = load_transcoder_gemma(device=device, **transcoder_kwargs)

    prompt = "The quick brown fox jumped over the lazy dog"
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    target_layer = transcoder_kwargs["layer"]
    cache = gather_transcoder_activations(model, target_layer, inputs)

    sae_input = cache["transcoder_input"].to(torch.float32).to(device)
    sae_target = cache["transcoder_target"].to(torch.float32).to(device)

    sae_acts = transcoder.encode(sae_input)
    recon = transcoder(sae_input)

    mse = torch.mean((recon[1:] - sae_target[1:].float())**2)
    var = sae_target[1:].float().var()
    fvu = mse / var
    l0 = (sae_acts[1:] > 0).float().sum(-1).mean()

    print(f"L0: {l0:.2f}")
    print(f"Fraction of variance unexplained: {mse / var:.2%}")
    

if __name__ == "__main__":

    # test gemma ======================================================================
    # test_gemma(device="cuda:2")
    # test_gemma_it(device="cuda:2")

    # test gemma sae ======================================================================
    if False:
        model_name = "google/gemma-3-1b-it"
        sae_kwargs = {
            "repo_id": "google/gemma-scope-2-1b-pt",  # "google/gemma-scope-2-1b-pt", "google/gemma-scope-2-1b-it"
            "sae_pos": "resid_post",
            "release": "gemma-scope-2-1b-pt-res",
            "layer": 22,
            "width": "65k",
            "L0": "medium",
        }
        # test_sae_gemma_pytorch(model_name, device="cuda:1", **sae_kwargs)
        # test_sae_gemma_pytorch_explore_one_feature(model_name, device="cuda:5", **sae_kwargs)
        # test_sae_gemma_pytorch_intervene_with_recon(model_name, device="cuda:1", **sae_kwargs)
        test_sae_gemma_pytorch_intervene_with_sae_features(model_name, device="cuda:5", **sae_kwargs)
    
    # test gemma transcoder ======================================================================
    if True:
        model_name = "google/gemma-3-1b-pt"
        transcoder_kwargs = {
            "repo_id": "google/gemma-scope-2-1b-pt",  # "google/gemma-scope-2-1b-pt", "google/gemma-scope-2-1b-it"
            "transcoder_pos": "transcoder",
            "release": "gemma-scope-2-1b-pt-transcoders",
            "layer": 17,
            "width": "65k",
            "L0": "medium",
        }
        test_sae_transcoder_pytorch(model_name, device="cuda:7", **transcoder_kwargs)

