import json
import re
import os
import struct
from pathlib import Path
import requests
from typing import Any, Protocol

import torch
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers, EntryNotFoundError

from sae_lens.loading.pretrained_sae_loaders import load_safetensors_weights, _infer_gemma_3_raw_cfg_dict


def patched_get_safetensors_tensor_shapes(repo_id: str, filename: str, local_files_only: bool = True) -> dict[str, list[int]]:
    """
    Get tensor shapes from a safetensors file on HuggingFace Hub
    without downloading the entire file.

    Args:
        repo_id: HuggingFace repo ID (e.g., "gg-gs/gemma-scope-2-1b-pt")
        filename: Path to the safetensors file within the repo

    Returns:
        Dictionary mapping tensor names to their shapes
    """
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=local_files_only  # patch here
        )

        with open(local_path, "rb") as f:
            # 读取前 8 个字节获取 header 长度
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                raise ValueError(f"Invalid safetensors file: {local_path}")
            
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            
            header_json_bytes = f.read(header_size)
            header = json.loads(header_json_bytes.decode("utf-8"))
        
            return {
                name: info["shape"] 
                for name, info in header.items() 
                if name != "__metadata__"
            }
    except Exception as e:
        if not local_files_only:
            print(f"Local file not found, falling back to network: {e}")
            url = hf_hub_url(repo_id, filename)

            # Get HuggingFace headers (includes auth token if available)
            hf_headers = build_hf_headers()

            # Fetch first 8 bytes to get metadata size
            headers = {**hf_headers, "Range": "bytes=0-7"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            meta_size = int.from_bytes(response.content, byteorder="little")

            # Fetch the metadata header
            headers = {**hf_headers, "Range": f"bytes=8-{8 + meta_size - 1}"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            metadata_json = response.content.decode("utf-8").strip()
            metadata = json.loads(metadata_json)

            # Extract tensor shapes, excluding the __metadata__ key
            return {
                name: info["shape"] for name, info in metadata.items() if name != "__metadata__"
            }
        else:
            raise FileNotFoundError(f"Could not find {filename} in local cache for {repo_id}. "
                                    "Set local_files_only=False to allow downloading.")

def patched_get_gemma_3_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Try to load config.json from the repo, fall back to inferring if it doesn't exist
    try:
        print(f"Downloading config.json from {repo_id}/{folder_name}")
        config_path = hf_hub_download(
            repo_id, f"{folder_name}/config.json", force_download=force_download
        )       
        with open(config_path) as config_file:
            raw_cfg_dict = json.load(config_file)
    except EntryNotFoundError:
        raw_cfg_dict = _infer_gemma_3_raw_cfg_dict(repo_id, folder_name)

    if raw_cfg_dict.get("architecture") != "jump_relu":
        raise ValueError(
            f"Unexpected architecture in Gemma 3 config: {raw_cfg_dict.get('architecture')}"
        )

    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match is None:
        raise ValueError(
            f"Could not extract layer number from folder_name: {folder_name}"
        )
    layer = int(layer_match.group(1))
    hook_name_out = None
    d_out = None
    if "resid_post" in folder_name:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif "attn_out" in folder_name:
        hook_name = f"blocks.{layer}.hook_attn_out"
    elif "mlp_out" in folder_name:
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif "transcoder" in folder_name or "clt" in folder_name:
        hook_name = f"blocks.{layer}.ln2.hook_normalized"
        hook_name_out = f"blocks.{layer}.hook_mlp_out"
    else:
        raise ValueError("Hook name not found in folder_name.")

    # hackily deal with clt file names
    params_file_part = "/params.safetensors"
    if "clt" in folder_name:
        params_file_part = ".safetensors"

    shapes_dict = patched_get_safetensors_tensor_shapes(  # patch here
        repo_id, f"{folder_name}{params_file_part}"
    )
    d_in, d_sae = shapes_dict["w_enc"]
    # TODO: update this for real model info
    model_name = raw_cfg_dict["model_name"]
    if "google" not in model_name:
        model_name = "google/" + model_name
    model_name = model_name.replace("-v3", "-3")
    if "270m" in model_name:
        # for some reason the 270m model on huggingface doesn't have the -pt suffix
        model_name = model_name.replace("-pt", "")

    architecture = "jumprelu"
    if "transcoder" in folder_name or "clt" in folder_name:
        architecture = "jumprelu_skip_transcoder"
        d_out = shapes_dict["w_dec"][-1]

    cfg = {
        "architecture": architecture,
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": model_name,
        "hook_name": hook_name,
        "hook_head_index": None,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
        "hf_hook_name": raw_cfg_dict.get("hf_hook_point_in"),
    }
    if hook_name_out is not None:
        cfg["hook_name_out"] = hook_name_out
        cfg["hf_hook_name_out"] = raw_cfg_dict.get("hf_hook_point_out")
    if d_out is not None:
        cfg["d_out"] = d_out
    if device is not None:
        cfg["device"] = device

    if cfg_overrides is not None:
        cfg.update(cfg_overrides)

    return cfg 

def patched_gemma_3_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Custom loader for Gemma 3 SAEs.
    """
    cfg_dict = patched_get_gemma_3_config_from_hf(  # patch here
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    params_file = "params.safetensors"
    if "clt" in folder_name:
        params_file = folder_name.split("/")[-1] + ".safetensors"
        folder_name = "/".join(folder_name.split("/")[:-1])

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=params_file,
        subfolder=folder_name,
        force_download=force_download,
    )

    raw_state_dict = load_safetensors_weights(
        sae_path, device=device, dtype=cfg_dict.get("dtype")
    )

    with torch.no_grad():
        w_dec = raw_state_dict["w_dec"]
        if "clt" in folder_name:
            w_dec = w_dec.sum(dim=1).contiguous()

    state_dict = {
        "W_enc": raw_state_dict["w_enc"],
        "W_dec": w_dec,
        "b_enc": raw_state_dict["b_enc"],
        "b_dec": raw_state_dict["b_dec"],
        "threshold": raw_state_dict["threshold"],
    }

    if "affine_skip_connection" in raw_state_dict:
        state_dict["W_skip"] = raw_state_dict["affine_skip_connection"].T

    return cfg_dict, state_dict, None

