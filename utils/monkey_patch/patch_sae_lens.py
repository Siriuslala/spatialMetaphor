import json
import re
import os
import struct
from pathlib import Path
import requests
from typing import Any, Protocol

from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers


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
            local_files_only=local_files_only
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

