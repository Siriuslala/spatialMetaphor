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
from utils.model_heplers import format_prompt_gemma
from tools.hook import gather_residual_activations
from utils.plot_helpers import (
    generate_token_activation_map,
    generate_multi_token_activation_maps
)

# monkey patch
from utils.monkey_patch.patch_sae_lens import patched_get_safetensors_tensor_shapes
sae_lens.loading.pretrained_sae_loaders.get_safetensors_tensor_shapes = patched_get_safetensors_tensor_shapes

