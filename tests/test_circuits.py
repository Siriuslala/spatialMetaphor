"""
Study the spatial metaphor from the circuit view
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
ROOT_DIR = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
DATA_DIR = Path(os.getenv('DATA_DIR'))
WORK_DIR = Path(os.getenv('WORK_DIR'))
sys.path.append(ROOT_DIR.as_posix())

import jsonlines

import torch
from transformer_lens import HookedTransformer

from tools.eap.eap.eap_wrapper import EAP_standard, EAP_IG_standard
from tools.eap.eap.patching_metric import avg_logit_diff
from tools.eap.circuit_data import EAPDataset
from tools.eap.circuit_analysis import compute_circuit_overlap


def get_circuit_dataset(model, tokenizer, theme="emotion", target="happy"):
    """
    emotion: happy/sad
    space: up/down
    """
    dataset = None
    if theme == "emotion":
        # To find the circuit that is responsible for detecting the emotion in the context
        clean_prompts = ["He just won the game and is feeling happy."]
        corrupt_prompts = ["He just lost the game and is feeling sad."]
        target_clean_tokens = ["happy"]
        target_corrupt_tokens = ["sad"]
        if target == "sad":
            clean_prompts, corrupt_prompts = corrupt_prompts, clean_prompts
            target_clean_tokens, target_corrupt_tokens = target_corrupt_tokens, target_clean_tokens
        print(clean_prompts, corrupt_prompts, target_clean_tokens, target_corrupt_tokens)
    elif theme == "space":
        # To find the circuit that is responsible for detecting the spatial direction in the context
        clean_prompts = [
            "The eagle is in the sky, so the child looked up.",
            "To get to the rooftop, the man walked up the stairs.",
            "The movie was excellent, so he gave it a thumbs up."
        ]
        corrupt_prompts = [
            "The mouse is on the floor, so the child looked down.",
            "To get to the basement, the man walked down the stairs.",
            "The movie was terrible, so he gave it a thumbs down."
        ]
        target_clean_tokens = ["up", "up", "up"]
        target_corrupt_tokens = ["down", "down", "down"]
        if target == "down":
            clean_prompts, corrupt_prompts = corrupt_prompts, clean_prompts
            target_clean_tokens, target_corrupt_tokens = target_corrupt_tokens, target_clean_tokens
    elif theme == "ioi":
        # To find the circuit that is responsible for detecting the indirect object identification in the context
        clean_prompts = [
            "John and Mary went to a store, and John gave a bottle of water to Mary."
        ]
        corrupt_prompts = [
            "John and Mary went to a store, and Mary gave a bottle of water to John."
        ]
        target_clean_tokens = ["Mary"]
        target_corrupt_tokens = ["John"]
    else:
        raise ValueError(f"Unknown theme: {theme}")
    dataset = EAPDataset(
        model=model,
        tokenizer=tokenizer,
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        target_clean_tokens=target_clean_tokens,
        target_corrupt_tokens=target_corrupt_tokens,
    )
    return dataset

def get_gpt2_circuit_eap(
    model_name, 
    device="cuda:0", 
    num_edges=50, 
    theme="emotion", 
    target="happy",
    lang="en",
    use_ig=False,
):
    """
    Get the circuit for a task using EAP.
    Params:
        theme: the target task,
        use_ig: whether to use integrated gradients,
    """
    # Prepare the model
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=device,
    )
    if model_name == "qwen2.5-7b":
        model.to(torch.bfloat16)
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)

    # Prepare the dataset
    dataset = get_circuit_dataset(model, model.tokenizer, theme=theme, target=target)
    batch = dataset.get_full_batch()

    # Circuit discovery
    model.reset_hooks()
    patching_metric_metaphor = avg_logit_diff
    eap_method = EAP_IG_standard if use_ig else EAP_standard
    graph = eap_method(
        model,
        batch,
        patching_metric_metaphor,
        upstream_nodes=["mlp", "head"],
        downstream_nodes=["mlp", "head"],
    )

    edges = graph.top_edges(n=num_edges, abs_scores=True)
    # edges = graph.get_all_existing_edges()
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    eap_method_name = "eap_ig" if use_ig else "eap"
    if target is not None and target != "":
        scene = f"{theme}_{target}"
    else:
        scene = theme
    fdir = ROOT_DIR / f"figures/circuits/{scene}/{lang}/{model_name}/{eap_method_name}"
    fdir.mkdir(parents=True, exist_ok=True)
    info_path = fdir / "info.jsonl"
    with jsonlines.open(info_path, "w") as f:
        f.write({"edges": edges})
    # fname = f"{theme}_{model_name}_circuit_edges{num_edges}.pdf"
    # graph.show(edges=edges, fname=fname, fdir=fdir)

def get_gpt2_circuit_atp(model_name, device="cuda:0", num_nodes=20, theme="emotion", lang="en"):
    # Prepare the model
    pass    


if __name__ == "__main__":

    # Get circuits via EAP ======================================================================
    if True:
        # 7b: >30000M (bsz=1)
        model_name = "qwen2.5-7b"  # "gpt2-small"  "qwen2.5-7b"
        device = "cuda:7"
        num_edges = 10000
        theme = "space"  # "emotion", "space"
        target = "down"
        lang = "en"
        use_ig = False
        get_gpt2_circuit_eap(model_name=model_name, device=device, num_edges=num_edges, theme=theme, target=target, lang=lang, use_ig=use_ig)
    
    # Compute circuits overlap ======================================================================
    if False:
        lang = "en"
        model_name = "qwen2.5-7b" #  "gpt2-small"  "qwen2.5-7b"
        scene_0 = ["space", "up"]
        scene_1 = ["emotion", "happy"]
        topn = 1000
        def get_scene_name(scene):
            if scene[1] is not None and scene[1] != "":
                return f"{scene[0]}_{scene[1]}"
            else:
                return scene[0]
        scene_0_name = get_scene_name(scene_0)
        scene_1_name = get_scene_name(scene_1)
        task1_circuit_path = ROOT_DIR / f"figures/circuits/{scene_0_name}/{lang}/{model_name}/eap/info.jsonl"
        task2_circuit_path = ROOT_DIR / f"figures/circuits/{scene_1_name}/{lang}/{model_name}/eap/info.jsonl"
        compute_circuit_overlap(task1_circuit_path, task2_circuit_path, topn=topn)

        # qwen2.5-7b (topn=1000):
        # emotion_happy vs. space_up: IoU^{n}: 0.4291, IoU^{e}: 0.1969
        # emotion vs. ioi: IoU^{n}: 0.2960
        # space vs. ioi: IoU^{n}: 0.2989

        # gpt2-small (topn=1000):
        # emotion_happy vs. space_up: IoU^{n}: 0.5600, IoU^{e}: 0.2151
        # emotion_sad vs. space_down: IoU^{n}: 0.6036, IoU^{e}: 0.2453
        # emotion_happy vs. space_down: IoU^{n}: 0.5928, IoU^{e}: 0.2477
        # emotion_sad vs. space_up: IoU^{n}: 0.5470, IoU^{e}: 0.2005
        # emotion_happy vs. ioi: IoU^{n}: 0.4585, IoU^{e}: 0.1325
        # emotion_sad vs. ioi: IoU^{n}: 0.4462, IoU^{e}: 0.1357
        
