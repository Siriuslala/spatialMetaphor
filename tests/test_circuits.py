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

from transformer_lens import HookedTransformer

from tools.eap.eap.eap_wrapper import EAP_standard, EAP_IG_standard
from tools.eap.eap.patching_metric import avg_logit_diff
from tools.circuit_data import EAPDataset
from tools.circuit_analysis import compute_circuit_overlap


def get_circuit_dataset(model, tokenizer, theme="emotion"):
    dataset = None
    if theme == "emotion":
        # To find the circuit that is responsible for detecting the emotion in the context
        clean_prompts = ["He just won the game and is feeling happy."]
        corrupt_prompts = ["He just lost the game and is feeling sad."]
        target_clean_tokens = ["happy"]
        target_corrupt_tokens = ["sad"]
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
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)

    # Prepare the dataset
    dataset = get_circuit_dataset(model, model.tokenizer, theme=theme)
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

    top_edges = graph.top_edges(n=num_edges, abs_scores=True)
    top_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    eap_method_name = "eap_ig" if use_ig else "eap"
    fdir = ROOT_DIR / f"figures/circuits/{theme}/{lang}/{model_name}/{eap_method_name}"
    fdir.mkdir(parents=True, exist_ok=True)
    info_path = fdir / "info.jsonl"
    with jsonlines.open(info_path, "w") as f:
        f.write({"edges": top_edges})
    fname = f"{theme}_{model_name}_circuit_edges{num_edges}.pdf"
    graph.show(edges=top_edges, fname=fname, fdir=fdir)

def get_gpt2_circuit_atp(model_name, device="cuda:0", num_nodes=20, theme="emotion", lang="en"):
    # Prepare the model
    pass    


if __name__ == "__main__":

    # Get circuits via EAP ======================================================================
    if False:
        # 7b: >34850M (bsz=1)
        model_name = "qwen2.5-7b"  # "gpt2-small"
        device = "cuda:2"
        num_edges = 1000
        theme = "ioi"  # "emotion", "space"
        lang = "en"
        use_ig = False
        get_gpt2_circuit_eap(model_name=model_name, device=device, num_edges=num_edges, theme=theme, lang=lang, use_ig=use_ig)
    
    # Compute circuits overlap ======================================================================
    if True:
        lang = "en"
        model_name = "qwen2.5-7b" #  "gpt2-small"
        task1_circuit_path = ROOT_DIR / f"figures/circuits/ioi/{lang}/{model_name}/eap/info.jsonl"
        task2_circuit_path = ROOT_DIR / f"figures/circuits/space/{lang}/{model_name}/eap/info.jsonl"
        compute_circuit_overlap(task1_circuit_path, task2_circuit_path)

        # emotion vs. space: Node-level IoU: 0.4291
        # emotion vs. ioi: Node-level IoU: 0.2960
        # space vs. ioi: Node-level IoU: 0.2989
