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

from transformer_lens import HookedTransformer

from tools.eap.eap.eap_wrapper import EAP
from tools.eap.eap.patching_metric import avg_logit_diff
from utils.circuit_data import EAPDataset


def get_circuit_dataset(model, tokenizer, theme="emotion"):
    dataset = None
    if theme == "emotion":
        clean_prompts = ["He just won the game and is feeling happy."]
        corrupt_prompts = ["He just lost the game and is feeling sad."]
        target_clean_tokens = ["happy"]
        target_corrupt_tokens = ["sad"]
    elif theme == "space":
        clean_prompts = []
        corrupt_prompts = []
        target_clean_tokens = ["up"]
        target_corrupt_tokens = ["down"]
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

def get_gpt2_circuit(model_name, device="cuda:0", theme="emotion", lang="en"):
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
    graph = EAP(
        model,
        batch,
        patching_metric_metaphor,
        upstream_nodes=["mlp", "head"],
        downstream_nodes=["mlp", "head"],
    )

    fdir = ROOT_DIR / f"figures/circuits/{theme}/{lang}"
    fdir.mkdir(parents=True, exist_ok=True)
    top_edges = graph.top_edges(n=10, abs_scores=True)
    info_path = fdir / "info.txt"
    with open(info_path, "w") as f:
        for from_edge, to_edge, score in top_edges:
            f.write(f'{from_edge} -> [{round(score, 3)}] -> {to_edge}\n')
    top_edges = graph.top_edges(n=50, abs_scores=True)
    fname = f"{theme}_{model_name}_circuit.pdf"
    graph.show(edges=top_edges, fname=fname, fdir=fdir)


if __name__ == "__main__":

    model_name = "gpt2-small"
    device = "cuda:3"
    theme = "emotion"
    lang = "en"
    get_gpt2_circuit(model_name=model_name, device=device, theme=theme, lang=lang)
