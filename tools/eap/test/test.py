import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_lens import HookedTransformer


from eap.patching_metric import avg_logit_diff
from eap.eap_wrapper import EAP

device = "cuda:2"

model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)


clean_prompts = ["He just won the game and is feeling happy."]
corrupt_prompts = ["He just lost the game and is feeling sad."]

clean_tokens = [model.tokenizer.tokenize(clean_prompt) for clean_prompt in clean_prompts]
corr_tokens = [model.tokenizer.tokenize(corr_prompt) for corr_prompt in corrupt_prompts]
clean_end_positions = [token_list.index("Ġhappy") - 1 for token_list in clean_tokens]
corr_end_positions = [token_list.index("Ġsad") - 1 for token_list in corr_tokens]
clean_token_ids = [model.tokenizer.convert_tokens_to_ids("Ġhappy") for _ in clean_tokens]
corr_token_ids = [model.tokenizer.convert_tokens_to_ids("Ġsad") for _ in corr_tokens]
print(clean_tokens)
print(corr_tokens)         
print(clean_end_positions)
print(corr_end_positions)
print(clean_token_ids)
print(corr_token_ids)

clean_inputs = model.tokenizer.batch_encode_plus(clean_prompts, padding=True, return_tensors='pt')
corr_inputs = model.tokenizer.batch_encode_plus(corrupt_prompts, padding=True, return_tensors='pt')      

batch = {
    "clean_inputs": {
        "input_ids": clean_inputs["input_ids"].to(device),
        "attention_mask": clean_inputs["attention_mask"].to(device),
        "end_positions": clean_end_positions,
        "clean_token_ids": clean_token_ids,
    },
    "corrupted_inputs": {
        "input_ids": corr_inputs["input_ids"].to(device),
        "attention_mask": corr_inputs["attention_mask"].to(device),
        "end_positions": corr_end_positions,
        "corrupted_token_ids": corr_token_ids,
    },
}
print(batch)

model.reset_hooks()
patching_metric_metaphor = avg_logit_diff
graph = EAP(
    model,
    batch,
    patching_metric_metaphor,
    upstream_nodes=["mlp", "head"],
    downstream_nodes=["mlp", "head"],
)

top_edges = graph.top_edges(n=10, abs_scores=True)
for from_edge, to_edge, score in top_edges:
    print(f'{from_edge} -> [{round(score, 3)}] -> {to_edge}')
top_edges = graph.top_edges(n=50, abs_scores=True)
graph.show(edges=top_edges)

