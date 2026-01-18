import gc
from functools import partial
from typing import Callable, List, Union, Dict

import einops
import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from .eap_graph import EAPGraph

def EAP_corrupted_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph,
    end_positions: Int[Tensor, "batch_size"],
    ie_over_seq: bool = False,
):
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        # We are in the case of a residual layer or MLP
        # Activations have shape [batch_size, seq_len, d_model]
        # We need to add an extra dimension to make it [batch_size, seq_len, 1, d_model]
        # The hook slice is a slice of length 1
        if ie_over_seq:
            upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
        else:
            # get the activations at the end positions
            upstream_activations_difference[:, hook_slice, :] = -activations[:, end_positions, :].squeeze(1).unsqueeze(-2)
    elif activations.ndim == 4:
        # We are in the case of an attention layer
        # Activations have shape [batch_size, seq_len, n_heads, d_model]
        if ie_over_seq:
            upstream_activations_difference[:, :, hook_slice, :] = -activations
        else:
            # get the activations at the end positions
            upstream_activations_difference[:, hook_slice, :] = -activations[:, end_positions, :, :].squeeze(1)

def EAP_clean_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph,
    end_positions: Int[Tensor, "batch_size"],
    ie_over_seq: bool = False,
):
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        if ie_over_seq:
            upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
        else:
            # get the activations at the end positions
            upstream_activations_difference[:, hook_slice, :] += activations[:, end_positions, :].squeeze(1).unsqueeze(-2)
    elif activations.ndim == 4:
        # print(f"end_positions: {end_positions}")
        # print(f"activations.shape: {activations.shape}")
        # print(f"activations[:, end_positions, :, :].shape: {activations[:, end_positions, :, :].shape}")
        # breakpoint()
        if ie_over_seq:
            upstream_activations_difference[:, :, hook_slice, :] += activations
        else:
            # get the activations at the end positions
            upstream_activations_difference[:, hook_slice, :] += activations[:, end_positions, :, :].squeeze(1)
    # print(f"upstream_activations_difference.shape: {upstream_activations_difference.shape}")
    # print(f"activations.shape: {activations.shape}")
    # breakpoint()

def EAP_clean_backward_hook(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph,
    end_positions: Int[Tensor, "batch_size"],
    ie_over_seq: bool = False,
):
    hook_slice = graph.get_hook_slice(hook.name)

    # we get the slice of all upstream nodes that come before this downstream node
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
    # we want to multiply it by the upstream activations difference
    if grad.ndim == 3:
        grad_expanded = grad.unsqueeze(-2)  # Shape: [batch_size, seq_len, 1, d_model]
    else:
        grad_expanded = grad  # Shape: [batch_size, seq_len, n_heads, d_model]

    # we compute the mean over the batch_size and seq_len dimensions
    if ie_over_seq:
        result = torch.matmul(
            upstream_activations_difference[:, :, earlier_upstream_nodes_slice],
            grad_expanded.transpose(-1, -2)
        ).sum(dim=0).sum(dim=0)  # we sum over the batch_size and seq_len dimensions
    else:
        grad_expanded = grad_expanded[:, end_positions, :, :].squeeze(1)
        result = torch.matmul(
            upstream_activations_difference[:, earlier_upstream_nodes_slice],
            grad_expanded.transpose(-1, -2)
        ).sum(dim=0)  # we sum over the batch_size dimension

    try:
        graph.eap_scores[earlier_upstream_nodes_slice, hook_slice] += result
    except:
        print(hook.name)
        print(f"grad.shape: {grad.shape}")
        print(f"hook_slice: {hook_slice}")
        print(f"result.shape: {result.shape}")
        # blocks.27.hook_v_input
        # grad.shape: torch.Size([1, 10, 4, 3584])
        # hook_slice: slice(2351, 2379, None)
        # result.shape: torch.Size([783, 4])
        breakpoint()

def EAP_downstream_patching_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph,
) -> Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]]:
    hook_slice = graph.downstream_hook_slice[hook.name]

    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # The tensor 'patch_difference' represents the sum of all upstream activation differences that are connected to this downstream node
    patch_difference = einops.einsum(
        graph.adj_matrix[earlier_upstream_nodes_slice, hook_slice],
        upstream_activations_difference[:, :, earlier_upstream_nodes_slice, :],
        "n_upstream n_downstream_at_hook, batch_size seq_len n_upstream d_model -> batch_size seq_len n_downstream_at_hook d_model"
    )

    # alternatively, it might be faster to
    # 1) gather the activation differences for every non-zero element in adj_matrix[:, hook_slice] and sum them
    # 2) use torch.sparse.mm to multiply the adj_matrix with the activation differences

    if activations.ndim == 3:
        assert patch_difference.shape[-2] == 1, "Number of downstream nodes should be 1 for this type of hook" 
        # patched_residual_stream = clean_residual_stream - (clean_upstream_activations - corrupted_upstream_activations)
        activations -= patch_difference.squeeze(-2)
    elif activations.ndim == 4:
        activations -= patch_difference
    
    return activations

def simple_activations_hook(act, hook, cache):
    cache[hook.name] = act.detach()

def EAP_IG_interpolation_forward_hook(
    activations: Tensor,
    hook: HookPoint,
    alpha: float,
    # cache_clean: Dict[str, Tensor],
    cache_corrupted: Dict[str, Tensor],
    end_positions: Int[Tensor, "batch_size"],
    ie_over_seq: bool = False,
):
    """
    IG 核心 Hook：在前向传播时将激活值替换为路径上的插值点
    A(alpha) = A_corrupted + alpha * (A_clean - A_corrupted)
    """
    # clean_act = cache_clean[hook.name]
    # corr_act = cache_corrupted[hook.name]
    # return corr_act + alpha * (clean_act - corr_act)  # this could destroy the original gradient flow

    corr_act = cache_corrupted[hook.name]
    if ie_over_seq:
        interpolated_act = corr_act + alpha * (activations - corr_act)
    else:
        interpolated_act = activations
        interpolated_act[:, end_positions, ...] = corr_act[:, end_positions, ...] + alpha * (interpolated_act[:, end_positions, ...] - corr_act[:, end_positions, ...])
    interpolated_act.requires_grad_(True)
    return interpolated_act

def EAP_standard(
    model: HookedTransformer,
    batch: dict,
    metric: Callable,
    upstream_nodes: List[str]=None,
    downstream_nodes: List[str]=None,
    calc_batch_size = None,
    ie_over_seq: bool = False,
):
    """
    "standard" means we create a corrupted prompt for the input.
    params:
        ie_over_seq: whether to patch on all tokens positions in a sequence
    """
    # process data
    device = model.cfg.device
    clean_input_ids = batch["clean_inputs"]["input_ids"].to(device)
    clean_attention_mask = batch["clean_inputs"]["attention_mask"].to(device)
    corrupted_input_ids = batch["corrupted_inputs"]["input_ids"].to(device)
    corrupted_attention_mask = batch["corrupted_inputs"]["attention_mask"].to(device)
    batch_size, seq_len = clean_input_ids.shape[0], clean_input_ids.shape[1]

    end_positions = batch["clean_inputs"]["end_positions"]
    clean_token_ids = batch["clean_inputs"]["clean_token_ids"]
    corrupted_token_ids = batch["corrupted_inputs"]["corrupted_token_ids"]

    # set up EAP graph
    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
    graph.reset_scores()

    upstream_hook_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_hook_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))

    # start
    if calc_batch_size is None:
        calc_batch_size = batch_size
    num_prompts = batch_size

    for idx in tqdm(range(0, num_prompts, calc_batch_size)):

        # prepare upstream activations difference
        if ie_over_seq:
            upstream_activations_difference = torch.zeros(
                (calc_batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
                device=model.cfg.device,
                dtype=model.cfg.dtype,
                requires_grad=False
            )
        else:
            upstream_activations_difference = torch.zeros(
                (calc_batch_size, graph.n_upstream_nodes, model.cfg.d_model),
                device=model.cfg.device,
                dtype=model.cfg.dtype,
                requires_grad=False
            )
        
        # define hooks
        corruped_upstream_hook_fn = partial(
            EAP_corrupted_forward_hook,
            upstream_activations_difference=upstream_activations_difference,
            graph=graph,
            end_positions=end_positions[idx:idx+calc_batch_size],
            ie_over_seq=ie_over_seq
        )

        clean_upstream_hook_fn = partial(
            EAP_clean_forward_hook,
            upstream_activations_difference=upstream_activations_difference,
            graph=graph,
            end_positions=end_positions[idx:idx+calc_batch_size],
            ie_over_seq=ie_over_seq
        )

        clean_downstream_hook_fn = partial(
            EAP_clean_backward_hook,
            upstream_activations_difference=upstream_activations_difference,
            graph=graph,
            end_positions=end_positions[idx:idx+calc_batch_size],
            ie_over_seq=ie_over_seq
        )

        # --- Step 1: Corrupted Forward ---
        # we first perform a forward pass on the corrupted input 
        model.add_hook(upstream_hook_filter, corruped_upstream_hook_fn, "fwd")

        # we don't need gradients for this forward pass
        # we'll take the gradients when we perform the forward pass on the clean input
        with torch.no_grad():
            model(
                corrupted_input_ids[idx:idx+calc_batch_size], 
                attention_mask=corrupted_attention_mask[idx:idx+calc_batch_size], 
                return_type=None
            )
        model.reset_hooks()

        # --- Step 2: Clean Forward ---
        # now we perform a forward and backward pass on the clean input
        model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
        model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")

        clean_logits = model(
            clean_input_ids[idx:idx+calc_batch_size], 
            attention_mask=clean_attention_mask[idx:idx+calc_batch_size], 
            return_type="logits"
        )

        # --- Step 3: Clean Backward ---
        value = metric(
            clean_logits, 
            end_positions[idx:idx+calc_batch_size], 
            clean_token_ids[idx:idx+calc_batch_size], 
            corrupted_token_ids[idx:idx+calc_batch_size]
        )
        value.backward()
        
        # We delete the activation differences tensor to free up memory
        upstream_activations_difference *= 0
        model.zero_grad()
        model.reset_hooks()

    del upstream_activations_difference
    gc.collect()
    torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= num_prompts
    graph.eap_scores = graph.eap_scores.cpu()

    return graph

def EAP_IG_standard(
    model: HookedTransformer,
    batch: dict,
    metric: Callable,
    upstream_nodes: List[str]=None,
    downstream_nodes: List[str]=None,
    calc_batch_size: int = None,
    ie_over_seq: bool = False,
    ig_steps: int = 5
):
    """
    "standard" means we create a corrupted prompt for the input.
    """
    # process data
    device = model.cfg.device
    clean_input_ids = batch["clean_inputs"]["input_ids"].to(device)
    clean_attention_mask = batch["clean_inputs"]["attention_mask"].to(device)
    corrupted_input_ids = batch["corrupted_inputs"]["input_ids"].to(device)
    corrupted_attention_mask = batch["corrupted_inputs"]["attention_mask"].to(device)
    batch_size, seq_len = clean_input_ids.shape[0], clean_input_ids.shape[1]

    end_positions = batch["clean_inputs"]["end_positions"]
    clean_token_ids = batch["clean_inputs"]["clean_token_ids"]
    corrupted_token_ids = batch["corrupted_inputs"]["corrupted_token_ids"]

    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)
    graph.reset_scores()

    upstream_hook_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_hook_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))

    # start
    if calc_batch_size is None:
        calc_batch_size = batch_size
    num_prompts = batch_size

    for idx in tqdm(range(0, num_prompts, calc_batch_size)):

        # prepare upstream activations difference
        if ie_over_seq:
            upstream_activations_difference = torch.zeros(
                (calc_batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
                device=model.cfg.device,
                dtype=model.cfg.dtype,
                requires_grad=False
            )
        else:
            upstream_activations_difference = torch.zeros(
                (calc_batch_size, graph.n_upstream_nodes, model.cfg.d_model),
                device=model.cfg.device,
                dtype=model.cfg.dtype,
                requires_grad=False
            )

        # --- Step 1: Precompute Clean and Corrupted ---
        # we first perform a forward pass on the corrupted input to get the corrupted activations
        cache_corrupted = {}
        corrupted_activations_hook_fn = partial(
            simple_activations_hook,
            cache=cache_corrupted
        )
        corruped_upstream_hook_fn = partial(
            EAP_corrupted_forward_hook,
            upstream_activations_difference=upstream_activations_difference,
            graph=graph,
            end_positions=end_positions[idx:idx+calc_batch_size],
            ie_over_seq=ie_over_seq
        )
        model.add_hook(upstream_hook_filter, corrupted_activations_hook_fn, "fwd")
        model.add_hook(upstream_hook_filter, corruped_upstream_hook_fn, "fwd")

        with torch.no_grad():
            model(
                corrupted_input_ids[idx:idx+calc_batch_size], 
                attention_mask=corrupted_attention_mask[idx:idx+calc_batch_size], 
                return_type=None
            )
        model.reset_hooks()

        # then we perform a forward pass on the clean input to get the clean activations
        clean_upstream_hook_fn = partial(
            EAP_clean_forward_hook,
            upstream_activations_difference=upstream_activations_difference,
            graph=graph,
            end_positions=end_positions[idx:idx+calc_batch_size],
            ie_over_seq=ie_over_seq,
        )
        model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
        
        with torch.no_grad():
            model(
                clean_input_ids[idx:idx+calc_batch_size], 
                attention_mask=clean_attention_mask[idx:idx+calc_batch_size], 
                return_type="logits"
            )
        model.reset_hooks()

        # --- Step 2: IG loop (sample gradients) ---
        # score = A_diff * Grad_sum = sum(A_diff * sub_grad)
        for step in range(1, ig_steps + 1):
            
            alpha = step / ig_steps

            interpolation_hook_fn = partial(
                EAP_IG_interpolation_forward_hook,
                alpha=alpha, 
                cache_corrupted=cache_corrupted,
                end_positions=end_positions[idx:idx+calc_batch_size],
                ie_over_seq=ie_over_seq,
            )
            clean_backward_hook_fn = partial(
                EAP_clean_backward_hook,
                upstream_activations_difference=upstream_activations_difference,
                graph=graph,
                end_positions=end_positions[idx:idx+calc_batch_size],
                ie_over_seq=ie_over_seq,
            )

            # interpolate between corrupted and clean activations
            model.add_hook(upstream_hook_filter, interpolation_hook_fn, "fwd")
            model.add_hook(downstream_hook_filter, clean_backward_hook_fn, "bwd")

            logits = model(
                input=clean_input_ids[idx:idx+calc_batch_size], 
                attention_mask=clean_attention_mask[idx:idx+calc_batch_size], 
                return_type="logits"
            )
            value = metric(
                logits=logits, 
                end_positions=end_positions[idx:idx+calc_batch_size], 
                clean_token_ids=clean_token_ids[idx:idx+calc_batch_size], 
                corrupted_token_ids=corrupted_token_ids[idx:idx+calc_batch_size]
            )
            value.backward(retain_graph=(True))
            
            model.zero_grad()
            model.reset_hooks()
    
    graph.eap_scores /= ig_steps

    del upstream_activations_difference
    gc.collect()
    torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= num_prompts
    graph.eap_scores = graph.eap_scores.cpu()

    return graph