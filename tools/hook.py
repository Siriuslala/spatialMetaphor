from functools import partial

from transformers.tokenization_utils_base import BatchEncoding


def gather_acts_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
    # print(f"inputs type: {type(inputs)}")  # tuple
    # print(f"outputs type: {type(outputs)}")  # torch.Tensor
    # breakpoint()
    # print(f"inputs shape: {inputs[0].shape}")
    # print(f"outputs shape: {outputs[0].shape}")
    # breakpoint()
    acts = inputs if use_input else outputs
    if isinstance(acts, tuple):
        acts = acts[0]
    cache[key] = acts.detach()  # (bsz, seq_len, hidden_size)
    return outputs
    
def gather_residual_activations(model, target_layer, inputs):
    cache = {}
    # register the `gather_acts_hook` to the target layer
    handles = []
    if isinstance(target_layer, int):
        handle = model.model.layers[target_layer].register_forward_hook(
            partial(gather_acts_hook, cache=cache, key="resid_post", use_input=False)
        )
        handles.append(handle)
    elif isinstance(target_layer, list):
        for layer_id in target_layer:
            handle = model.model.layers[layer_id].register_forward_hook(
                partial(gather_acts_hook, cache=cache, key=f"resid_post_layer{layer_id}", use_input=False)
            )
            handles.append(handle)
    else:
        raise ValueError(f"target_layer must be int or list, but got {type(target_layer)}")

    try:
        if isinstance(inputs, (dict, BatchEncoding)):
            _ = model(**inputs)
        else:
            _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    if isinstance(target_layer, int):
        return cache["resid_post"]
    else:
        return cache

def gather_transcoder_activations(model, target_layer, inputs):

    cache = {}
    handles = []
    if isinstance(target_layer, int):
        handle_input = model.model.layers[target_layer].pre_feedforward_layernorm.register_forward_hook(
            partial(gather_acts_hook, cache=cache, key="transcoder_input", use_input=False)
        )
        handle_target = model.model.layers[target_layer].post_feedforward_layernorm.register_forward_hook(
            partial(gather_acts_hook, cache=cache, key="transcoder_target", use_input=False)
        )
        handles.append(handle_input)
        handles.append(handle_target)
    elif isinstance(target_layer, list):
        for layer_id in target_layer:
            handle_input = model.model.layers[layer_id].pre_feedforward_layernorm.register_forward_hook(
                partial(gather_acts_hook, cache=cache, key=f"transcoder_input_layer{layer_id}", use_input=False)
            )
            handle_target = model.model.layers[layer_id].post_feedforward_layernorm.register_forward_hook(
                partial(gather_acts_hook, cache=cache, key=f"transcoder_target_layer{layer_id}", use_input=False)
            )
            handles.append(handle_input)
            handles.append(handle_target)
    else:
        raise ValueError(f"target_layer must be int or list, but got {type(target_layer)}")
    
    try:
        if isinstance(inputs, (dict, BatchEncoding)):
            _ = model(**inputs)
        else:
            _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    return cache
