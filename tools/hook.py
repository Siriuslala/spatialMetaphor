from functools import partial

def gather_acts_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
    # print(f"inputs type: {type(inputs)}")
    # print(f"outputs type: {type(outputs)}")
    # print(f"inputs len: {len(inputs)}")
    # print(f"outputs len: {len(outputs)}")
    # breakpoint()
    # In a layer, the inputs and outputs are a tuple of one tensor of size (batch_size, seq_len, hidden_size)
    acts = inputs[0].squeeze(0) if use_input else outputs[0]
    cache[key] = acts.detach()
    return outputs
    
def gather_residual_activations(model, target_layer, inputs):
    cache = {}
    # register the `gather_acts_hook` to the target layer
    handle = model.model.layers[target_layer].register_forward_hook(
        partial(gather_acts_hook, cache=cache, key="resid_post", use_input=False)
    )
    try:
        _ = model(inputs)
    finally:
        handle.remove()

    return cache["resid_post"]

def gather_transcoder_activations(model, target_layer, inputs):

    cache = {}

    handle_input = model.model.layers[target_layer].pre_feedforward_layernorm.register_forward_hook(
        partial(gather_acts_hook, cache=cache, key="transcoder_input", use_input=False)
    )
    handle_target = model.model.layers[target_layer].post_feedforward_layernorm.register_forward_hook(
        partial(gather_acts_hook, cache=cache, key="transcoder_target", use_input=False)
    )

    try:
        _ = model.forward(inputs)
    finally:
        handle_input.remove()
        handle_target.remove()

    return cache
