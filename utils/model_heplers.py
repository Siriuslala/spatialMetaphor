import chardet
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from typing import List, Tuple
import os

def bytes_to_unicode():
    """
    Create a forward mapping from byte values to Unicode characters.
    Returns:
        dict: A dictionary mapping byte values (0-255) to their corresponding Unicode characters.
    """
    # 原始保留的字节范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) +          # ASCII可打印字符（33-126）
        list(range(ord("¡"), ord("¬") + 1)) +          # 西班牙语特殊字符（161-172）
        list(range(ord("®"), ord("ÿ") + 1))            # 其他扩展字符（174-255）
    )
    
    cs = bs.copy()  # 初始字符列表
    n = 0
    
    # 遍历所有可能的字节（0-255）
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)  # 超出原始范围的字节映射到更高Unicode码位
            n += 1
    
    # 将码位转换为Unicode字符
    cs = [chr(code) for code in cs]
    
    return dict(zip(bs, cs))

def unicode_str_to_bytes(unicode_str):
    forward_map = bytes_to_unicode()
    reverse_map = {v: k for k, v in forward_map.items()}
    return bytes([reverse_map[c] for c in unicode_str])

def convert_to_readable_token(unicode_token):
    byte_seq = unicode_str_to_bytes(unicode_token)
    try:
        return byte_seq.decode("utf-8")
    except UnicodeDecodeError:
        # return byte_seq.decode("latin-1")
        detected_encoding_info = chardet.detect(byte_seq)
        detected_encoding = detected_encoding_info.get('encoding')
        if detected_encoding:
            return byte_seq.decode(detected_encoding, errors='replace')
        else:
            return byte_seq.decode("latin-1", errors='replace')

def convert_token_list_to_readable(token_list):
    return [convert_to_readable_token(token) for token in token_list]

def load_model_and_tokenizer(model_name, device='cuda'):
    print(f"Loading model {model_name} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto" # 使用 bfloat16 或 float16 节省显存
    ).to(device)
    model.eval() # 设置为评估模式

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model {model_name} loaded successfully.")

    return model, tokenizer

def batch_infer(model, tokenizer, batch, max_tokens: int = 512, temperature: float = 0.0):
    """
    Args:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for tokenization.
        batch: A list of input strings to be processed by the model.
    Returns:
        The output of the model for the given batch.
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # get length of each input
    input_lengths = [len(ids) for ids in inputs['input_ids']]
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=False,
    )
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_texts = [text[len(prompt):].strip() for text, prompt in zip(generated_texts, batch)]
    return generated_texts

def load_model_and_tokenizer_vllm(model_name: str, gpu_ids: List[int]):

    print(f"Loading model {model_name} and tokenizer using vLLM...")

    gpu_string = ",".join(map(str, gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_string
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=len(gpu_ids),
        dtype="auto", 
        trust_remote_code=True,
        gpu_memory_utilization=0.25,
        swap_space=4,
    )
    print(f"Model {model_name} loaded successfully on {len(gpu_ids)} GPUs.")

    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return llm, tokenizer

def vllm_infer(llm, batch, max_tokens: int = 32, temperature: float = 0.7):
    """
    Args:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for tokenization.
        batch: A list of input strings to be processed by the model.
    Returns:
        The output of the model for the given batch.
    """
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    outputs = llm.generate(batch, sampling_params)

    generated_texts = []
    for output in outputs:
        if output.outputs:
            generated_texts.append(output.outputs[0].text.strip())
        else:
            generated_texts.append("")
            
    return generated_texts


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    batch = ["你好", "0.10000+0.200000等于多少？"] * 4
    use_vllm = False
    if use_vllm:
        gpu_ids = [0, 1]
        llm, tokenizer = load_model_and_tokenizer_vllm(model_name, gpu_ids)
        outputs = vllm_infer(llm, batch)
        print(outputs)
    else:
        model, tokenizer = load_model_and_tokenizer(model_name, device='cuda:2')
        outputs = batch_infer(model, tokenizer, batch)
        