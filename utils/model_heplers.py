import chardet
from transformers import AutoTokenizer, AutoModelForCausalLM


def bytes_to_unicode():
    """
    生成字节到Unicode字符的正向映射表
    返回字典：{byte_value: unicode_char}
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
        detected_encoding = chardet.detect(bytes)['encoding']
        return bytes.decode(detected_encoding, errors='replace')

def convert_token_list_to_readable(token_list):
    return [convert_to_readable_token(token) for token in token_list]

def load_model_and_tokenizer(model_name, device='cuda'):
    print(f"正在加载模型和Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=False,
        torch_dtype="auto" # 使用 bfloat16 或 float16 节省显存
    ).to(device)
    model.eval() # 设置为评估模式

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"模型 {model_name} 加载完毕。")

    return model, tokenizer
