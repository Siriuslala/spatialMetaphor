import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch.nn.functional import cosine_similarity

import jsonlines
from tqdm import tqdm

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(root_dir.as_posix())


# --- 1. 配置 ---
MODEL_NAME = "Qwen/Qwen3-8B" # 或者 Qwen/Qwen2-1.5B
LAYER_TO_PROBE = 30 # 探测中间层 (Qwen2-7B有32层, 1.5B有28层, 选个中间的)
device = "cuda:5" if torch.cuda.is_available() else "cpu"

# --- 2. 加载模型和Tokenizer ---
# 注意：Qwen2需要 trust_remote_code=True
print("正在加载模型和Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    torch_dtype="auto" # 使用 bfloat16 或 float16 节省显存
).to(device)
model.eval() # 设置为评估模式

# 适配 Qwen2 Tokenizer 的 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("模型加载完毕。")

# --- 3. 探针数据集 ---
# (为了演示，我们用少量句子，实际实验应使用成百上千句)
language = "en"
data_dir = root_dir / f"data/data_single_context/{language}"
concepts = ["emotion_positive", "emotion_negative", "orientation_positive", "orientation_negative"]
data = {}
for concept in concepts:
    concept_data_path = data_dir / f"{concept}.jsonl"
    with jsonlines.open(concept_data_path) as f:
        data[concept] = {
            "sentences": [],
            "target": []
        }
        for line in f:
            data[concept]["sentences"].append(line["sentence"])
            data[concept]["target"].append(line["keyword"])

# --- 4. 核心功能：表征提取 ---
@torch.no_grad()
def extract_representations(sentences, target_words, model, tokenizer, layer_index):
    """
    从一个batch的句子中提取目标词的表征。
    """
    # get tokens
    # tokenized_sentences = []
    # for i, sentence in enumerate(sentences):
    #     tokens = tokenizer.tokenize(sentence)
    #     tokenized_sentences.append(tokens)
    #     print(f"Sentence {i}: {tokens}")
    inputs = tokenizer(
        sentences, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        return_offsets_mapping=True # <-- 请求偏移量映射
    ).to(device)
    # print(inputs)
    # breakpoint()
    
    # 运行模型
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[layer_index] # Shape: [batch_size, seq_len, hidden_dim]
    offset_mappings = inputs.offset_mapping.cpu().numpy()
    
    target_vectors = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        target_word = target_words[i]
        
        target_start_char = sentence.find(target_word)
        if target_start_char == -1:
            print(f"警告：在句子 '{sentence}' 中未找到子字符串 '{target_word}'")
            continue
        target_end_char = target_start_char + len(target_word)

        offsets = offset_mappings[i]

        # 存放所有属于 target_word 的 token 的索引
        target_token_indices = []
        for k, (start_char, end_char) in enumerate(offsets):
            # (start_char, end_char) == (0, 0) 表示这是特殊 token (如 [CLS], [PAD], <|im_start|>)
            # 我们需要跳过它们
            if start_char == 0 and end_char == 0:
                continue  
            # 检查这个 token 的 字符范围 是否
            # "落在" 目标词的 字符范围 之内
            # 条件：token的起始 >= 词的起始 AND token的结束 <= 词的结束
            if end_char > target_start_char and start_char < target_end_char:
                target_token_indices.append(k)
        
        if target_token_indices:
            # target_token_indices 可能是 [5], 或者 [5, 6] (如果 "高兴" 被分成2个token)
            # 从 hidden_states 中提取所有对应的向量
            vectors_slice = hidden_states[i, target_token_indices, :]
            
            # 计算平均值 (即使只有一个 token，取 "mean" 也是安全的)
            mean_vector = torch.mean(vectors_slice, dim=0).cpu()
            target_vectors.append(mean_vector)
        else:
            # 这种情况不应该发生，除非 find() 找到了，但 offsets 没匹配上
            # (可能是因为 `tokenizer` 对原始字符串做了规范化，如NFC)
            # 但对于我们的模板化句子，这基本不可能
            print(f"警告：在 '{sentence}' 中找到了 '{target_word}'，但未能在 offsets 中匹配到 token。")
            print(f"    Offsets: {offsets}")
            print(f"Target position (chars): [{target_start_char}, {target_end_char}]")
            print(f"    Target Chars: [{target_start_char}, {target_end_char}]")

    if not target_vectors:
        print("警告：没有提取到任何向量。")
        return torch.empty(0, model.config.hidden_size)

    return target_vectors


# --- 5. 运行提取 ---
print("正在提取表征...")
representations = {}
for concept, info in data.items():
    sentences = info["sentences"]
    target = info["target"]
    batch_size = 16
    target_vectors_all = []
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Processing {concept}"):
        batch_sentences = sentences[:batch_size]
        batch_target = target[:batch_size]
        batch_vectors = extract_representations(
            batch_sentences, 
            batch_target, 
            model, 
            tokenizer, 
            LAYER_TO_PROBE
        )
        target_vectors_all.extend(batch_vectors)
        sentences = sentences[batch_size:]
        target = target[batch_size:]
    if sentences:
        batch_vectors = extract_representations(
            sentences, 
            target, 
            model, 
            tokenizer, 
            LAYER_TO_PROBE
        )
        target_vectors_all.extend(batch_vectors)

    target_vectors_all = torch.stack(target_vectors_all)
    representations[concept] = target_vectors_all
    print(f"提取到 {concept} 的 {len(target_vectors_all)} 个表征")

# --- 6. 分析与量化 ---

# 方法一：质心距离
print("\n--- 方法一：质心余弦相似度 ---")
centroids = {concept: torch.mean(reps, dim=0, keepdim=True) for concept, reps in representations.items()}

mu_happy = centroids["emotion_positive"]
mu_up = centroids["orientation_positive"]
mu_sad = centroids["emotion_negative"]
mu_down = centroids["orientation_negative"]

sim_happy_up = cosine_similarity(mu_happy, mu_up).item()
sim_sad_down = cosine_similarity(mu_sad, mu_down).item()
sim_happy_down = cosine_similarity(mu_happy, mu_down).item()
sim_sad_up = cosine_similarity(mu_sad, mu_up).item()

print(f"sim(happy, up):   {sim_happy_up:.4f}")
print(f"sim(sad, down):   {sim_sad_down:.4f}")
print(f"sim(happy, down): {sim_happy_down:.4f}")
print(f"sim(sad, up):     {sim_sad_up:.4f}")

# 方法三：方向相似性
print("\n--- 方法三：方向（轴）相似度 ---")
d_sentiment = mu_happy - mu_sad
d_spatial = mu_up - mu_down

axis_similarity = cosine_similarity(d_sentiment, d_spatial).item()
print(f"情感轴 (happy-sad) 与 空间轴 (up-down) 的相似度: {axis_similarity:.4f}")

# --- 7. 可视化 ---

# 方法二：t-SNE 聚类可视化
print("\n--- 方法二：生成t-SNE可视化 ---")
all_vectors = []
labels = []
colors = []
color_map = {"emotion_positive": "red", "orientation_positive": "orange", "emotion_negative": "blue", "orientation_negative": "purple"}

for concept, reps in representations.items():
    all_vectors.append(reps)
    for _ in range(reps.shape[0]):
        labels.append(concept)
        colors.append(color_map[concept])

if all_vectors:
    all_vectors_cat = torch.cat(all_vectors, dim=0).float().numpy()
    
    # 归一化可能有助于t-SNE
    all_vectors_cat = normalize(all_vectors_cat, axis=1)

    tsne = TSNE(n_components=2, perplexity=min(5, all_vectors_cat.shape[0]-1), random_state=42, init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(all_vectors_cat)

    plt.figure(figsize=(10, 8))
    
    # 为了图例
    for concept in color_map:
        idx = [i for i, label in enumerate(labels) if label == concept]
        if idx:
            plt.scatter(
                vectors_2d[idx, 0], 
                vectors_2d[idx, 1], 
                c=color_map[concept], 
                label=concept, 
                alpha=0.7
            )

    plt.legend()
    plt.title(f't-SNE of Representations (Layer {LAYER_TO_PROBE})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig("tsne_representation.png")
    print("t-SNE图像已保存为 'tsne_representation.png'")
else:
    print("没有提取到足够的向量用于可视化。")