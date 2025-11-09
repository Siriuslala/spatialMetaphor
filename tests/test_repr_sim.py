import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch.nn.functional import cosine_similarity

import jsonlines
from tqdm import tqdm
import random

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(root_dir.as_posix())

from utils.model_heplers import convert_token_list_to_readable, load_model_and_tokenizer


def load_data(language="en", use_wild_data=True, use_template=False, single_word=False, other_keywords=False, max_data_num=2000):
    if use_wild_data:
        data_dir = root_dir / f"data/wiki/{language}"
    else:
        if use_template:
            data_dir = root_dir / f"data/data_single_context/{language}"
        else:
            data_dir = root_dir / f"data/data_various_context/{language}"

    if not single_word:
        concepts = ["emotion_positive", "emotion_negative", "orientation_positive", "orientation_negative"]
    else:
        concepts = ["emotion_happy", "emotion_sad", "orientation_up", "orientation_down"]
    
    if other_keywords:
        other_concepts = ["emotion_cheerful", "others_beautiful", "others_old", "others_water", "others_boy", "others_know", "others_go", "others_she"]
        concepts += other_concepts

    data = {}
    for concept in concepts:
        concept_data_path = data_dir / f"{concept}.jsonl"
        with jsonlines.open(concept_data_path) as f:
            data[concept] = {
                "sentences": [],
                "target": []
            }
            cnt = 0
            for line in f:
                data[concept]["sentences"].append(line["sentence"])
                data[concept]["target"].append(line["keyword"])
                cnt += 1
                if cnt >= max_data_num:
                    break
    # data["others"] = {
    #     "sentences": [],
    #     "target": []
    # }
    # for concept in other_concepts:
    #     data["others"]["sentences"].extend(data[concept]["sentences"])
    #     data["others"]["target"].extend(data[concept]["target"])
    #     del data[concept]
    return data

@torch.no_grad()
def extract_representations(sentences, target_words, model, tokenizer, layer_index, device):
    """
    Extract representations of target words from a batch of sentences.
    """
    # get tokens
    tokenized_sentences = []
    for i, sentence in enumerate(sentences):
        if "gpt2" in model_name:
            continue
        tokens = tokenizer.tokenize(sentence)
        try:        
            tokens = convert_token_list_to_readable(tokens)
        except ValueError as e:
            raise ValueError(f"Error: {e} In sentence {i}: {sentence}, {tokens}")
        tokenized_sentences.append(tokens)
        # print(f"Sentence {i}: {tokens}")
    # breakpoint()

    inputs = tokenizer(
        sentences, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        return_offsets_mapping=True # <-- 请求偏移量映射
    ).to(device)
    # print(inputs)
    # breakpoint()
    
    # Run the model
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[layer_index] # Shape: [batch_size, seq_len, hidden_dim]
    offset_mappings = inputs.offset_mapping.cpu().numpy()
    
    # Get target vectors
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
            # (start_char, end_char) == (0, 0) 表示这是特殊 token (如 [CLS], [PAD], <|im_start|>),需要跳过
            if start_char == 0 and end_char == 0:
                continue  
            # 检查这个 token 的 字符范围 是否"落在" 目标词的 字符范围 之内
            # 条件：token的起始 >= 词的起始 AND token的结束 <= 词的结束
            if end_char > target_start_char and start_char < target_end_char:
                target_token_indices.append(k)
        
        if target_token_indices:
            # target_token_indices 可能是 [5], 或者 [5, 6] (如果 "高兴" 被分成2个token)
            # 从 hidden_states 中提取所有对应的向量
            vectors_slice = hidden_states[i, target_token_indices, :]
            mean_vector = torch.mean(vectors_slice, dim=0).cpu()
            target_vectors.append(mean_vector)
            # verify
            # extracted_tokens = [tokenized_sentences[i][idx] for idx in target_token_indices]
            # print(f"Sentence {i}: {tokenized_sentences[i]}, Extracted tokens for '{target_word}': {extracted_tokens}")
        else:
            print(f"警告：在 '{sentence}' 中找到了 '{target_word}'，但未能在 offsets 中匹配到 token。")
            print(f"    Offsets: {offsets}")
            print(f"Target position (chars): [{target_start_char}, {target_end_char}]")
            print(f"    Target Chars: [{target_start_char}, {target_end_char}]")
    # breakpoint()

    # randomly select a vector from each sentence
    # random_vectors = []
    # for i in range(len(sentences)):
    #     # get random index (not pad token)
    #     random_idx = torch.randint(0, hidden_states.shape[1], (1,)).item()
    #     while inputs["input_ids"][i, random_idx] == tokenizer.pad_token_id:
    #         random_idx = torch.randint(0, hidden_states.shape[1], (1,)).item()
    #     random_vectors.append(hidden_states[i, random_idx, :].cpu())

    # return torch.stack(target_vectors), torch.stack(random_vectors)

    return torch.stack(target_vectors)

def extract_representations_batch(sentences, target_words, model, tokenizer, layer_index, device, batch_size=16):
    """
    Extract representations of target words from a batch of sentences.
    """
    target_vectors_all = torch.empty(0, model.config.hidden_size).to("cpu")
    # random_vectors_all = torch.empty(0, model.config.hidden_size).to("cpu")
    for _ in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[:batch_size]
        batch_target = target_words[:batch_size]
        batch_vectors = extract_representations(
            batch_sentences, 
            batch_target, 
            model, 
            tokenizer, 
            layer_index,
            device,
        )
        # stack the target vectors
        target_vectors_all = torch.cat((target_vectors_all, batch_vectors), dim=0)  # shape: [num_samples, hidden_dim]
        # random_vectors_all = torch.cat((random_vectors_all, random_vectors), dim=0)  # shape: [num_samples, hidden_dim]
        sentences = sentences[batch_size:]
        target_words = target_words[batch_size:]
    if sentences:
        batch_vectors = extract_representations(
            sentences, 
            target_words,
            model, 
            tokenizer, 
            layer_index,
            device,
        )
        target_vectors_all = torch.cat((target_vectors_all, batch_vectors), dim=0)  # shape: [num_samples, hidden_dim]
        # random_vectors_all = torch.cat((random_vectors_all, random_vectors), dim=0)  # shape: [num_samples, hidden_dim]
    
    # return target_vectors_all, random_vectors_all
    return target_vectors_all

@torch.no_grad()
def extract_token_reps(sentences, model, tokenizer, layer_index, max_tokens=20000):
    all_vectors = []
    total_tokens = 0
    
    for sentence in sentences:
        if total_tokens >= max_tokens:
            break
            
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_index].cpu() # [1, seq_len, dim]
        # print(f"Hidden states shape: {hidden_states.shape}")
        # breakpoint()
        
        pad_token_id = tokenizer.pad_token_id
        non_pad_mask = inputs.input_ids.cpu() != pad_token_id
        vectors = hidden_states[non_pad_mask]
        
        all_vectors.append(vectors)
        total_tokens += vectors.shape[0]
        
    return torch.cat(all_vectors, dim=0)

def get_principal_components(vectors, k=5):
    """Compute the first K principal components using PCA."""
    # The sklearn PCA expects (n_samples, n_features)
    # We have vectors of shape (n_tokens, hidden_dim)

    vectors_np = vectors.float().numpy()
    pca = PCA(n_components=k)
    pca.fit(vectors_np)
    
    # The components_ of sklearn PCA are of shape (k, n_features)
    return torch.from_numpy(pca.components_) # unit vectors

def clean_vectors(vectors, components):
    """
    Subtract the projection of vectors onto components from vectors.
    vectors: (N, dim)
    components: (K, dim)
    """
    # components.T: (dim, K)
    # projections = v @ c.T  -> (N, K)
    projections = torch.matmul(vectors, components.T)
    
    # bias_to_remove = projections @ c  -> (N, K) @ (K, dim) -> (N, dim)
    bias_to_remove = torch.matmul(projections, components)
    
    # cleaned_vectors = v - bias_to_remove
    return vectors - bias_to_remove

def analyze_representations(model, tokenizer, data, layer_idx, device, batch_size, exp_name, single_word=False, other_keywords=None, **kwargs):
    print(f"Extracting representations at layer {layer_idx}...")

    # get target representations
    representations = {}
    for concept, info in data.items():
        sentences = info["sentences"]
        target = info["target"]
        target_vectors_all = torch.empty(0, model.config.hidden_size).to('cpu')
        target_vectors_all = extract_representations_batch(
            sentences, 
            target, 
            model, 
            tokenizer, 
            layer_idx,
            device,
            batch_size=batch_size,
        )
        representations[concept] = target_vectors_all
        # representations["random"] = [random_vectors_all] if "random" not in representations else representations["random"] + [random_vectors_all]
        print(f"Extracted representations for {concept}: {len(target_vectors_all)} vectors")
    # representations["random"] = torch.cat(representations["random"], dim=0)
    
    # get centroids
    ## get general reprs
    use_standardization = kwargs.get("use_standardization", False)
    del_rogue_method = kwargs.get("del_rogue_dim", None)

    if use_standardization or del_rogue_dim:
        general_data_path = root_dir / "data/pile10k/pile_en_1000.jsonl"
        general_sentences = []
        with jsonlines.open(general_data_path, "r") as f:
            for line in f:
                general_sentences.append(line["sentence"])
        general_vectors = extract_token_reps(general_sentences, model, tokenizer, layer_idx)

    ## standarization
    if use_standardization:
        mean_vec = general_vectors.mean(dim=0, keepdim=True)
        std_vec = general_vectors.std(dim=0, keepdim=True)
        for concept, reps in representations.items():
            representations[concept] = (reps - mean_vec) / std_vec

    ## delete rogue dimensions
    # del_rogue_method = exp_name.split("del_rogue_dim_")[-1]
    if del_rogue_method == "del_pca" and not use_standardization:
        pca_components = get_principal_components(general_vectors)
        # clean representations
        for concept, reps in representations.items():
            reps = clean_vectors(reps, pca_components)
            representations[concept] = reps

    # start analysis
    print("\n--- Method 1: Centroid Cosine Similarity ---")
    centroids = {concept: torch.mean(reps, dim=0, keepdim=True) for concept, reps in representations.items()}

    mu_happy = centroids["emotion_positive"] if not single_word else centroids["emotion_happy"]
    mu_up = centroids["orientation_positive"] if not single_word else centroids["orientation_up"]
    mu_sad = centroids["emotion_negative"] if not single_word else centroids["emotion_sad"]
    mu_down = centroids["orientation_negative"] if not single_word else centroids["orientation_down"]
    if other_keywords:
        mu_cheerful = centroids["emotion_cheerful"] if not single_word else centroids["emotion_cheerful"]
        other_centroids = {concept: centroids[concept] for concept in centroids if concept not in ["emotion_positive", "orientation_positive", "emotion_negative", "orientation_negative", "emotion_cheerful"]}

    sim_happy_up = cosine_similarity(mu_happy, mu_up).item()
    sim_sad_down = cosine_similarity(mu_sad, mu_down).item()
    sim_happy_down = cosine_similarity(mu_happy, mu_down).item()
    sim_sad_up = cosine_similarity(mu_sad, mu_up).item()
    if other_keywords:
        sim_happy_cheerful = cosine_similarity(mu_happy, mu_cheerful).item()
        sim_happy_sad = cosine_similarity(mu_happy, mu_sad).item()
        sim_happy_other = torch.mean(torch.stack([cosine_similarity(mu_happy, other) for other in other_centroids.values()])).item()
        sim_sad_other = torch.mean(torch.stack([cosine_similarity(mu_sad, other) for other in other_centroids.values()])).item()
        sim_up_other = torch.mean(torch.stack([cosine_similarity(mu_up, other) for other in other_centroids.values()])).item()
        sim_down_other = torch.mean(torch.stack([cosine_similarity(mu_down, other) for other in other_centroids.values()])).item()
    sim_pairs = {
        "happy_up": sim_happy_up,
        "sad_down": sim_sad_down,
        "happy_down": sim_happy_down,
        "sad_up": sim_sad_up,
    }
    if other_keywords:
        sim_pairs.update({
            "happy_cheerful": sim_happy_cheerful,
            "happy_sad": sim_happy_sad,
            "happy_other": sim_happy_other,
            "sad_other": sim_sad_other,
            "up_other": sim_up_other,
            "down_other": sim_down_other,
        })
    print(f"sim(happy, up):   {sim_happy_up:.4f}")
    print(f"sim(sad, down):   {sim_sad_down:.4f}")
    print(f"sim(happy, down): {sim_happy_down:.4f}")
    print(f"sim(sad, up):     {sim_sad_up:.4f}")
    if other_keywords:
        print(f"sim(happy, cheerful):     {sim_happy_cheerful:.4f}")
        print(f"sim(happy, sad):     {sim_happy_sad:.4f}")
        print(f"sim(happy, other):     {sim_happy_other:.4f}")
        print(f"sim(sad, other):     {sim_sad_other:.4f}")
        print(f"sim(up, other):     {sim_up_other:.4f}")
        print(f"sim(down, other):     {sim_down_other:.4f}")

    # Axis Similarity
    print("\n--- Method 2: Axis Similarity ---")
    d_sentiment = mu_happy - mu_sad
    d_spatial = mu_up - mu_down

    axis_similarity = cosine_similarity(d_sentiment, d_spatial).item()
    print(f"Similarity between sentiment axis (happy-sad) and spatial axis (up-down): {axis_similarity:.4f}")

    return sim_pairs, axis_similarity


    # Visualization
    print("\n--- Method 3: t-SNE Visualization ---")
    all_vectors = []
    labels = []
    colors = []
    color_map = {"emotion_positive" if not single_word else "emotion_happy": "red", 
                 "orientation_positive" if not single_word else "orientation_up": "orange", 
                 "emotion_negative" if not single_word else "emotion_sad": "blue", 
                 "orientation_negative" if not single_word else "orientation_down": "purple"}

    for concept, reps in representations.items():
        all_vectors.append(reps)
        for _ in range(reps.shape[0]):
            labels.append(concept)
            colors.append(color_map[concept])

    if all_vectors:
        all_vectors_cat = torch.cat(all_vectors, dim=0).float().numpy()
        
        all_vectors_cat = normalize(all_vectors_cat, axis=1)

        tsne = TSNE(n_components=2, perplexity=min(5, all_vectors_cat.shape[0]-1), random_state=41, init='pca', learning_rate='auto')
        vectors_2d_tsne = tsne.fit_transform(all_vectors_cat)

        pca = PCA(n_components=2)
        vectors_2d_pca = pca.fit_transform(all_vectors_cat)

        def draw(vectors_2d, method="tsne"):
            plt.figure(figsize=(10, 8))
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
            plt.title(f'{method} of Representations (Layer {layer_idx})')
            plt.xlabel(f'{method} Component 1')
            plt.ylabel(f'{method} Component 2')
            plt.grid(True)
            fig_dir = root_dir / f"figures/{exp_name}"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_dir / f"{method}_representation_layer_{layer_idx}.png")
            print(f"{method} image saved to {fig_dir / f'{method}_representation_layer_{layer_idx}.png'}")
        draw(vectors_2d_tsne, method="tsne")
        draw(vectors_2d_pca, method="pca")
    else:
        print("No sufficient vectors to visualize.")
    
    return sim_happy_up, sim_sad_down, sim_happy_down, sim_sad_up, axis_similarity

def check_spatial_metaphor_repr(model_name, device, batch_size=32, use_wild_data=True, use_template=False, single_word=False, del_rogue_dim=False, del_rogue_method="del_pca", use_standarization=False, other_keywords=None):
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    data = load_data(language="en", use_wild_data=use_wild_data, use_template=use_template, single_word=single_word, other_keywords=other_keywords)

    model_name = model_name.split("/")[-1]
    exp_name = f"repr_sim"
    if use_template:
        exp_name += "-use_template"
    if single_word:
        exp_name += "-single_word"
    if del_rogue_dim and use_standarization:
        raise ValueError("use_standarization must be False when del_rogue_dim is True")
    if del_rogue_dim:
        exp_name += f"-del_rogue_dim_{del_rogue_method}"
    if use_standarization:
        exp_name += "-use_standarization"
    exp_name = f"{exp_name}/{model_name}"

    sim_happy_up_list = []
    sim_sad_down_list = []
    sim_happy_down_list = []
    sim_sad_up_list = []
    if other_keywords:
        sim_cheerful_happy_list = []
        sim_happy_sad_list = []
        sim_happy_other_list = []
        sim_sad_other_list = []
        sim_up_other_list = []
        sim_down_other_list = []
    axis_similarity_list = []
    for layer_id in range(model.config.num_hidden_layers):
        sim_pairs, axis_similarity = analyze_representations(
            model, 
            tokenizer, 
            data,
            layer_id, 
            device, 
            batch_size,
            exp_name, 
            single_word=single_word, 
            del_rogue_method=del_rogue_method, 
            use_standarization=use_standarization,
            other_keywords=other_keywords
        )
        sim_happy_up_list.append(sim_pairs["happy_up"])
        sim_sad_down_list.append(sim_pairs["sad_down"])
        sim_happy_down_list.append(sim_pairs["happy_down"])
        sim_sad_up_list.append(sim_pairs["sad_up"])
        if other_keywords:
            sim_cheerful_happy_list.append(sim_pairs["happy_cheerful"])
            sim_happy_sad_list.append(sim_pairs["happy_sad"])
            sim_happy_other_list.append(sim_pairs["happy_other"])
            sim_sad_other_list.append(sim_pairs["sad_other"])
            sim_up_other_list.append(sim_pairs["up_other"])
            sim_down_other_list.append(sim_pairs["down_other"])
        axis_similarity_list.append(axis_similarity)
    
    # plot layer-wise results
    layers = list(range(model.config.num_hidden_layers))
    plt.figure(figsize=(12, 8))
    plt.plot(layers, sim_happy_up_list, label='sim(happy, up)', marker='o')
    plt.plot(layers, sim_sad_down_list, label='sim(sad, down)', marker='o')
    plt.plot(layers, sim_happy_down_list, label='sim(happy, down)', marker='o')
    plt.plot(layers, sim_sad_up_list, label='sim(sad, up)', marker='o')
    if other_keywords:
        plt.plot(layers, sim_cheerful_happy_list, label='sim(cheerful, happy)', marker='o')
        plt.plot(layers, sim_happy_sad_list, label='sim(happy, sad)', marker='o')
        plt.plot(layers, sim_happy_other_list, label='sim(happy, other)', marker='o')
        plt.plot(layers, sim_sad_other_list, label='sim(sad, other)', marker='o')
        plt.plot(layers, sim_up_other_list, label='sim(up, other)', marker='o')
        plt.plot(layers, sim_down_other_list, label='sim(down, other)', marker='o')
    plt.plot(layers, axis_similarity_list, label='axis similarity', marker='o')

    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    fig_dir = root_dir / f"figures/{exp_name}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "layer_wise_similarities.png")
    print(f"Layer-wise similarity plot saved to {fig_dir / 'layer_wise_similarities.png'}")

    # save info in jsonl
    info_path = fig_dir / "layer_wise_similarities.jsonl"
    with jsonlines.open(info_path, "w") as f:
        for layer_idx in range(model.config.num_hidden_layers):
            f.write({
                "layer_idx": layer_idx,
                "sim_happy_up": sim_happy_up_list[layer_idx],
                "sim_sad_down": sim_sad_down_list[layer_idx],
                "sim_happy_down": sim_happy_down_list[layer_idx],
                "sim_sad_up": sim_sad_up_list[layer_idx],
                "axis_similarity": axis_similarity_list[layer_idx],
                "sim_cheerful_happy": sim_cheerful_happy_list[layer_idx] if other_keywords else None,
                "sim_happy_sad": sim_happy_sad_list[layer_idx] if other_keywords else None,
                "sim_happy_other": sim_happy_other_list[layer_idx] if other_keywords else None,
                "sim_sad_other": sim_sad_other_list[layer_idx] if other_keywords else None,
                "sim_up_other": sim_up_other_list[layer_idx] if other_keywords else None,
                "sim_down_other": sim_down_other_list[layer_idx] if other_keywords else None,
            })


if __name__ == "__main__":
    pass

    model_name = "Qwen/Qwen2.5-7B-Instruct"  # "Qwen/Qwen3-8B", "Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B-Instruct", "openai-community/gpt2"
    device = "cuda:5"
    batch_size = 32  # qwen3-8B bsz=32 mem=18000M
    use_wild_data = True
    use_template = False
    single_word = True  # happy, cheerful, ... or simply "happy"
    del_rogue_dim = False
    del_rogue_method = "del_pca"
    use_standarization = False
    other_keywords = None
    check_spatial_metaphor_repr(model_name, device, batch_size=batch_size, use_wild_data=use_wild_data, use_template=use_template, single_word=single_word, del_rogue_dim=del_rogue_dim, del_rogue_method=del_rogue_method, use_standarization=use_standarization, other_keywords=other_keywords)

