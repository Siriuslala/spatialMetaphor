"""
Define and train probes to identify metaphor-related axes in model representations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import jsonlines
from sklearn.model_selection import train_test_split

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(root_dir.as_posix())

from utils.model_heplers import load_model_and_tokenizer
from tests.test_repr_sim import extract_representations, extract_representations_batch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- 3. 新的、多样化的数据集 ---
# !! 警告：这个数据量 (20) 仍然太小，仅用于演示。
# !! 你必须将每个列表扩展到至少 100-200 个！

# (happy, sad) -> ("高兴", "悲伤")
emotion_data = {}
# (up, down) -> ("向上", "向下")
orientation_data = {}

def load_data_gemini_gen(language="en"):
    data_dir = root_dir / f"data/data_various_context/{language}"
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
    return data

def load_data_pile(language="en"):
    data_dir = root_dir / f"data/pile10k"
    concepts = ["emotion_positive", "emotion_negative", "orientation_positive", "orientation_negative"]
    data = {}
    for concept in concepts:
        concept_data_path_name = data_dir / f"pile_{concept}_{language}_*.jsonl"
        concept_data_paths = list(concept_data_path_name.parent.glob(concept_data_path_name.name))
        with jsonlines.open(concept_data_paths[0]) as f:
            data[concept] = {
                "sentences": [],
                "target": []
            }
            for line in f:
                data[concept]["sentences"].append(line["sentence"])
                data[concept]["target"].append(line["keyword"])
    return data

def load_train_data(language="en"):
    data = load_data_pile(language)
    emotion_data = {
        "happy": {
            "sentences": data["emotion_positive"]["sentences"],
            "target": data["emotion_positive"]["target"]
        },
        "sad": {
            "sentences": data["emotion_negative"]["sentences"],
            "target": data["emotion_negative"]["target"]
        }
    }
    orientation_data = {
        "up": {
            "sentences": data["orientation_positive"]["sentences"],
            "target": data["orientation_positive"]["target"]
        },
        "down": {
            "sentences": data["orientation_negative"]["sentences"],
            "target": data["orientation_negative"]["target"]
        }
    }
    return emotion_data, orientation_data


class ProbeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Probe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Probe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_probe(probe, data_loaders, args):
    """
    Args:
        probe: Probe,
        data_loaders: (train_data_loader, val_data_loader),
        args: ArgumentParser,
    """
    train_data_loader, val_data_loader = data_loaders

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    probe.to(device)
    probe.train()
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_data_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        # 在最后一个epoch打印信息
        if epoch == args.num_epochs - 1:
            print(f'  Probe Training: Epoch [{epoch+1}/{args.num_epochs}], Loss: {total_loss/len(train_data_loader):.4f}, Acc: {accuracy:.2f}%')
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = probe(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f'  Probe Training Finished. Test Accuracy: {test_accuracy:.2f}%')

    probe.to("cpu")

    return test_accuracy

# --- 5. 主实验循环 ---
def main(args):
    model, tokenizer = load_model_and_tokenizer(args.model_name, device=args.device)
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    args.model_name = args.model_name.split('/')[-1]

    emotion_data, orientation_data = load_train_data(language=args.language)

    # hidden_states[0] is embedding，
    num_layers_to_probe = num_layers + 1 # e.g., 33 for 32-layer model
    axis_similarities = []
    emotion_probe_accuracies = []
    spatial_probe_accuracies = []

    for layer_index in range(num_layers_to_probe):
        print(f"\n--- Processing layer {layer_index} ---")
        
        # Training probes for happy/sad
        ## Prepare data for probes
        V_happy = extract_representations_batch(
            emotion_data["happy"]["sentences"], 
            [keyword for keyword in emotion_data["happy"]["target"]],
            model, tokenizer, layer_index, args.device, batch_size=args.batch_size
        )
        V_sad = extract_representations_batch(
            emotion_data["sad"]["sentences"], 
            [keyword for keyword in emotion_data["sad"]["target"]],
            model, tokenizer, layer_index, args.device, batch_size=args.gen_batch_size
        )

        features_emotion = torch.cat([V_happy, V_sad]).float()
        labels_emotion = torch.cat([
            torch.ones(V_happy.shape[0]), 
            torch.zeros(V_sad.shape[0])
        ]).long()

        # Split data into train and test sets
        indices = list(range(len(features_emotion)))
        train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=41, stratify=labels_emotion)
        
        full_dataset_emo = ProbeDataset(features_emotion, labels_emotion)
        train_dataset_emo = Subset(full_dataset_emo, train_indices)
        test_dataset_emo = Subset(full_dataset_emo, test_indices)
        
        train_loader_emo = DataLoader(train_dataset_emo, batch_size=32, shuffle=True)
        test_loader_emo = DataLoader(test_dataset_emo, batch_size=32, shuffle=False)
        emotion_loader = (train_loader_emo, test_loader_emo)
        
        ## Define and train probes
        emotion_probe = Probe(hidden_dim, 2)
        emotion_acc = train_probe(emotion_probe, emotion_loader, args)
        emotion_probe_accuracies.append(emotion_acc)
        
        # 提取权重。
        # W_0 = 类别0 (sad) 的权重向量
        # W_1 = 类别1 (happy) 的权重向量
        # 我们关心的方向是 W_1 - W_0
        W_emotion = (emotion_probe.linear.weight.data[1] - 
                    emotion_probe.linear.weight.data[0])

        # Training probes for up/down
        ## Prepare data for probes
        V_up = extract_representations_batch(
            orientation_data["up"]["sentences"], 
            [keyword for keyword in orientation_data["up"]["target"]],
            model, tokenizer, layer_index, args.device, batch_size=args.gen_batch_size
        )
        V_down = extract_representations_batch(
            orientation_data["down"]["sentences"], 
            [keyword for keyword in orientation_data["down"]["target"]],
            model, tokenizer, layer_index, args.device, batch_size=args.gen_batch_size
        )

        features_spatial = torch.cat([V_up, V_down]).float()
        labels_spatial = torch.cat([
            torch.ones(V_up.shape[0]), 
            torch.zeros(V_down.shape[0])
        ]).long()

        # Split data into train and test sets
        indices = list(range(len(features_spatial)))
        train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=41, stratify=labels_spatial)
        
        full_dataset_spat = ProbeDataset(features_spatial, labels_spatial)
        train_dataset_spat = Subset(full_dataset_spat, train_indices)
        test_dataset_spat = Subset(full_dataset_spat, test_indices)
        
        train_loader_spat = DataLoader(train_dataset_spat, batch_size=32, shuffle=True)
        test_loader_spat = DataLoader(test_dataset_spat, batch_size=32, shuffle=False)
        spatial_loader = (train_loader_spat, test_loader_spat)

        ## Define and train probes
        spatial_probe = Probe(hidden_dim, 2)
        spatial_acc = train_probe(spatial_probe, spatial_loader, args)
        spatial_probe_accuracies.append(spatial_acc)

        # 提取权重 (W_1 - W_0)
        W_spatial = (spatial_probe.linear.weight.data[1] - 
                    spatial_probe.linear.weight.data[0])

        # 3. 计算相似度
        # (如果探针的准确率太低，比如低于60%，说明它没学到任何东西，
        # 它的权重是随机的，计算相似度没有意义)
        if emotion_acc < 60 or spatial_acc < 60:
            print(f"  The accuracy of probes is too low (Emo: {emotion_acc}%, Spat: {spatial_acc}%)，results may be unreliable.")
            sim = np.nan
        else:
            sim_tensor = cosine_similarity(W_emotion.unsqueeze(0), W_spatial.unsqueeze(0))
            sim = sim_tensor.item()
        
        axis_similarities.append(sim)
        print(f"--- Layer {layer_index} Axis Similarity: {sim:.4f} ---")

    # Plotting results
    plt.figure(figsize=(12, 7))
    plt.plot(range(num_layers_to_probe), axis_similarities, marker='o')
    # plt.title('Probe Axis Similarity vs. Layer Index')
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity of Probe Weights (W_emo vs W_spat)')
    plt.xticks(range(0, num_layers_to_probe, 2))
    plt.grid(True)
    fig_dir = root_dir / "figures/probing"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"probe_axis_similarity_{args.model_name}_{args.language}-epoch_{args.num_epochs}-lr_{args.learning_rate}-wd_{args.weight_decay}.png"
    plt.savefig(fig_path)
    print(f"\nThe experiment is complete. The result figure is saved as {fig_path}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.10)
    parser.add_argument("--language", type=str, default="en")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
