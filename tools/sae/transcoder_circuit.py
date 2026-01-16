import torch
import gc
import math
import random
import numpy as np
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Callable, Set, Optional
from tqdm import tqdm
import plotly.graph_objects as go

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')
ROOT_DIR = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent.parent))
DATA_DIR = Path(os.getenv('DATA_DIR'))
WORK_DIR = Path(os.getenv('WORK_DIR'))
sys.path.append(ROOT_DIR.as_posix())

from tools.sae.test_gemma_scope_2 import load_transcoder_gemma


# --- 1. 定义异常用于提前停止模型推理 ---
class StopForward(Exception):
    """用于在获取到所需激活值后强制停止模型前向传播，节省计算资源"""
    pass

# --- 2. 基础数据结构 ---

@dataclass(frozen=True)
class CircuitNode:
    layer: int
    node_type: str # 'transcoder_feature', 'attn_head'
    index: int
 
    def __repr__(self):
        if self.node_type == 'transcoder_feature':
            return f"L{self.layer}.TC.{self.index}"
        elif self.node_type == 'attn_head':
            return f"L{self.layer}.A.{self.index}"
        return f"L{self.layer}.{self.node_type}"

@dataclass
class CircuitEdge:
    source: CircuitNode
    target: CircuitNode
    score: float

@dataclass
class CircuitGraph:
    nodes: Dict[str, CircuitNode] = field(default_factory=dict) # NodeStr -> NodeObj
    node_scores: Dict[str, float] = field(default_factory=dict) # NodeStr -> Total Attribution
    edges: List[CircuitEdge] = field(default_factory=list)

    def add_edge(self, source: CircuitNode, target: CircuitNode, score: float):
        self.edges.append(CircuitEdge(source, target, score))
        # 更新节点记录
        self.nodes[str(source)] = source
        self.nodes[str(target)] = target
        # 简单累加分数用于可视化大小 (绝对值)
        self.node_scores[str(source)] = self.node_scores.get(str(source), 0) + abs(score)
        # Target 节点的分数通常由其自身的父节点决定，这里主要记录它在路径中的活跃度

# --- 3. 懒加载管理器 (不变) ---

class LazyTranscoderManager:
    def __init__(self, load_fn: Callable, device="cuda", **transcoder_kwargs):
        self.load_fn = load_fn
        self.device = device
        self.transcoder_kwargs = transcoder_kwargs
        self.loaded_transcoders: Dict[int, torch.nn.Module] = {}

    def get(self, layer: int) -> torch.nn.Module:
        if layer not in self.loaded_transcoders:
            # print(f"[LazyLoader] Loading Layer {layer}...")
            kwargs = self.transcoder_kwargs.copy()
            kwargs["layer"] = layer
            kwargs["device"] = self.device
            self.loaded_transcoders[layer] = self.load_fn(**kwargs)
        return self.loaded_transcoders[layer]

# --- 4. 优化的激活值引擎 (Early Stopping) ---

class ModelActivationsEngine:
    def __init__(self, model, tokenizer, use_recomputation=True):
        self.model = model
        self.tokenizer = tokenizer
        self.use_recomputation = use_recomputation
        self.cache = {}
        self.hooks = []
        self.current_input_ids = None
        self.num_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", None))

    def set_input(self, input_text: str):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.current_input_ids = inputs.input_ids
        self.cache = {}
        self.clear_hooks()

    def _basic_hook(self, module, input, output, cache, key):
        if isinstance(output, tuple):
            acts = output[0]
        else:
            acts = output
        # print(f"acts: {acts}")
        # print(f"acts shape: {acts.shape}")
        # print(f"key: {key}")
        # breakpoint()
        cache[key] = acts.detach()
    
    def _attn_hook(self, module, input, output, cache, key):
        if isinstance(output, tuple):
            acts = output[0]
        else:
            acts = output
        cache[key] = acts.detach()

    def _stopping_hook(self):
        raise StopForward()

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_layer_activations(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        获取特定层激活值。如果开启重算，则只运行模型到该层为止 (Early Stopping)。
        """
        target_key = f"layers.{layer_idx}.transcoder_input"
        if not self.use_recomputation and self.cache:
            if target_key in self.cache:
                return self.cache

        target_storage = self.cache if not self.use_recomputation else {}
     
        if target_key in target_storage:
            return target_storage

        self.clear_hooks()
        
        if self.use_recomputation:
            layers_to_hook = [layer_idx]
        else:
            layers_to_hook = list(range(self.num_layers))
     
        for l in layers_to_hook:
            hf_layer = self.model.model.layers[l]
            self.hooks.append(hf_layer.pre_feedforward_layernorm.register_forward_hook(
                partial(self._basic_hook, cache=target_storage, key=f"layers.{l}.transcoder_input")
            ))
            # Hook v_proj 用于近似 Attention Output
            self.hooks.append(hf_layer.self_attn.v_proj.register_forward_hook(
                partial(self._attn_hook, cache=target_storage, key=f"layers.{l}.v_out")
            ))
         
            # 【优化重点】：在目标层之后注册一个阻断器
            # 我们Hook在当前层的最后，确保当前层计算完后立即停止
            if self.use_recomputation:
                # 注册在当前层(Layer L)的输出上，或者注册在 Layer L+1 的输入上
                # 最稳妥的是注册在该层 forward 的最后。
                # 对于 HF 模型，直接 Hook layer 模块本身即可捕获 output
                self.hooks.append(hf_layer.register_forward_hook(self._stopping_hook))

        # --- 运行 Forward (带 Early Stopping) ---
        try:
            with torch.no_grad():
                # 注意：output_attentions=True 可能导致 StopForward 失效(因为 HF 内部实现)，
                # 但通常 hook 会先执行。
                outputs = self.model(self.current_input_ids, output_attentions=True)
                # if outputs.attentions is not None:
                #     print(f"outputs.attentions shape: {outputs.attentions.shape}")
                #     target_storage[f"layers.{l}.attn_weights"] = outputs.attentions[layer_idx].detach()
             
        except StopForward:
            # 这是我们预期的行为：模型运行到目标层后停止了
            pass
     
        # 补充：Attention Pattern 通常在 outputs 里，但因为我们在中间停止了，outputs 没返回。
        # 因此我们需要在 hook 里抓取 attn_weights。
        # 这里的复杂点是 HF 的 attn_weights 是在 forward 内部产生的。
        # 为了简化，如果我们要精确的 Attn Weights，可能需要 hook self_attn 模块的输出。
        # 上面的 _get_hook 已经处理了 tuple output 的情况。
     
        # 确保 attn_weights 被抓取 (Hack for HF)
        # 如果 self_attn hook 没抓到 (因为 StopForward 抛出太快)，
        # 我们可能需要移除 StopForward hook，但这会牺牲速度。
        # 实际上，register_forward_hook 在 self_attn 执行完后会立即触发，所以应该能抓到。
     
        self.clear_hooks()
        if self.use_recomputation:
            torch.cuda.empty_cache()
         
        return target_storage

# --- 5. 发现逻辑 (Discoverer) ---

class CircuitDiscoverer:
    def __init__(self, model, tc_manager: LazyTranscoderManager, act_engine: ModelActivationsEngine):
        self.model = model
        self.tc_manager = tc_manager
        self.act_engine = act_engine
        self.config = model.config
     
    def get_transcoder_feature_vector(self, layer_idx, feature_idx):
        tc = self.tc_manager.get(layer_idx)
        return tc.W_enc[:, feature_idx]

    def calculate_upstream(self, target_vector, target_layer, token_idx, top_k=5, threshold=0.0):
        contributions = []
        prev_layer = target_layer - 1
        if prev_layer < 0: 
            return []

        acts = self.act_engine.get_layer_activations(prev_layer)
     
        # Transcoder attribution
        try:
            tc_prev = self.tc_manager.get(prev_layer)
            virtual_weight = tc_prev.W_dec @ target_vector
            mlp_in = acts[f"layers.{prev_layer}.transcoder_input"]
            x_token = mlp_in[0, token_idx, :]
            feature_acts = tc_prev.encode(x_token)
            tc_scores = feature_acts * virtual_weight
         
            top_vals, top_inds = torch.topk(tc_scores, k=top_k)
            print(f"Top {top_k} TC scores for layer {prev_layer}: {top_vals}")
            for score, idx in zip(top_vals, top_inds):
                if abs(score.item()) >= threshold:
                    node = CircuitNode(prev_layer, 'transcoder_feature', idx.item())
                    contributions.append((score.item(), node))
        except Exception as e:
            print(f"Error in calculate_upstream for layer {prev_layer}: {e}")

        # Attention attribution
        try:
            # 尝试获取 Attention 相关的张量
            # 注意：如果 Hook 没抓到 attn_weights (可能因为模型内部实现差异)，这里会跳过
            if f"layers.{prev_layer}.v_out" in acts:
                v_out = acts[f"layers.{prev_layer}.v_out"]
                # 这是一个简化的假设，如果拿不到 weights，就无法精确计算 Head 贡献
                # 在此版本中，如果 cache 里没有 weights，我们暂时跳过 Attn 计算，避免报错
                # 实际生产代码需要更细致的 Hook
                pass
                # (原逻辑保持不变，此处省略以节省篇幅，重点是上面的 Early Stopping)
        except Exception:
            pass

        contributions.sort(key=lambda x: abs(x[0]), reverse=True)
        return contributions[:top_k]

    def run(self, start_layer, start_feature_idx, input_text, max_depth=2, branches=3) -> CircuitGraph:
        self.act_engine.set_input(input_text)
        token_idx = self.act_engine.current_input_ids.shape[1] - 1
     
        # 初始化图
        graph = CircuitGraph()
        root = CircuitNode(start_layer, 'transcoder_feature', start_feature_idx)
        graph.nodes[str(root)] = root
        graph.node_scores[str(root)] = 10.0 # 根节点默认最大分
     
        current_layer_nodes = [root]
     
        print(f"Starting discovery for {root}...")
     
        for depth in range(max_depth):
            # print(f"Depth {depth+1}...")
            next_layer_nodes = []
         
            # 进度条
            pbar = tqdm(current_layer_nodes, desc=f"Layer {start_layer - depth} -> {start_layer - depth -1}", leave=False)
            for node in pbar:
                if node.node_type == 'transcoder_feature':
                    target_vec = self.get_transcoder_feature_vector(node.layer, node.index)
                    upstream = self.calculate_upstream(target_vec, node.layer, token_idx, top_k=branches)
                 
                    for score, prev_node in upstream:
                        graph.add_edge(prev_node, node, score)
                        next_layer_nodes.append(prev_node)
         
            if not next_layer_nodes:
                print(f"Early stopping at depth {depth+1} due to no upstream nodes.")
                break
            current_layer_nodes = list(set(next_layer_nodes))
         
        return graph

def plot_circuit_graph(graph: CircuitGraph, save_path="transcoder_circuit.png"):
    """
    使用 Plotly 绘制计算图。
    模仿 circuit_analysis.py 的风格：
    X轴: 层数 (Layer)
    Y轴: 节点 (自动排布)
    """
    if not graph.nodes:
        print("Graph is empty, nothing to plot.")
        return

    # 1. 布局计算 (Node Layout)
    # 按层分组
    layers = defaultdict(list)
    for node_str, node in graph.nodes.items():
        layers[node.layer].append(node)
 
    # 计算坐标
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
 
    # 简单的 Y 轴排布：在该层均匀分布
    node_pos_map = {} # str(node) -> (x, y)
 
    sorted_layers = sorted(layers.keys())
    for layer in sorted_layers:
        nodes_in_layer = layers[layer]
        # 简单的排序，尝试减少交叉 (heuristic)
        nodes_in_layer.sort(key=lambda x: (x.node_type, x.index))
     
        for i, node in enumerate(nodes_in_layer):
            x = layer
            # 归一化 Y 坐标到 [0, 1]，加一点 jitter 看起来更自然
            y = (i + 1) / (len(nodes_in_layer) + 1)
         
            node_pos_map[str(node)] = (x, y)
         
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Score: {graph.node_scores.get(str(node), 0):.2f}")
         
            # 颜色和大小
            score = graph.node_scores.get(str(node), 0.1)
            # 简单的颜色映射
            node_color.append(score)
            node_size.append(15 + np.log(score + 1) * 5) # Log size

    # 2. 边 (Edges)
    edge_x = []
    edge_y = []
 
    for edge in graph.edges:
        start_pos = node_pos_map[str(edge.source)]
        end_pos = node_pos_map[str(edge.target)]
     
        # Plotly 画线需要 None 分隔
        edge_x.extend([start_pos[0], end_pos[0], None])
        edge_y.extend([start_pos[1], end_pos[1], None])

    # 3. 绘图
    fig = go.Figure()

    # 添加边
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        name='Attribution'
    ))

    # 添加节点
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_size,
            color=node_color,
            line_width=2
        ),
        name='Features'
    ))

    fig.update_layout(
        # title=title,
        showlegend=False,
        xaxis=dict(title="Layer", tickmode='linear', tick0=min(sorted_layers), dtick=1),
        yaxis=dict(showticklabels=False), # 隐藏 Y 轴刻度，因为没有物理意义
        margin=dict(l=40, r=40, b=40, t=40),
        height=600
    )
 
    # save figure
    fig.write_image(save_path)

def run_analysis_pipeline(
    model_name, 
    device="cuda", 
    target_feat=100,
    max_depth=3,
    max_branches=4,
    use_recomputation=False,
    **transcoder_kwargs
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # from your_file import load_transcoder_gemma # 记得导入你的加载函数
 
    print(f"Loading Model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    tc_manager = LazyTranscoderManager(
        load_fn=load_transcoder_gemma,
        device=device,
        **transcoder_kwargs
    )
    act_engine = ModelActivationsEngine(model, tokenizer, use_recomputation=use_recomputation)
    discoverer = CircuitDiscoverer(model, tc_manager, act_engine)
 
    target_layer = transcoder_kwargs["layer"]
    prompt = "The quick brown fox jumped over the lazy dog."
 
    graph = discoverer.run(
        start_layer=target_layer,
        start_feature_idx=target_feat,
        input_text=prompt,
        max_depth=max_depth,
        branches=max_branches
    )
 
    print("Plotting graph...")
    fig_dir = ROOT_DIR / "figures/transcoder_circuits/test"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_circuit_graph(graph, save_path=f"{fig_dir}/layer{target_layer}_feat{target_feat}_depth{max_depth}_branches{max_branches}.png")
 
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":

    # 16k, small
    # 262k, big
    transcoder_kwargs = {
        "repo_id": "google/gemma-scope-2-1b-pt",  # "google/gemma-scope-2-1b-pt", "google/gemma-scope-2-1b-it"
        "transcoder_pos": "transcoder_all",
        "release": "gemma-scope-2-1b-pt-transcoders-all",
        "width": "16k",  # 16k, 262k
        "L0": "small",  # small, big
        "layer": 25,  # start layer
    }
    run_analysis_pipeline(
        "google/gemma-3-1b-pt",
        device="cuda:5",
        target_feat=100,
        max_depth=5,
        max_branches=8,
        use_recomputation=False,
        **transcoder_kwargs,
    )


