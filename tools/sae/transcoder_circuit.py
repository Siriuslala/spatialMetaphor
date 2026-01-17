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


class StopForward(Exception):
    """用于在获取到所需激活值后强制停止模型前向传播，节省计算资源"""
    pass


@dataclass(frozen=True)
class CircuitNode:
    layer: int
    node_type: str # 'transcoder_feature', 'attn_head'
    index: int
    src_token: Optional[int] = None  # 用于 Attn Head 指明源 Token
 
    def __repr__(self):
        if self.node_type == 'transcoder_feature':
            return f"L{self.layer}.TC.{self.index}"
        elif self.node_type == 'attn_head':
            token_str = f".T{self.src_token}" if self.src_token is not None else ""
            return f"L{self.layer}.A{self.index}{token_str}"
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

    def _basic_hook(self, module, input, output, cache, key, save_input=False):
        
        if save_input:
            acts = input[0]
        else:
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
    
    def _attn_weights_hook(self, module, input, output, cache, key):
        # output[1] 通常是 attn_weights (batch, heads, q_len, k_len)
        # 注意：必须确保模型推理时 output_attentions=True
        if len(output) > 1 and output[1] is not None:
            cache[key] = output[1].detach()

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

            # get attention input and output attentions
            self.hooks.append(hf_layer.input_layernorm.register_forward_hook(
                partial(self._basic_hook, cache=target_storage, key=f"layers.{l}.attn_input_residual", save_input=True)
            ))
            self.hooks.append(hf_layer.input_layernorm.register_forward_hook(
                partial(self._basic_hook, cache=target_storage, key=f"layers.{l}.attn_input", save_input=False)
            ))
            self.hooks.append(hf_layer.self_attn.register_forward_hook(
                partial(self._attn_weights_hook, cache=target_storage, key=f"layers.{l}.attn_scores")
            ))

            # get transcoder input
            self.hooks.append(hf_layer.pre_feedforward_layernorm.register_forward_hook(
                partial(self._basic_hook, cache=target_storage, key=f"layers.{l}.transcoder_input_residual", save_input=True)
            ))
            self.hooks.append(hf_layer.pre_feedforward_layernorm.register_forward_hook(
                partial(self._basic_hook, cache=target_storage, key=f"layers.{l}.transcoder_input", save_input=False)
            ))
         
            if self.use_recomputation:
                self.hooks.append(hf_layer.register_forward_hook(self._stopping_hook))

        # Forward
        try:
            with torch.no_grad():
                self.model.set_attn_implementation('eager')
                self.model(self.current_input_ids, output_attentions=True)
        except StopForward:
            pass
     
        self.clear_hooks()
        if self.use_recomputation:
            torch.cuda.empty_cache()
         
        return target_storage


class CircuitDiscoverer:
    def __init__(self, model, tc_manager: LazyTranscoderManager, act_engine: ModelActivationsEngine):
        self.model = model
        self.tc_manager = tc_manager
        self.act_engine = act_engine
        self.config = model.config
     
    def get_transcoder_feature_vector(self, layer_idx, feature_idx):
        # W_enc: (d_model, d_sae)
        tc = self.tc_manager.get(layer_idx)
        return tc.W_enc[:, feature_idx]

    def calculate_upstream(self, target_vector, target_layer, token_idx, top_k=5, threshold=0.0):
        """
        We want to compute the upstream contributions in layer l(l'-1) to the target feature in the target layer l'.
        Params:
            target_vector: the feature vector of the target feature
            target_layer: l'
            token_idx: int
            top_k: int
            threshold: float
        """
        contributions = []
        next_vectors = []  # pass to the next layer

        prev_layer = target_layer - 1
        if prev_layer < 0: 
            return [], []

        acts = self.act_engine.get_layer_activations(prev_layer)

        ln_in = acts[f"layers.{prev_layer}.transcoder_input_residual"][0, token_idx, :]
        ln_out = acts[f"layers.{prev_layer}.transcoder_input"][0, token_idx, :]
        scale = ln_out.norm() / (ln_in.norm() + 1e-6)
     
        # Transcoder attribution
        try:
            tc_prev = self.tc_manager.get(prev_layer)
            gradient = tc_prev.W_dec @ target_vector  # (d_sae, d_out) * (d_out,) -> (d_sae,)

            mlp_in = acts[f"layers.{prev_layer}.transcoder_input"]
            x_token = mlp_in[0, token_idx, :]
            with torch.no_grad():
                feature_acts = tc_prev.encode(x_token)

            tc_scores = feature_acts * gradient
         
            top_vals, top_inds = torch.topk(tc_scores, k=top_k)

            # print(f"Top {top_k} TC scores for layer {prev_layer}: {top_vals.tolist()}")
            for score, idx in zip(top_vals, top_inds):
                if abs(score.item()) >= threshold:
                    node = CircuitNode(prev_layer, 'transcoder_feature', idx.item())
                    contributions.append((score.item(), node))

                    # Next Vector = W_enc_prev[idx] * gradient[idx] * scale
                    W_enc_i = tc_prev.W_enc[:, idx.item()]
                    # grad_i = gradient[idx.item()]
                    # next_vector = W_enc_i * grad_i * scale
                    next_vector = W_enc_i * score.item() * scale
                    next_vectors.append(next_vector)
        except Exception as e:
            print(f"Error in calculate_upstream for layer {prev_layer}: {e}")
            return [], []

        # Attention attribution
        try:
            attn_scores = acts.get(f"layers.{prev_layer}.attn_scores")  # [1, n_heads, seq_len, seq_len]
            
            if attn_scores is not None:
                
                attn_layer = self.model.model.layers[prev_layer].self_attn
                W_V = attn_layer.v_proj.weight.to(target_vector.dtype)  # [num_kv_heads * head_dim, d_model]
                W_O = attn_layer.o_proj.weight.to(target_vector.dtype)  # [d_model, num_heads * head_dim]
                # # W_V shape: torch.Size([256, 1152]), W_O shape: torch.Size([1152, 1024])
                # print(f"W_V shape: {W_V.shape}, W_O shape: {W_O.shape}")
                
                num_heads = self.config.num_attention_heads
                num_kv_heads = self.config.num_key_value_heads
                head_dim = self.config.head_dim
                groups = num_heads // num_kv_heads
                # # num_heads: 4, num_kv_heads: 1, head_dim: 256, groups: 4
                # print(f"num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, head_dim: {head_dim}, groups: {groups}")
                
                # 注意：Attn 涉及多个源 Token，每个 Token 的 Scale 不同
                # 形状: [seq_len, 1]
                ln_in_attn = acts[f"layers.{prev_layer}.attn_input_residual"][0].to(target_vector.dtype) # [seq, d_model]
                ln_out_attn = acts[f"layers.{prev_layer}.attn_input"][0].to(target_vector.dtype)  # [seq, d_model]
                scale_attn = (ln_out_attn.norm(dim=-1) / (ln_in_attn.norm(dim=-1) + 1e-6))  # [seq]
                
                # 2. 计算 OV Circuit 的投影
                # 对于每个 Head h:
                # Out_h = (x @ W_V_h) @ W_O_h
                # Contribution = (Out_h) @ target_vector
                #              = x @ W_V_h @ W_O_h @ target_vector
                # 我们可以先计算: virtual_grad = W_O @ target_vector -> [n_heads * d_head]
                
                # [n_heads * d_head]
                grad_at_heads = torch.matmul(W_O.t(), target_vector).view(num_heads, head_dim)
                # # target_vector shape: torch.Size([1152])
                # print(f"target_vector shape: {target_vector.shape}")

                # 3. 遍历所有 Heads 计算贡献 
                head_token_contribs = []
                
                for h in range(num_heads):
                    kv_h = h // groups
                    W_V_h = W_V[kv_h * head_dim : (kv_h + 1) * head_dim, :]

                    v_values = torch.matmul(ln_out_attn, W_V_h.t())

                    # 该 Head 在残差流中的投影方向: direction = W_V_h.T @ grad_at_head_h
                    ov_direction = torch.matmul(W_V_h.t(), grad_at_heads[h])  # [d_model]

                    # 计算所有 source tokens 对 dst_token 的归因
                    # pattern: [seq_len]
                    pattern = attn_scores[0, h, token_idx, :].to(target_vector.dtype)
                    
                    # 归因 = Pattern * (x_pre @ ov_direction)
                    # 这里的 x_pre 包含了 V 投影前的所有信息
                    token_attribs = pattern * torch.matmul(v_values, grad_at_heads[h])
                    for s_idx in range(len(token_attribs)):
                        score = token_attribs[s_idx].item()
                        if abs(score) >= threshold:
                            # 保存 (分数, head_idx, source_token_idx, pullback_vector)
                            # 注意：pullback_vector 需要乘以该位置的 LN scale
                            next_vec = ov_direction * pattern[s_idx] * scale_attn[s_idx].item()
                            head_token_contribs.append((score, h, s_idx, next_vec))
                    
                # 排序并取 Top-K
                head_token_contribs.sort(key=lambda x: abs(x[0]), reverse=True)
                for score, h_idx, s_idx, vec in head_token_contribs[:top_k]:
                    node = CircuitNode(prev_layer, 'attn_head', h_idx, src_token=s_idx)
                    contributions.append((score, node))
                    next_vectors.append(vec)
                    
        except Exception as e:
             import traceback
             traceback.print_exc()
             print(f"[Warning] Attention calcs failed for L{prev_layer}: {e}")

        return contributions, next_vectors

    def run(self, start_layer, start_feature_idx, input_text, max_depth=2, branches=3) -> CircuitGraph:
        self.act_engine.set_input(input_text)
        token_idx = self.act_engine.current_input_ids.shape[1] - 1
     
        # 初始化图
        graph = CircuitGraph()
        root = CircuitNode(start_layer, 'transcoder_feature', start_feature_idx)
        root_vector = self.get_transcoder_feature_vector(start_layer, start_feature_idx)
        
        root_acts = self.act_engine.get_layer_activations(start_layer)
        ln_in = root_acts[f"layers.{start_layer}.transcoder_input_residual"][0, token_idx, :]
        ln_out = root_acts[f"layers.{start_layer}.transcoder_input"][0, token_idx, :]
        scale = ln_out.norm() / (ln_in.norm() + 1e-6)
        root_vector = root_vector * scale

        graph.nodes[str(root)] = root
        graph.node_scores[str(root)] = 10.0 # 根节点默认最大分
     
        current_layer_nodes = [(root, root_vector)]
     
        print(f"Starting discovery for {root}...")
     
        for depth in range(max_depth):
            # print(f"Depth {depth+1}...")
            next_layer_dict = {}  # {node_str: (node, vector)}
         
            pbar = tqdm(current_layer_nodes, desc=f"Layer {start_layer - depth} -> {start_layer - depth -1}", leave=False)
            for node, target_vec in pbar:
                if node.node_type == 'transcoder_feature':
                    upstream_contribs, upstream_vectors = self.calculate_upstream(
                        target_vec, node.layer, token_idx, top_k=branches
                    )
                 
                    for (score, prev_node), next_vec in zip(upstream_contribs, upstream_vectors):
                        graph.add_edge(prev_node, node, score)
                        k = str(prev_node)
                        if k not in next_layer_dict:
                            next_layer_dict[k] = (prev_node, next_vec)
                        else:
                            existing_node, existing_vec = next_layer_dict[k]
                            next_layer_dict[k] = (existing_node, existing_vec + next_vec)
         
            if not next_layer_dict:
                print(f"Early stopping at depth {depth+1} due to no upstream nodes.")
                break

            current_layer_nodes = list(next_layer_dict.values())
         
        return graph

def print_circuit(graph: CircuitGraph):
    """
    仿照作者风格，打印计算图中所有显著的路径连接
    """
    print("\n" + "="*60)
    print(f"{'TRANSCODER CIRCUIT ANALYSIS REPORT':^60}")
    print("="*60)
    
    # 将边按目标节点的层数从高到低排序（从输出往输入看）
    sorted_edges = sorted(graph.edges, key=lambda e: (e.target.layer, e.source.layer), reverse=True)
    
    current_target = None
    for edge in sorted_edges:
        if str(edge.target) != current_target:
            current_target = str(edge.target)
            print(f"\n[Target Node]: {edge.target}")
            print(f"  {'Contribution':<15} | {'Source Node':<20}")
            print(f"  {'-'*15}-+-{'-'*20}")
        
        # 打印贡献分数和来源节点
        print(f"  {edge.score:>+15.4f} | {edge.source}")

    print("\n" + "="*60)

import plotly.graph_objects as go
import numpy as np
from collections import defaultdict

def plot_circuit_graph(graph: CircuitGraph, save_path="circuit_viz.png"):
    """
    改进版绘图函数：自适应大小、清晰连线、改进配色、解决重叠。
    """
    if not graph.nodes:
        print("Graph is empty, nothing to plot.")
        return

    # --- 1. 数据准备与自适应尺度计算 ---
    layers = sorted(list(set(n.layer for n in graph.nodes.values())))
    if not layers: return

    # 统计每一层的节点数，找出最大值用于计算画布高度
    nodes_per_layer = defaultdict(list)
    max_nodes_in_a_layer = 0
    for node in graph.nodes.values():
        nodes_per_layer[node.layer].append(node)
    
    for layer in layers:
        # 先按类型排，再按索引排，保证布局整齐
        nodes_per_layer[layer].sort(key=lambda x: (x.node_type, x.index))
        max_nodes_in_a_layer = max(max_nodes_in_a_layer, len(nodes_per_layer[layer]))

    # 【核心改进：自适应画布大小】
    # 高度：保证每个节点至少有 60px 的垂直空间，最低 700px
    dynamic_height = max(700, max_nodes_in_a_layer * 60 + 200) 
    # 宽度：保证每层之间有足够的水平空间
    layer_span = max(layers) - min(layers) + 1
    dynamic_width = max(900, layer_span * 180)

    # 计算节点坐标
    pos = {} # node_str -> (x, y)
    node_list = [] # 用于后续批量添加 Trace

    for layer in layers:
        layer_nodes = nodes_per_layer[layer]
        n_nodes = len(layer_nodes)
        for i, node in enumerate(layer_nodes):
            # X轴：层数
            # Y轴：在 [0, 1] 区间内均匀分布。由于画布高度自适应，物理间隔会拉大。
            # 使用 (i + 0.5) / n_nodes 让节点居中分布
            y_pos = (i + 0.5) / n_nodes
            pos[str(node)] = (layer, y_pos)
            node_list.append(node)

    fig = go.Figure()

    # --- 2. 绘制连线 (Edges) ---
    # 【核心改进：提高可见性】
    max_score = max([abs(e.score) for e in graph.edges]) if graph.edges else 1.0
    
    for edge in graph.edges:
        x0, y0 = pos[str(edge.source)]
        x1, y1 = pos[str(edge.target)]
        
        abs_score = abs(edge.score)
        # 归一化分数用于计算线宽和透明度
        norm_score = abs_score / (max_score + 1e-6)
        
        # 线宽：基础 1.5px，最粗 6px
        width = 1.5 + norm_score * 4.5
        # 透明度：基础 0.4 (保证可见)，最高 0.9 (深色)
        opacity = 0.4 + norm_score * 0.5
        line_color = f'rgba(80, 80, 80, {opacity:.2f})'

        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=line_color),
            hoverinfo='text',
            hovertext=f"Score: {edge.score:.4f}<br>From: {edge.source}<br>To: {edge.target}",
            showlegend=False
        ))

    # --- 3. 绘制节点 (Nodes) ---
    node_x, node_y, node_text, node_labels, node_color, node_size = [], [], [], [], [], []
    
    for node in node_list:
        x, y = pos[str(node)]
        node_x.append(x)
        node_y.append(y)
        
        # 节点标签
        type_prefix = "TC" if node.node_type == 'transcoder_feature' else "Attn"
        token_suffix = f"@T{node.src_token}" if node.src_token is not None else ""
        label = f"{type_prefix}[{node.index}]{token_suffix}"
        node_labels.append(label)
        
        # 悬浮文本
        total_score = graph.node_scores.get(str(node), 0)
        node_text.append(f"<b>{label}</b><br>Layer: {node.layer}<br>Total Attribution: {total_score:.4f}")
        
        # 【核心改进：配色】
        # TC用青色(Teal)，Attention用鲜明的皇家蓝(RoyalBlue)
        c = 'teal' if node.node_type == 'transcoder_feature' else 'royalblue'
        node_color.append(c)
        
        # 节点大小随总贡献度变化
        s = 20 + np.log(total_score + 1) * 10
        node_size.append(s)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        textfont=dict(size=11, color='black'),
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white'), # 加个白边让节点更突出
            opacity=0.9
        ),
        name='Nodes',
        showlegend=False
    ))

    # --- 4. 布局设置 ---
    fig.update_layout(
        title=dict(text=f"Transcoder Circuit Graph (Adaptive Layout)", x=0.5, font=dict(size=20)),
        xaxis=dict(
            title="Model Layer (Input ← → Output)", 
            tickmode='array', 
            tickvals=layers, 
            gridcolor='rgba(200,200,200,0.5)',
            zeroline=False
        ),
        yaxis=dict(
            showticklabels=False, # 隐藏Y轴刻度，因为是相对位置
            showgrid=False, 
            zeroline=False,
            range=[-0.05, 1.05] # 稍微留点边距
        ),
        plot_bgcolor='rgba(250, 250, 250, 1)', # 极淡的灰色背景
        # 【应用自适应宽高】
        height=dynamic_height,
        width=dynamic_width,
        margin=dict(l=60, r=60, b=80, t=100)
    )
    
    fig.write_image(save_path)
    print(f"Improved circuit visualization saved to: {save_path}")
    print(f"Canvas size: {dynamic_width}x{dynamic_height} (Adaptive based on {len(layers)} layers, max {max_nodes_in_a_layer} nodes/layer)")

def run_analysis_pipeline(
    model_name, 
    device="cuda", 
    prompt="The quick brown fox jumped over the lazy dog.",
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
 
    graph = discoverer.run(
        start_layer=target_layer,
        start_feature_idx=target_feat,
        input_text=prompt,
        max_depth=max_depth,
        branches=max_branches
    )

    print_circuit(graph)

    print("Plotting graph...")
    fig_dir = ROOT_DIR / "figures/transcoder_circuits/test"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_circuit_graph(graph, save_path=f"{fig_dir}/layer{target_layer}_feat{target_feat}_depth{max_depth}_branches{max_branches}.pdf")
 
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
        target_feat=101,
        max_depth=3,
        max_branches=6,
        use_recomputation=False,
        **transcoder_kwargs,
    )


