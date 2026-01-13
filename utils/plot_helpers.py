import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

# pip install playwright
# playwright install
from playwright.sync_api import sync_playwright


def format_plotly_figure(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        # title="",
        xaxis_title="SAE feature IDs",
        yaxis_title="Activation values",
        margin=dict(
            l=5,  # 左边距 (Left)
            r=5,  # 右边距 (Right)
            b=5,  # 下边距 (Bottom)
            t=5,  # 上边距 (Top)
            pad=4  # 内部填充 (Padding)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.update_xaxes(
        showline=False, 
        showgrid=True, 
        gridcolor='lightgrey', 
        gridwidth=1, 
        zeroline=True, 
        zerolinewidth=1, 
        zerolinecolor='black', 
        autorange=True, 
        anchor='y', 
        side='bottom', 
        ticks='outside', 
        ticklen=5, 
        ticklabelposition='outside'
    )
    fig.update_yaxes(
        showline=False, 
        showgrid=True, 
        gridcolor='lightgrey', 
        gridwidth=1, 
        zeroline=True, 
        zerolinewidth=1, 
        zerolinecolor='black', 
        autorange=True, 
        anchor='x', 
        side='left', 
        ticks='outside', 
        ticklen=5, 
        ticklabelposition='outside'
    )
    return fig

def save_activations_to_pdf(str_toks, activations, filename="output.pdf"):

    norm_acts = np.array(activations) / (1e-6 + np.max(activations))
    
    fig, ax = plt.subplots(figsize=(len(str_toks) * 0.8, 1)) # 根据长度调整尺寸
    ax.set_axis_off()
    
    fig.patch.set_facecolor("#fdfdfd")
    
    x_pos = 0
    for t, v in zip(str_toks, norm_acts):
        # 绘制背景色矩形 (RGBA)
        t_obj = ax.text(
            x_pos, 
            0.5, 
            t, 
            color='white', 
            fontsize=12,
            va='center', ha='left',
            bbox=dict(facecolor=(1, 0, 0, v), edgecolor='none', pad=2),
        )
        
        # 粗略估计下一个单词的位置
        # 注意：这里可能需要根据实际字体微调
        x_pos += len(t) * 0.05 + 0.02 

    plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.show()

def html_activations(str_toks: list[str], activations: list[float]):
    return "".join(
        f'<span style="background-color: rgba(255,0,0,{v}); padding: 4px 0px;">{t}</span>'
        for t, v in zip(str_toks, np.array(activations) / (1e-6 + np.max(activations)), strict=True)
    )

def save_html_to_pdf(html_str, output_path):

    full_html = f"""
    <html>
    <body style="margin: 0; padding: 0; background-color: #ffffff;">
        <div id="viz-container" style="
            display: inline-block; 
            white-space: nowrap; 
            font-family: monospace; 
            font-size: 20px; 
            color: black; 
            padding: 10px;">
            {html_str}
        </div>
    </body>
    </html>
    """

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(full_html)

        dimensions = page.evaluate("""() => {
            const el = document.getElementById('viz-container');
            const rect = el.getBoundingClientRect();
            return {
                width: rect.width,
                height: rect.height
            };
        }""")
        
        # 导出为 PDF (支持背景色、页边距等配置)
        page.pdf(
            path=output_path,
            width=f"{dimensions['width']}px",
            height=f"{dimensions['height']}px",
            print_background=True,
            margin={
                "top": "0px",
                "right": "0px",
                "bottom": "0px",
                "left": "0px"
            },
        )
        
        browser.close()
        print(f"PDF saved to {output_path} with dimensions {dimensions}")

def generate_token_activation_map(str_toks: list[str], activations: list[float], output_path: str):
    html_str = html_activations(str_toks, activations)
    save_html_to_pdf(html_str, output_path)

def generate_multi_token_activation_maps(all_data, output_path="combined_activations.pdf"):
    """
    all_data: list of dicts, each dict has {'tokens': [...], 'activations': [...]}
    """
    
    # 1. 构造所有行的 HTML 碎片
    rows_html = ""
    for idx, entry in enumerate(all_data):
        str_toks = entry['tokens']
        acts = entry['activations']
        
        # 归一化当前行的激活值
        norm_acts = np.array(acts) / (1e-6 + np.max(acts))
        
        # 构造这一行的 spans
        # margin-right: -0.2px 用于消除背景缝隙；white-space: pre 确保空格被正确渲染
        spans = "".join(
            f'<span style="background-color: rgba(255,0,0,{v}); padding: 4px 0px; margin-right: -0.2px; white-space: pre;">{t}</span>'
            for t, v in zip(str_toks, norm_acts)
        )
        
        # 将每一行放入一个 div，设置下边距隔离
        if idx == len(all_data) - 1:
            margin_bottom = 0
        else:
            margin_bottom = 20
        rows_html += f'<div class="prompt-row" style="margin-bottom: {margin_bottom}px;">{spans}</div>'

    # 2. 构造完整的 HTML 框架
    full_html = f"""
    <html>
    <body style="margin: 0; padding: 0; background-color: #ffffff;">
        <div id="viz-container" style="
            display: inline-block; 
            padding: 10px;
            color: black;
            font-family: monospace;
            font-size: 20px;
            background-color: #ffffff;">
            {rows_html}
        </div>
    </body>
    </html>
    """

    # 3. 使用 Playwright 渲染并导出
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(full_html)
        
        # 获取整体容器的尺寸
        dimensions = page.evaluate("""() => {
            const el = document.getElementById('viz-container');
            const rect = el.getBoundingClientRect();
            return { width: rect.width, height: rect.height };
        }""")
        
        page.pdf(
            path=output_path,
            width=f"{dimensions['width']}px",
            height=f"{dimensions['height']}px",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}
        )
        browser.close()
        print(f"PDF saved to {output_path} with dimensions {dimensions}")



