import plotly.express as px
import plotly.graph_objects as go

def format_plotly_figure(fig: go.Figure, ) -> go.Figure:
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