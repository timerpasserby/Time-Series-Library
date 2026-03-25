import torch
import torch.nn as nn
import torch.nn.functional as F


class STGCNBlock(nn.Module):
    def __init__(self, hidden_dim, num_nodes, kernel_size, dropout):
        super().__init__()
        padding = kernel_size // 2
        self.temporal1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), padding=(0, padding))
        self.temporal2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), padding=(0, padding))
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def graph_conv(self, x):
        adjacency = torch.softmax(F.relu(self.adj), dim=-1)
        return torch.einsum('ij,bdjt->bdit', adjacency, x)

    def forward(self, x):
        residual = x
        x = F.gelu(self.temporal1(x))
        x = self.graph_conv(x)
        x = F.gelu(self.temporal2(x))
        x = self.dropout(self.norm(x))
        return x + residual


class Model(nn.Module):
    """
    Minimal learnable-graph STGCN baseline for forecasting in the TSLib interface.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.num_nodes = configs.enc_in
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        hidden_dim = configs.d_model
        kernel_size = max(3, configs.d_conv if configs.d_conv % 2 == 1 else configs.d_conv + 1)

        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=(1, 1))
        self.blocks = nn.ModuleList(
            [STGCNBlock(hidden_dim, self.num_nodes, kernel_size, configs.dropout) for _ in range(max(1, configs.e_layers))]
        )
        self.node_head = nn.Linear(hidden_dim, self.pred_len)
        self.channel_head = nn.Linear(self.num_nodes, self.c_out)

    def forecast(self, x_enc):
        x = x_enc.permute(0, 2, 1).unsqueeze(1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        last_step = x[:, :, :, -1].permute(0, 2, 1)
        output = self.node_head(last_step).permute(0, 2, 1)
        return self.channel_head(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        return None
