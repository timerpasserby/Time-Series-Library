import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.net(x) + residual


class Model(nn.Module):
    """
    Minimal TCN baseline for forecasting in the TSLib interface.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        kernel_size = max(2, configs.d_conv)
        channels = [configs.d_model] * max(1, configs.e_layers)

        blocks = []
        in_channels = configs.enc_in
        for level, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** level,
                    dropout=configs.dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(channels[-1], self.pred_len * self.c_out)

    def forecast(self, x_enc):
        features = self.network(x_enc.transpose(1, 2))
        last_step = features[:, :, -1]
        output = self.head(last_step)
        return output.view(x_enc.size(0), self.pred_len, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        return None
