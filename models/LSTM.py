import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Minimal LSTM baseline for forecasting in the TSLib interface.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        hidden_size = configs.d_model
        num_layers = configs.e_layers
        dropout = configs.dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, self.pred_len * self.c_out)

    def forecast(self, x_enc):
        _, (hidden, _) = self.encoder(x_enc)
        last_hidden = hidden[-1]
        output = self.head(last_hidden)
        return output.view(x_enc.size(0), self.pred_len, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        return None
