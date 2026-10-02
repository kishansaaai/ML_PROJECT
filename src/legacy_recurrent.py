"""GRU and LSTM equations used by the authors' legacy Keras implementation.

The legacy GRU applies its reset gate before the recurrent matrix product.
Both cells use hard sigmoid recurrent gates. No attention or extra layers.
"""
from __future__ import annotations

import torch
from torch import nn


@torch.jit.script
def gru_state(inputs: torch.Tensor, recurrent: torch.Tensor, hidden: torch.Tensor):
    size = hidden.size(1)
    for step in range(inputs.size(1)):
        value = inputs[:, step]
        gate_values = hidden @ recurrent[:, :2 * size]
        update = (0.2 * (value[:, :size] + gate_values[:, :size]) + 0.5).clamp(0, 1)
        reset = (0.2 * (value[:, size:2 * size] + gate_values[:, size:]) + 0.5).clamp(0, 1)
        candidate = torch.tanh(value[:, 2 * size:] + (reset * hidden) @ recurrent[:, 2 * size:])
        hidden = update * hidden + (1 - update) * candidate
    return hidden


@torch.jit.script
def lstm_state(inputs: torch.Tensor, recurrent: torch.Tensor, hidden: torch.Tensor):
    cell = torch.zeros_like(hidden)
    size = hidden.size(1)
    for step in range(inputs.size(1)):
        value = inputs[:, step] + hidden @ recurrent
        incoming = (0.2 * value[:, :size] + 0.5).clamp(0, 1)
        forget = (0.2 * value[:, size:2 * size] + 0.5).clamp(0, 1)
        candidate = torch.tanh(value[:, 2 * size:3 * size])
        outgoing = (0.2 * value[:, 3 * size:] + 0.5).clamp(0, 1)
        cell = forget * cell + incoming * candidate
        hidden = outgoing * torch.tanh(cell)
    return hidden


class LegacyRecurrent(nn.Module):
    def __init__(self, word_dim: int, hidden_dim: int, kind: str):
        super().__init__()
        self.kind = kind
        self.hidden_dim = hidden_dim
        gates = 3 if kind == "gru" else 4
        self.kernel = nn.Parameter(torch.empty(word_dim, gates * hidden_dim))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_dim, gates * hidden_dim))
        self.bias = nn.Parameter(torch.zeros(gates * hidden_dim))
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        if kind == "lstm":
            with torch.no_grad():
                self.bias[hidden_dim:2 * hidden_dim].fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = inputs @ self.kernel + self.bias
        hidden = inputs.new_zeros((inputs.shape[0], self.hidden_dim))
        if self.kind == "gru":
            return gru_state(projected, self.recurrent_kernel, hidden)
        return lstm_state(projected, self.recurrent_kernel, hidden)
