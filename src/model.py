"""Paper's projected VGG feature and GRU/LSTM caption encoder."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RetrievalModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_feature_dim: int = 4096,
        word_dim: int = 300,
        embedding_dim: int = 1024,
        recurrent_type: str = "gru",
        glove_matrix: np.ndarray | None = None,
    ):
        super().__init__()
        if recurrent_type not in {"gru", "lstm"}:
            raise ValueError("recurrent_type must be 'gru' or 'lstm'")
        self.recurrent_type = recurrent_type
        self.image_projection = nn.Linear(image_feature_dim, embedding_dim, bias=False)
        # Start the projected image vector near the recurrent state scale.
        nn.init.normal_(self.image_projection.weight, std=(image_feature_dim * embedding_dim) ** -0.5)
        self.word_embeddings = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        if glove_matrix is not None:
            if glove_matrix.shape != (vocab_size, word_dim):
                raise ValueError("GloVe matrix shape does not match vocabulary and word dimension")
            self.word_embeddings.weight.data.copy_(torch.from_numpy(glove_matrix))
        rnn_type = nn.GRU if recurrent_type == "gru" else nn.LSTM
        self.rnn = rnn_type(word_dim, embedding_dim, batch_first=True)

    def encode_images(self, features: torch.Tensor) -> torch.Tensor:
        return self.image_projection(features)

    def encode_captions(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.word_embeddings(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.rnn(packed)
        if self.recurrent_type == "lstm":
            state = state[0]
        return state[-1]

    def forward(self, features: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor):
        return self.encode_images(features), self.encode_captions(tokens, lengths)


def similarity_matrix(images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
    """S(i,c) = -||fi(i) - fc(c)||^2; rows are images, columns captions."""
    squared = images.square().sum(1, keepdim=True) + captions.square().sum(1)[None, :]
    squared = squared - 2 * images @ captions.T
    return -squared.clamp_min(0)
