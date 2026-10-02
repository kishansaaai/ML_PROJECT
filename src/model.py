"""Paper's projected VGG feature and GRU/LSTM caption encoder."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



class RetrievalModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_feature_dim: int = 4096,
        word_dim: int = 300,
        embedding_dim: int = 1024,
        recurrent_type: str = "gru",
        glove_matrix: np.ndarray | None = None,
        normalize_embeddings: bool = False,
        projection_bias: bool = False,
        recurrent_implementation: str = "torch",
        caption_padding_length: int = 0,
    ):
        super().__init__()
        if recurrent_type not in {"gru", "lstm"}:
            raise ValueError("recurrent_type must be 'gru' or 'lstm'")
        self.recurrent_type = recurrent_type
        self.normalize_embeddings = normalize_embeddings
        self.caption_padding_length = caption_padding_length
        self.recurrent_implementation = recurrent_implementation
        self.image_projection = nn.Linear(image_feature_dim, embedding_dim, bias=projection_bias)
        # Start the projected image vector near the recurrent state scale.
        if recurrent_implementation == "keras_legacy":
            nn.init.xavier_uniform_(self.image_projection.weight)
            if self.image_projection.bias is not None:
                nn.init.zeros_(self.image_projection.bias)
        else:
            nn.init.normal_(self.image_projection.weight, std=(image_feature_dim * embedding_dim) ** -0.5)
        self.word_embeddings = nn.Embedding(vocab_size, word_dim,
                                           padding_idx=None if recurrent_implementation == "keras_legacy" else 0)
        if recurrent_implementation == "keras_legacy":
            nn.init.uniform_(self.word_embeddings.weight, -0.05, 0.05)
        if glove_matrix is not None:
            if glove_matrix.shape != (vocab_size, word_dim):
                raise ValueError("GloVe matrix shape does not match vocabulary and word dimension")
            self.word_embeddings.weight.data.copy_(torch.from_numpy(glove_matrix))
        if recurrent_implementation == "keras_legacy":
            from .legacy_recurrent import LegacyRecurrent
            self.rnn = LegacyRecurrent(word_dim, embedding_dim, recurrent_type)
        elif recurrent_implementation == "torch":
            rnn_type = nn.GRU if recurrent_type == "gru" else nn.LSTM
            self.rnn = rnn_type(word_dim, embedding_dim, batch_first=True)
        else:
            raise ValueError("Unsupported recurrent implementation")

    def encode_images(self, features: torch.Tensor) -> torch.Tensor:
        result = self.image_projection(features)
        return F.normalize(result, dim=1) if self.normalize_embeddings else result

    def encode_captions(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.recurrent_implementation == "keras_legacy":
            if tokens.shape[1] > self.caption_padding_length:
                raise ValueError("Caption exceeds the reference padding length")
            tokens = F.pad(tokens, (0, self.caption_padding_length - tokens.shape[1]))
            result = self.rnn(self.word_embeddings(tokens))
            return F.normalize(result, dim=1) if self.normalize_embeddings else result
        embedded = self.word_embeddings(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.rnn(packed)
        if self.recurrent_type == "lstm":
            state = state[0]
        result = state[-1]
        return F.normalize(result, dim=1) if self.normalize_embeddings else result

    def forward(self, features: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor):
        return self.encode_images(features), self.encode_captions(tokens, lengths)


def similarity_matrix(images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
    """S(i,c) = -||fi(i) - fc(c)||^2; rows are images, columns captions."""
    squared = images.square().sum(1, keepdim=True) + captions.square().sum(1)[None, :]
    squared = squared - 2 * images @ captions.T
    return -squared.clamp_min(0)
