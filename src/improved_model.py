"""Separate bidirectional caption encoder for the authorized improvement study.

This is an extension of the project, not a reproduction of the original paper.
The hardest negative term is inspired by Faghri et al., VSE++ (2018):
https://arxiv.org/abs/1707.05612
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BidirectionalRetrievalModel(nn.Module):
    def __init__(self, vocab_size: int, image_feature_dim: int = 4096,
                 word_dim: int = 300, embedding_dim: int = 1024,
                 hidden_dim: int = 512, dropout: float = 0.1,
                 glove_matrix: np.ndarray | None = None):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        if glove_matrix is not None:
            if glove_matrix.shape != (vocab_size, word_dim):
                raise ValueError("GloVe dimensions do not match this model")
            with torch.no_grad():
                self.word_embeddings.weight.copy_(torch.from_numpy(glove_matrix))
                self.word_embeddings.weight[0].zero_()
        self.word_dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(word_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Combine the full caption's mean states and both final directional states.
        self.caption_projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(4 * hidden_dim, embedding_dim))
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, embedding_dim), nn.LayerNorm(embedding_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_images(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_projection(features), dim=1)

    def encode_captions(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or len(lengths) != len(tokens):
            raise ValueError("Each caption must have one length")
        if bool((lengths < 1).any()) or bool((lengths > tokens.shape[1]).any()):
            raise ValueError("Caption lengths are outside the padded token array")
        embedded = self.word_dropout(self.word_embeddings(tokens))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, states = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        mean = output.sum(1) / lengths.to(output.device)[:, None]
        final = torch.cat((states[0], states[1]), dim=1)
        return F.normalize(self.caption_projection(torch.cat((mean, final), dim=1)), dim=1)

    def forward(self, features, tokens, lengths):
        return self.encode_images(features), self.encode_captions(tokens, lengths)


class ValidationEnsemble(nn.Module):
    """Combine two normalized encoders using a validation selected weight.

    Concatenation with square root weights makes squared distance equal to
    the weighted sum of the two original distances, in both retrieval directions.
    """
    def __init__(self, original: nn.Module, candidate: nn.Module, original_weight: float):
        super().__init__()
        if not 0 <= original_weight <= 1:
            raise ValueError("Ensemble weight must be between zero and one")
        self.original = original
        self.candidate = candidate
        self.original_scale = original_weight ** 0.5
        self.candidate_scale = (1 - original_weight) ** 0.5

    def encode_images(self, features):
        return torch.cat((self.original_scale * self.original.encode_images(features),
                          self.candidate_scale * self.candidate.encode_images(features)), dim=1)

    def encode_captions(self, tokens, lengths):
        return torch.cat((self.original_scale * self.original.encode_captions(tokens, lengths),
                          self.candidate_scale * self.candidate.encode_captions(tokens, lengths)), dim=1)

    def forward(self, features, tokens, lengths):
        return self.encode_images(features), self.encode_captions(tokens, lengths)


def contrastive_retrieval_loss(images: torch.Tensor, captions: torch.Tensor,
                               image_ids: torch.Tensor, temperature: float = 0.07,
                               margin: float = 0.2, hard_weight: float = 0.25) -> torch.Tensor:
    """Symmetric contrastive loss plus the hardest incorrect match per query.

    Other captions of the same image are positives, never false negatives.
    Both directions are averaged so the scale does not grow with batch size.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    scores = images @ captions.T
    positive = image_ids[:, None] == image_ids[None, :]
    logits = scores / temperature
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    contrastive = 0.5 * (
        (torch.logsumexp(logits, 1) - torch.logsumexp(positive_logits, 1)).mean()
        + (torch.logsumexp(logits, 0) - torch.logsumexp(positive_logits, 0)).mean())
    incorrect = ~positive
    if not bool(incorrect.any()) or hard_weight == 0:
        return contrastive
    matched = scores.diag()
    caption_cost = F.relu(margin + scores - matched[:, None]).masked_fill(positive, 0)
    image_cost = F.relu(margin + scores - matched[None, :]).masked_fill(positive, 0)
    hardest = 0.5 * (caption_cost.max(1).values.mean() + image_cost.max(0).values.mean())
    return contrastive + hard_weight * hardest
