"""Bidirectional in-batch margin ranking loss from Equation 2."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import similarity_matrix


def ranking_loss(
    image_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor,
    image_ids: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    scores = similarity_matrix(image_embeddings, caption_embeddings)
    positive = scores.diag()
    negatives = image_ids[:, None] != image_ids[None, :]
    if not negatives.any():
        return scores.sum() * 0
    wrong_caption = F.relu(margin + scores - positive[:, None])
    wrong_image = F.relu(margin + scores - positive[None, :])
    # Column j uses the positive score for caption j; rows are incorrect images.
    return (wrong_caption[negatives].mean() + wrong_image[negatives].mean())
