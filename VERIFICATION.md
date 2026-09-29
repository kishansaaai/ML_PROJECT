# Verification run (small COCO subset)

This run verifies execution with real COCO 2014 images and captions. It is **not** the 1,000-image experiment in the paper and does not estimate full COCO retrieval quality.

- Date: 29 September 2026
- Karpathy split: 20 train, 20 validation, 20 test images, each with five captions
- Image representation: ImageNet-pretrained VGG-19 `fc7`, ten crops averaged, 4096 dimensions
- Text representation: Stanford GloVe 6B 300-dimensional vectors for baseline and pretrained neural variants; 133 of 135 vocabulary entries found, including two special tokens
- Neural training: up to 10 epochs, batch size 128, Adam, learning rate 0.00001 for stability on this tiny subset; validation-based best checkpoint and early stopping
- Test gallery: 20 images. Normal methods use 100 caption queries. Baseline + Weight pools five held-out captions into 20 queries.

| Method | R@1 | R@10 | Mean rank |
| --- | ---: | ---: | ---: |
| Baseline | 7.00% | 56.00% | 9.63 |
| Baseline + Weight | 10.00% | 50.00% | 9.50 |
| GRU | 6.00% | 50.00% | 10.49 |
| GRU + GloVe | 5.00% | 50.00% | 10.47 |
| LSTM + GloVe | 5.00% | 50.00% | 10.51 |

The paper's 1,000-image gallery and full training split remain necessary for a meaningful comparison. These models had only 20 training images, and sample queries often returned unrelated images. The paper's stated Adam learning rate of 0.05 is still the default in `config.yaml`; a one-epoch trial with that rate on the three-image subset produced a very large validation loss, so the 20-image verification run used 0.00001. This change is confined to the verification commands and does not alter the required model or loss.

The local `outputs/experiments/` directory contains the generated comparison CSV, validation R@10 curves, caption-length plot, and short/long query examples. COCO images, downloaded GloVe, VGG weights, embeddings, and checkpoints are excluded from Git because of their size.
