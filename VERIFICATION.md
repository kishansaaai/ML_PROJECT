# Verification run (small COCO subset)

This run verifies execution with real COCO 2014 images and captions. It is **not** the 1,000-image experiment in the paper and does not estimate full COCO retrieval quality.

- Date: 29 September 2026
- Karpathy split: 20 train, 20 validation, 20 test images, each with five captions
- Image representation: ImageNet-pretrained VGG-19 `fc7`, ten crops averaged, 4096 dimensions
- Text representation: Stanford GloVe 6B 300-dimensional vectors for baseline and pretrained neural variants; 133 of 135 vocabulary entries found. The two unmatched entries are the special tokens.
- Neural training: up to 30 epochs, batch size 128, Adam, learning rate 0.0001 for this tiny subset; validation-based best checkpoint and early stopping
- Test gallery: 20 images. Normal methods use 100 caption queries. Baseline + Weight pools five held-out captions into 20 queries.

Verification training commands (after downloading the subset and extracting features):

```powershell
python -m src.train --variant gru --epochs 30 --learning-rate 0.0001
python -m src.train --variant gru_glove --epochs 30 --learning-rate 0.0001
python -m src.train --variant lstm_glove --epochs 30 --learning-rate 0.0001
```

| Method | R@1 | R@10 | Mean rank |
| --- | ---: | ---: | ---: |
| Baseline | 7.00% | 56.00% | 9.63 |
| Baseline + Weight | 10.00% | 50.00% | 9.50 |
| GRU | 6.00% | 51.00% | 9.92 |
| GRU + GloVe | 11.00% | 58.00% | 9.22 |
| LSTM + GloVe | 6.00% | 51.00% | 10.31 |

The paper's 1,000-image gallery and full training split remain necessary for a meaningful comparison. These models had only 20 training images, and sample queries often returned unrelated images. The paper's stated Adam learning rate of 0.05 is still the default in `config.yaml`; a one-epoch trial with that rate on the three-image subset produced a very large validation loss, so the 20-image verification run used 0.0001. This change is confined to the verification commands and does not alter the required model or loss. The image projection uses a small initial weight scale so its output starts near the recurrent caption state scale; the image encoder, similarity equation, and ranking loss remain as specified.

The local `outputs/experiments/` directory contains the generated comparison CSV, validation R@10 curves, caption-length plot, and short/long query examples. COCO images, downloaded GloVe, VGG weights, embeddings, and checkpoints are excluded from Git because of their size.
