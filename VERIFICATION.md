# Verification and measured results

## Source fidelity follow up

The completed measurements below have not been changed. Inspection of the authors' original source revealed differences in feature normalization, recurrent implementation, caption sampling, padding, vocabulary construction, and optimizer configuration. A new isolated run is in progress using the verified original feature archive. The details and unresolved evaluation ambiguity are in [REPRODUCTION.md](REPRODUCTION.md). Exact reproduction of Table 1 has not been established. Both the existing smoke test and the new source fidelity tests pass.

## Full COCO run (30 September 2026)

COCO 2014 and the Karpathy split provided 113,287 training images, 5,000 validation images, and 5,000 test images. All images received ten-crop, ImageNet-pretrained VGG-19 `fc7` features. The linear baseline and three neural variants completed training. The paper's learning rate of 0.05 gave a first full-run GRU validation R@10 of 0.9% and very large loss; the full neural run was restarted from scratch with an explicit learning-rate override of 0.0001. That initial epoch is a diagnostic, not a test result. Batch size was 128, the maximum was 50 epochs, and early stopping patience was seven epochs.

| Neural method | Epochs completed | Best validation R@10 |
| --- | ---: | ---: |
| GRU | 18 | 75.2% |
| GRU + GloVe | 16 | 78.3% |
| LSTM + GloVe | 14 | 76.8% |

The test gallery contains the first 1,000 Karpathy test images. Standard methods use the first held-out caption of each image, giving 1,000 queries per method. Baseline + Weight pools exactly five held-out captions into each of its 1,000 queries. One source test image has a sixth caption, which is excluded from its pooled query. The evaluator uses the best validation checkpoint for each neural method.

| Method | R@1 | R@10 | Mean rank |
| --- | ---: | ---: | ---: |
| Baseline | 4.5% | 21.8% | 170.46 |
| Baseline + Weight | 4.9% | 24.2% | 158.75 |
| GRU | 24.8% | 72.4% | 16.57 |
| GRU + GloVe | 23.8% | 76.1% | 12.30 |
| LSTM + GloVe | 21.0% | 72.0% | 14.81 |

These numbers were generated locally in `outputs/experiments/comparison.csv` and each method's `outputs/*_test_metrics.json`; they are not paper scores. Baseline + Weight has extra test-caption information and should be interpreted separately. The retrieval command `python -m src.retrieve --method gru_glove --query "a guy riding a bike next to a train" --top-k 5` ranked COCO image 483108 first; its held-out caption is “A man on a bicycle riding next to a train.” The experiment output also includes validation curves, caption-length analysis, and short/long query examples.

## Earlier small COCO subset check

This run verifies execution with real COCO 2014 images and captions. It is **not** the 1,000-image experiment in the paper and does not estimate full COCO retrieval quality.

- Date: 29 September 2026
- Karpathy split: 20 train, 20 validation, 20 test images, each with five captions
- Image representation: ImageNet-pretrained VGG-19 `fc7`, ten crops averaged, 4096 dimensions
- Text representation: Stanford GloVe 6B 300-dimensional vectors for baseline and pretrained neural variants; 133 of 135 vocabulary entries found. The two unmatched entries are the special tokens.
- Neural training: up to 30 epochs, batch size 128, Adam, learning rate 0.0001 for this tiny subset; validation-based best checkpoint and early stopping
- Test gallery: 20 images. This earlier pipeline check used all five captions as separate queries (100 queries) for normal methods. The final paper-protocol evaluation now selects one held-out caption per image (20 queries in this small gallery); Baseline + Weight pools all five into 20 queries. The scores below are from the earlier check and are not final project results.

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

The small-check models had only 20 training images, and sample queries often returned unrelated images; the full comparison above uses the paper's 1,000-image gallery and full training split. The paper's stated Adam learning rate of 0.05 is still the default in `config.yaml`; a one-epoch trial with that rate on the three-image subset produced a very large validation loss, so the 20-image verification run used 0.0001. The full run also used an explicit 0.0001 override after the initial 0.05 diagnostic. The image projection uses a small initial weight scale so its output starts near the recurrent caption state scale; the image encoder, similarity equation, and ranking loss remain as specified.

The local `outputs/experiments/` directory contains the generated comparison CSV, validation R@10 curves, caption-length plot, and short/long query examples. COCO images, downloaded GloVe, VGG weights, embeddings, and checkpoints are excluded from Git because of their size.
