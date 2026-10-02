# Verification and measured results

## Final original feature run (2 October 2026)

Training and final evaluation completed on the full COCO split using the authors' original feature archive and `config_reference.yaml`. The archive contains 113,287 training images and 5,000 images in each held out split, with exactly five captions per image. All feature row identities were verified against Karpathy captions. The vocabulary contains 27,011 entries and GloVe initialized 22,061 entries. Embeddings are normalized and legacy Keras recurrent equations are used.

All three models completed 50 epochs with Adam learning rate 0.001, batch size 128, margin 0.1, one randomly sampled caption per training image, and summed ranking loss. Validation alone selected the saved best checkpoints. GRU selected epoch 42 with 87.9% validation R@10, GRU + GloVe selected epoch 28 with 87.6%, and LSTM + GloVe selected epoch 24 with 86.5%.

Each final method has exactly 1,000 queries on the first 1,000 Karpathy test images. Standard methods use the first caption. Baseline + Weight pools five captions per query. The following values were recomputed from saved checkpoints and matched every recorded metric:

```text
Method                    R@1     R@10    Mean rank
Baseline                   9.5%   45.2%     37.06
Baseline + Weight         11.0%   52.2%     25.64
GRU                       34.6%   83.1%      9.36
GRU + GloVe               32.6%   82.9%      9.19
LSTM + GloVe              27.9%   81.2%      9.39
```

The recheck verified finite saved model parameters, 50 epoch histories, the actual learning rate, validation checkpoint selection, and all gallery and query counts. The original smoke test and source fidelity tests pass. The experiment figures use the new histories and measured caption length results. Generated measurements are stored locally under `outputs/reference` and are excluded from Git.

The GRU + GloVe query “a guy riding a bike next to a train” returns COCO 352476 first and the intended cyclist beside a train, COCO 483108, second. Their scores are approximately negative 1.227210 and negative 1.238391. The query “dog” returns COCO 255209 first, with score approximately negative 1.102623. Both complete top five rankings were reproduced and every image path exists.

The new results improve upon the earlier neural run but do not reproduce Table 1 exactly. Measured neural R@10 is below the paper by 2.7 points for GRU, 3.9 for GRU + GloVe, and 5.0 for LSTM + GloVe. Published and measured baselines also differ. See [REPRODUCTION.md](REPRODUCTION.md) for every reference score and the original evaluator's caption protocol ambiguity. No published value is substituted for a measured result.

The final write up and presentation report this completed run. The older measurements below are retained as historical evidence.

The final PDF was rendered and visually inspected on both pages. It has exactly two well filled pages with black text and rules on white, no draft label, and no dash characters in visible text or metadata. The final ten slide PowerPoint passed package and layout checks, was rendered, and was visually inspected on every slide. All backgrounds are white, the test table matches the recorded measurements, and the example is labelled rank 2. README prose has no dash characters and runnable command flags remain valid.

## Earlier local feature run (30 September 2026)

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
