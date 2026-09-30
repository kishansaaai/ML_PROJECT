# Neural Caption Image Retrieval

This repository implements the caption to image retrieval project described by Qian and Lamberti. A natural language query is encoded and compared with images from MS COCO. The project follows the specified VGG 19, GloVe, GRU, and LSTM approach.

## Project status

Full COCO feature extraction and training are complete. All five specified methods were evaluated on a fixed gallery of 1,000 test images. Each method received 1,000 queries. The recorded measurements are in [VERIFICATION.md](VERIFICATION.md). The presentation is [ML_PROJECT_review.pptx](ML_PROJECT_review.pptx).

## Data and image features

The Karpathy split contains 113,287 training images, 5,000 validation images, and 5,000 test images. Images usually have five captions, although a few source records contain extras. Training captions alone define the vocabulary. Captions are lowercased, tokenized, limited to 40 tokens, and padded within each batch. The recurrent encoders use packed sequences for variable caption lengths.

The image pipeline resizes each COCO image to 256 by 256 pixels, creates ten 224 by 224 crops including horizontal flips, and averages their 4,096 dimensional VGG 19 fc7 features. VGG 19 uses weights pretrained on ImageNet. Saved feature arrays are mapped into memory during training, so the full image collection is not loaded into training memory.

The project uses 300 dimensional GloVe 6B word vectors. The pretrained recurrent embedding tables are fine tuned. The linear baseline averages available word vectors in each caption.

Data sources: [COCO](https://cocodataset.org/#download), [Karpathy annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/), and [GloVe](https://nlp.stanford.edu/projects/glove/).

## Methods

The linear baseline fits a multiple response least squares map from averaged GloVe caption vectors to VGG image features. Retrieval ranks images by squared Euclidean distance to the mapped query. The Baseline + Weight evaluation combines the first five held out captions of each test image into one query.

The neural variants are GRU, GRU + GloVe, and LSTM + GloVe. Each projects the VGG vector into a 1,024 dimensional shared space and represents the caption using the final recurrent state. Similarity is the negative squared Euclidean distance between those two representations. A bidirectional margin ranking loss separates matching pairs from wrong captions and wrong images within each batch. Other captions of the same image are excluded from negative pairs.

The configuration uses a margin of 0.1, batch size 128, Adam, a maximum of 50 epochs, and validation based early stopping with patience seven. The paper lists a learning rate of 0.05. Its first full GRU epoch gave ineffective validation, so the completed neural runs were restarted with an explicit learning rate of 0.0001. The actual value is used in the commands below.

## Setup and full run

From the repository root, use Python 3.12. Install the requirements before preparing data. The download command starts a large COCO and GloVe transfer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.prepare_data --download-karpathy --download-images --download-glove
python -m src.features --split all --overwrite
python -m src.baseline
python -m src.train --variant gru --learning-rate 0.0001
python -m src.train --variant gru_glove --learning-rate 0.0001
python -m src.train --variant lstm_glove --learning-rate 0.0001
```

For a supported NVIDIA GPU, install matching CUDA builds of PyTorch and torchvision using the [official PyTorch site](https://pytorch.org/) before the full run. Check GPU availability:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If the data has already been downloaded, place `dataset_coco.json` under `data/coco/`, the COCO `train2014/` and `val2014/` folders there, and `glove.6B.300d.txt` under `data/glove/`. Then prepare manifests without download flags:

```powershell
python -m src.prepare_data
```

Interrupted feature extraction and training can resume from saved progress. Select the relevant split and model:

```powershell
python -m src.features --split train --resume
python -m src.train --variant gru --resume --learning-rate 0.0001
```

For a small execution check with real COCO images, a subset command downloads 20 images from each split. This verifies the pipeline and does not produce full project results:

```powershell
python -m src.prepare_data --download-subset 20 --download-glove
```

Regenerate full manifests before running the full experiment after a subset check.

## Evaluation and retrieval

The first 1,000 test images form the gallery. Normal methods use the first held out caption of each image. Baseline + Weight pools exactly five held out captions per image. One source image has a sixth caption, which is excluded from that pooled query. The pooled method has more text information and should be compared with that qualification.

Evaluation reports Recall@1, Recall@10, and mean rank. Validation selects the best neural checkpoint; test data is used only for final measurement.

```powershell
python -m src.evaluate --method baseline
python -m src.evaluate --method baseline_weighted
python -m src.evaluate --method gru
python -m src.evaluate --method gru_glove
python -m src.evaluate --method lstm_glove
python -m src.experiments
```

The experiment command writes the measured comparison, validation Recall@10 curves, caption length analysis, and short and long retrieval examples. Search the saved test gallery from a terminal:

```powershell
python -m src.retrieve --method gru_glove --query "a guy riding a bike next to a train" --top-k 5
python -m src.retrieve --method baseline --query "dog" --top-k 5
```

Retrieval returns ranked image paths, scores, and a dataset caption. VGG features are reused from saved arrays instead of being extracted from every image for every query.

## Measured results

These values come from the local full COCO run and are not copied from the paper. Every row uses 1,000 test queries against the same 1,000 image gallery. The pooled baseline receives five captions per query; the other methods receive one.

```text
Method                    R@1     R@10    Mean rank
Baseline                  4.5%    21.8%   170.46
Baseline + Weight          4.9%    24.2%   158.75
GRU                       24.8%   72.4%    16.57
GRU + GloVe               23.8%   76.1%    12.30
LSTM + GloVe              21.0%   72.0%    14.81
```

GRU + GloVe achieved the highest single caption Recall@10. GRU achieved the highest Recall@1. Best validation Recall@10 was 75.2% for GRU, 78.3% for GRU + GloVe, and 76.8% for LSTM + GloVe. Training ended after 18, 16, and 14 epochs. [VERIFICATION.md](VERIFICATION.md) records the run conditions and a checked retrieval example.

## Verification

The synthetic COCO format fixture checks manifest loading, GloVe parsing, baseline fitting, recurrent forward and backward passes, training, checkpoint loading, evaluation, and retrieval. Additional modes check VGG feature extraction.

```powershell
python -m tests.smoke_test
python -m tests.smoke_test --vgg
python -m tests.smoke_test --pretrained-vgg
```

The fixture proves execution, not retrieval quality. Actual COCO results are the measured values above.

## Local files and limits

COCO images, GloVe files, VGG weights, feature arrays, checkpoints, and generated experiment outputs are excluded from Git because of their size. Reproducing the full results on another computer requires the data download, feature extraction, and training commands above. The working local environment already contains the completed data and model checkpoints for a live demonstration.
