# Neural Caption Image Retrieval

This repository implements the five caption to image retrieval methods in the project by Qian and Lamberti: a linear baseline, Baseline + Weight, GRU, GRU + GloVe, and LSTM + GloVe.

## Project status

The final run with the authors' original feature archive completed on 2 October 2026. All three neural models completed 50 epochs. All five methods were evaluated with exactly 1,000 test queries on the same 1,000 image gallery. Recomputing results from the saved checkpoints reproduced every recorded score and both retrieval examples.

The final presentation is [ML_PROJECT_review.pptx](ML_PROJECT_review.pptx), and the two page write up is [ML_PROJECT_writeup.pdf](ML_PROJECT_writeup.pdf). [VERIFICATION.md](VERIFICATION.md) records measured results and checks. [REPRODUCTION.md](REPRODUCTION.md) describes source details, the comparison with the paper, and unresolved evaluation ambiguity. The repository is private. The owner will grant faculty and TA access separately.

## Data and representations

The Karpathy split contains 113,287 training images, 5,000 validation images, and 5,000 test images. The [original feature archive](https://www.cs.toronto.edu/~vendrov/order/coco.zip), linked by the [authors](https://github.com/giacomolamberti90/CS229_project), supplies five captions per image and normalized 4,096 dimensional VGG 19 fc7 features from ten crops. The importer checks the caption order against the annotations for every feature row. The archive checksum is recorded in REPRODUCTION.

Following the released source, the vocabulary includes training captions and captions from the first 1,000 validation images. It has 27,011 entries including two special tokens. Captions use whitespace tokenization and 50 padded positions. Neural GloVe initialization uses 300 dimensional GloVe 6B vectors, with zero vectors for missing words. Pretrained word embeddings are fine tuned.

The earlier local feature extraction pipeline is retained with `config.yaml`. It averages ten crops of torchvision VGG 19 features and uses a different vocabulary and packed recurrent sequences. Those older checkpoints and measurements remain available for comparison. The final deliverables use `config_reference.yaml` and the isolated reference run.

## Methods and training

The baseline averages word vectors and fits a multiple response least squares map to VGG image features. Baseline + Weight pools five held out captions into one query. Standard methods use one held out caption per image.

The neural methods project image features with a bias into a 1,024 dimensional space and encode captions with GRU or LSTM. Both resulting embeddings are normalized. Negative squared Euclidean distance scores a match. The bidirectional margin ranking loss sums mismatched pairs with margin 0.1.

The reference cells reproduce legacy Keras gate equations, including the GRU reset convention, hard sigmoid recurrent gates, and the LSTM forget bias. They process all 50 positions, including padding. Each epoch samples one of five captions per image. Batch size is 128. Each model completed the configured maximum of 50 epochs; validation selected the checkpoint. The local random seed is 42 and random state is saved for resumption.

Adam uses an explicit learning rate of 0.001. The paper lists 0.05, but the released source does not pass its `lrate` field to the optimizer. The effective default and the missing original software environment are documented in REPRODUCTION. Test scores did not select settings or checkpoints.

## Setup and final run

From the repository root, use Python 3.12 and install the requirements. COCO images are needed for retrieval figures and a live demonstration. Downloads are large.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.prepare_data --download-karpathy --download-images --download-glove
New-Item -ItemType Directory -Force data/paper_reference
Invoke-WebRequest https://www.cs.toronto.edu/~vendrov/order/coco.zip -OutFile data/paper_reference/coco.zip
python -m src.prepare_reference_data --config config_reference.yaml
python -m src.reference_run --config config_reference.yaml
python -m src.experiments --config config_reference.yaml
```

For an NVIDIA GPU, install compatible CUDA builds using the [official PyTorch instructions](https://pytorch.org/). Check availability:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If COCO annotations, images, and GloVe are already present, skip their download command. Put `dataset_coco.json` and the image folders under `data/coco`, and `glove.6B.300d.txt` under `data/glove`. Import the original feature archive with the command above. The reference run uses those imported arrays directly and does not require another feature extraction pass.

The reference runner resumes completed stages and saved neural progress. Run it again after an interruption. Individual neural training can also resume:

```powershell
python -m src.train --config config_reference.yaml --variant gru_glove --resume --learning-rate 0.001
```

## Evaluation and live retrieval

The first 1,000 Karpathy test images form the gallery. Standard methods use the first caption of each image. Baseline + Weight pools exactly five captions per image, so it receives more query information. The report describes one caption per image, but the released evaluator scores all five separately. The original Table 1 caption selection is unspecified. Our declared protocol has 1,000 queries for every method.

```powershell
python -m src.evaluate --config config_reference.yaml --method baseline
python -m src.evaluate --config config_reference.yaml --method baseline_weighted
python -m src.evaluate --config config_reference.yaml --method gru
python -m src.evaluate --config config_reference.yaml --method gru_glove
python -m src.evaluate --config config_reference.yaml --method lstm_glove
python -m src.experiments --config config_reference.yaml
python -m src.retrieve --config config_reference.yaml --method gru_glove --query "a guy riding a bike next to a train" --top-k 5
python -m src.retrieve --config config_reference.yaml --method gru_glove --query "dog" --top-k 5
```

Retrieval returns ranked image paths, scores, and a dataset caption. The longer example puts the cyclist beside a train at rank 2, with a bicycle attached to a bus first. The short query returns a dog at rank 1. These are recorded observations and do not imply every query is correct.

## Measured final test results

Every row below contains local measurements from the completed reference run. Recall@1 and Recall@10 are percentages. Lower mean rank is better.

```text
Method                    R@1     R@10    Mean rank
Baseline                   9.5%   45.2%     37.06
Baseline + Weight         11.0%   52.2%     25.64
GRU                       34.6%   83.1%      9.36
GRU + GloVe               32.6%   82.9%      9.19
LSTM + GloVe              27.9%   81.2%      9.39
```

GRU achieved the highest measured Recall@1 and Recall@10. GRU + GloVe had the lowest mean rank. Best validation Recall@10 was 87.9% for GRU, 87.6% for GRU + GloVe, and 86.5% for LSTM + GloVe. Validation selected epochs 42, 28, and 24 respectively.

The published neural Recall@10 values are 85.8%, 86.8%, and 86.2%. Our values are lower by 2.7, 3.9, and 5.0 percentage points respectively. The baseline results also differ. Exact paper score reproduction has not been achieved. The original trained checkpoints, random seed, dependency versions, and Table 1 caption selection are unavailable. REPRODUCTION lists the full reference table and measured comparison.

## Verification and local files

```powershell
python -m tests.smoke_test
python -m tests.reference_test
```

The synthetic fixture checks data loading, baseline fitting, training, checkpoint loading, evaluation, and retrieval. The source fidelity tests independently check the GRU and LSTM equations, gradients, embedding normalization, padding, summed loss, and compatibility with the earlier checkpoints. Synthetic checks measure execution rather than COCO accuracy.

The final local measurements are in `outputs/reference`, with comparison figures in `outputs/reference/experiments`. Checkpoints are in `checkpoints/reference`. Both example rankings were reproduced and every returned image path exists. All saved model tensors and feature arrays are finite. The working local environment contains the data and checkpoints for a live demonstration.

Datasets, original feature arrays, downloaded weights, GloVe, checkpoints, and generated experiment outputs are excluded from Git. The source code, README, verification records, presentation, and write up are hosted in the private repository. Reproducing measurements on another machine requires the data and training commands above.
