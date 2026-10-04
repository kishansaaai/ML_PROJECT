# Neural Caption Image Retrieval

This repository implements the five caption to image retrieval methods in the project by Qian and Lamberti: a linear baseline, Baseline + Weight, GRU, GRU + GloVe, and LSTM + GloVe.

## Project status

The final author convention run completed on 3 October 2026. All three neural models completed 50 epochs. All five methods were evaluated with exactly 1,000 test queries on the same 1,000 image gallery. The final documents report measured results from config_author.yaml. Earlier reference results remain available for comparison. Exact paper score reproduction has not been achieved.

The presentation is [ML_PROJECT_review.pptx](ML_PROJECT_review.pptx), and the two page write up is [ML_PROJECT_writeup.pdf](ML_PROJECT_writeup.pdf). [VERIFICATION.md](VERIFICATION.md) records checks and measurements. [REPRODUCTION.md](REPRODUCTION.md) compares every protocol with the paper and documents the missing original artifacts. The repository is private. The owner will grant faculty and TA access separately.

## Optional local interface

The Streamlit page runs the saved GRU + GloVe checkpoint on the existing COCO image gallery. After installing the requirements and preparing the data, start it from the repository root:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501 in a browser, enter a caption, and select the number of results. The page shows ranked images from the same 1,000 image test gallery used by the terminal retrieval command.

## Data and representations

The Karpathy split contains 113,287 training images, 5,000 validation images, and 5,000 test images. The [original feature archive](https://www.cs.toronto.edu/~vendrov/order/coco.zip), linked by the [authors](https://github.com/giacomolamberti90/CS229_project), supplies five captions per image and normalized 4,096 dimensional VGG 19 fc7 features from ten crops. Every feature row was checked against the Karpathy caption order. The archive checksum is in REPRODUCTION.

Following the released source, the vocabulary includes training captions and captions from the first 1,000 validation images. It has 27,011 entries including two special tokens. Captions use whitespace tokenization and 50 padded positions. Neural GloVe initialization uses 300 dimensional GloVe 6B vectors, with zeros for missing words. Word embeddings are fine tuned.

The earlier local feature extraction pipeline remains under config.yaml. The completed original feature reference run remains under config_reference.yaml. Final deliverables use config_author.yaml and its isolated measurements.

## Methods and training

The baseline averages GloVe word vectors and fits a multiple response least squares map to VGG features. Baseline + Weight pools five held out captions into one query. Standard methods use one held out caption per image.

The neural models project images with a bias into a 1,024 dimensional space and encode captions with GRU or LSTM. Both embeddings are normalized. Negative squared Euclidean distance scores a match. The bidirectional margin ranking loss sums mismatched pairs with margin 0.1.

Legacy recurrent cells reproduce the Keras GRU reset convention, hard sigmoid gates, and LSTM forget bias. They process all 50 positions including padding. Each caption cycle samples one caption per image. The source generator uses 886 full batches per epoch, wrapping into a second shuffled cycle as required. Batch size is 128 and the local seed is 42. Random state and optimizer progress are saved for resumption.

Legacy Adam uses an explicit learning rate of 0.001 and epsilon 0.0000001 before moment bias correction. The paper lists 0.05, but its released source never passes that field to Adam. Validation uses exactly 1,000 images and 5,000 separate caption queries. Validation alone selected epochs 46, 46, and 38 for GRU, GRU + GloVe, and LSTM + GloVe. Preset author log epochs 49 and 40 are retained separately. Test scores did not select settings or checkpoints.

## Setup and final run

From the repository root, use Python 3.12 and install the requirements. COCO images are needed for retrieval figures and a live demonstration.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.prepare_data --download-karpathy --download-images --download-glove
New-Item -ItemType Directory -Force data/paper_reference
Invoke-WebRequest https://www.cs.toronto.edu/~vendrov/order/coco.zip -OutFile data/paper_reference/coco.zip
python -m src.prepare_reference_data --config config_author.yaml
python -m src.reference_run --config config_author.yaml
python -m src.experiments --config config_author.yaml
```

For an NVIDIA GPU, install compatible CUDA builds using the [official PyTorch instructions](https://pytorch.org/). Check availability:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If the files are already present, skip downloading them. Put dataset_coco.json and the image folders under data/coco, and glove.6B.300d.txt under data/glove. The final run uses imported feature arrays directly and does not need another feature extraction pass. If the earlier reference baseline checkpoint is unavailable, the runner fits the baseline from those arrays.

The runner resumes completed stages and saved neural progress. Run it again after an interruption. Individual neural training can resume:

```powershell
python -m src.train --config config_author.yaml --variant gru_glove --resume --learning-rate 0.001
```

## Evaluation and live retrieval

The first 1,000 Karpathy test images form the gallery. Standard methods use the first caption of each image. Baseline + Weight pools exactly five captions per image into one query and receives more query information. Every main comparison row has exactly 1,000 queries. Actual embedding dimensions are checked, including the final partial image batch.

```powershell
python -m src.evaluate --config config_author.yaml --method baseline
python -m src.evaluate --config config_author.yaml --method baseline_weighted
python -m src.evaluate --config config_author.yaml --method gru
python -m src.evaluate --config config_author.yaml --method gru_glove
python -m src.evaluate --config config_author.yaml --method lstm_glove
python -m src.experiments --config config_author.yaml
python -m src.retrieve --config config_author.yaml --method gru_glove --query "a guy riding a bike next to a train" --top-k 5
python -m src.retrieve --config config_author.yaml --method gru_glove --query "dog" --top-k 5
```

Retrieval returns ranked image paths, scores, and a dataset caption. The longer example returns the cyclist beside a train at rank 1, and the short query returns a dog at rank 1. Both top five rankings were reproduced on the actual 1,000 image gallery. These examples do not imply every query is correct.

## Measured final test results

These are local measurements from the completed author convention run. Recall@1 and Recall@10 are percentages. Lower mean rank is better.

```text
Method                    R@1     R@10    Mean rank
Baseline                  9.5%   45.2%     37.06
Baseline + Weight        11.0%   52.2%     25.64
GRU                      34.5%   82.8%      9.13
GRU + GloVe              34.5%   82.6%      9.70
LSTM + GloVe             31.7%   83.1%      8.64
```

GRU and GRU + GloVe tie for the highest Recall@1. LSTM + GloVe has the highest Recall@10 and lowest mean rank under this protocol. Neural Recall@10 is below the paper by 3.0, 4.2, and 3.1 percentage points in GRU, GRU + GloVe, and LSTM + GloVe order. The baseline results also differ substantially.

## Released evaluator comparison

The report describes one caption per image, while the released evaluator scores all five separately. The original Table 1 caption selection is unspecified. Following the released evaluator with 5,000 caption queries and 1,000 images, the validation selected checkpoints measure GRU 84.52%, GRU + GloVe 85.04%, and LSTM + GloVe 84.96% Recall@10. They are 1.28, 1.76, and 1.24 percentage points below the corresponding paper values. These results are separate from the 1,000 query table above.

All final epoch and predetermined author log epoch measurements are reported in REPRODUCTION. They are not used to select whichever test score is closest to the paper. Original weights, seeds, software versions, and baseline code are unavailable. Identical scores cannot be guaranteed.

```powershell
python -m src.source_evaluation --config config_author.yaml --selection all
```

## Verification and local files

```powershell
python -m tests.smoke_test
python -m tests.reference_test
```

The tests check data loading, baseline fitting, neural training, resumption, checkpoint loading, evaluation, and retrieval. Source fidelity checks compare recurrent equations and Adam updates with independent NumPy calculations, test gradients, normalization and padding, and validate full batch counts and actual gallery boundaries. Synthetic checks measure execution rather than COCO accuracy.

Final measurements are under outputs/author, figures under outputs/author/experiments, and checkpoints under checkpoints/author. Earlier measurements and checkpoints remain under outputs/reference and checkpoints/reference. All final model tensors and training losses are finite. The working local environment contains data and checkpoints for a live demonstration.

Datasets, original features, downloaded weights, GloVe, checkpoints, and generated measurements are excluded from Git. Source code, documentation, the presentation, and the write up are hosted in the private repository. Reproducing measurements elsewhere requires the data and training commands above.
