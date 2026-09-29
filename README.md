# Neural Caption-Image Retrieval

This repository implements the caption-to-image retrieval project in the supplied CS229 document by Junyang Qian and Giacomo Lamberti. A caption is mapped to the same feature space as a COCO image, and the highest-scoring images are returned. It contains the paper's average-GloVe linear regression baseline and its VGG-19 plus GRU/LSTM models. There is no web application or later-generation image-text model.

## Project status

No existing implementation was supplied with the two project PDFs, so this repository was built from the paper's specification. A 20-image-per-split COCO run verifies the pipeline with real images, captions, pretrained VGG-19, and GloVe; its measured results are in [VERIFICATION.md](VERIFICATION.md). The full COCO run is underway: the complete dataset has been prepared and VGG-19 feature extraction is in progress. Neural training and the paper's 1,000-image test gallery evaluation are still pending, so the subset scores are not final project results.

## Data and preprocessing

The project uses COCO 2014 images and the [Karpathy train/validation/test split](https://cs.stanford.edu/people/karpathy/deepimagesent/) used in the paper: 113,287 training images and 5,000 images each for validation and test, with five captions per image. `restval` joins training. Vocabulary is built only from training captions. Captions are lowercased, tokenized, truncated to 40 tokens, padded within each batch, and packed for the recurrent encoder. Unknown words have a learned embedding in the neural models; the linear baseline omits unknown words from its average.

VGG-19 uses ImageNet pretrained weights, 256-pixel resize, ten 224-pixel crops (including flips), and the 4096-dimensional `fc7` output averaged over those crops. Features are saved as `.npy` arrays and memory-mapped during training; original images are never loaded into training RAM together. The pretrained 300-dimensional GloVe vectors initialize the neural embedding table and are fine-tuned.

Download sources: [COCO](https://cocodataset.org/#download), [Karpathy annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/), and [GloVe 6B](https://nlp.stanford.edu/projects/glove/). COCO images and GloVe require substantial disk space. The download command is explicit so it cannot start an unplanned large transfer.

## Models and objective

The baseline averages GloVe vectors in each caption and solves multi-response least squares for a matrix mapping the 300-dimensional caption feature to the 4096-dimensional VGG feature. Retrieval selects images with the smallest squared Euclidean distance to that mapped caption.

The neural models map VGG features linearly into a 1024-dimensional shared space and encode caption words with a GRU or LSTM into that same space. The three variants are `gru` (word embeddings trained from scratch), `gru_glove`, and `lstm_glove`. Their similarity is the paper's `S(i,c) = -||f_i(i)-f_c(c)||²`. The loss applies a margin of 0.1 to both wrong-caption and wrong-image pairs within each batch. Other captions of the same image are excluded from negatives.

The paper's starting configuration is retained: batch size 128, up to 50 epochs, Adam with learning rate 0.05, and early stopping from validation R@10 and loss. These settings can be changed in `config.yaml` or through the training command where supported.

## Setup and run

From the repository root, with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.prepare_data --download-karpathy --download-images --download-glove
python -m src.features --split all --overwrite
python -m src.baseline
python -m src.train --variant gru
python -m src.train --variant gru_glove
python -m src.train --variant lstm_glove
```

On a supported NVIDIA GPU, install the matching CUDA builds of PyTorch and torchvision using the [official PyTorch selector](https://docs.pytorch.org/get-started/locally/) before the full run. Check `python -c "import torch; print(torch.cuda.is_available())"`; `True` means this project will use the GPU. If feature extraction is interrupted, run `python -m src.features --split train --resume` (or `val`/`test`). If neural training is interrupted, run `python -m src.train --variant gru --resume` with the applicable variant. Progress is saved after each completed epoch.

If the files have already been downloaded, put `dataset_coco.json` under `data/coco/`, COCO `train2014/` and `val2014/` there, and `glove.6B.300d.txt` under `data/glove/`. Then run `python -m src.prepare_data` without download flags. `src.features` downloads the pretrained torchvision VGG-19 weights automatically on its first run.

For a local COCO subset without the full image archives, `python -m src.prepare_data --download-subset 20 --download-glove` downloads 20 real COCO images from each split and builds matching manifests. This is for pipeline checks only. Regenerate full manifests before full experiments.

## Evaluation and retrieval

Evaluate on the first 1,000 test images with one held-out caption per image, as in the paper's normal-method comparison. Metrics are Recall@1, Recall@10, and mean rank; percentages are saved as JSON. The first 1,000 images in the test manifest define the reproducible gallery.

```powershell
python -m src.evaluate --method baseline
python -m src.evaluate --method baseline_weighted
python -m src.evaluate --method gru
python -m src.evaluate --method gru_glove
python -m src.evaluate --method lstm_glove
python -m src.experiments
```

`baseline_weighted` reproduces the paper's special test where all five held-out captions of an image are averaged into one query. It uses information unavailable for a normal single-caption search, so compare it with this caveat. `src.experiments` writes the comparison table, validation R@10 curves, caption-length analysis, and short/long query examples to `outputs/experiments/`. It plots only metrics produced by local evaluation.

Search the test gallery from a terminal:

```powershell
python -m src.retrieve --method gru_glove --query "a guy riding a bike next to a train" --top-k 5
python -m src.retrieve --method baseline --query "dog" --top-k 5
```

Results contain ranked image paths, similarity scores, and a dataset caption. VGG features are loaded from the saved arrays, not recalculated per query.

## Verification

Run the self-contained synthetic COCO-format fixture:

```powershell
python -m tests.smoke_test
python -m tests.smoke_test --vgg
python -m tests.smoke_test --pretrained-vgg
```

The first command checks manifest loading, GloVe parsing, baseline fitting, GRU/LSTM training, loss/backward, checkpoint loading, retrieval, and evaluation. The second checks ten-crop preprocessing and a VGG-19 `fc7` forward pass with random weights. The third extracts a synthetic image with pretrained VGG-19 to verify the complete feature path. This fixture proves execution, not retrieval quality. Actual COCO feature extraction is performed by `python -m src.features`.

## Results and limits

The original paper reports results for its own implementation. This repository does not copy those numbers into its output. See [VERIFICATION.md](VERIFICATION.md) for measured results from a 20-image-per-split COCO execution check. The complete dataset is prepared, and the full run must finish before paper-scale results can be reported. After the commands above run, `outputs/experiments/comparison.csv` contains this implementation's measured results. Caption length groups with fewer than five queries are omitted from the plot.
