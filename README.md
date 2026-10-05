# Neural Caption Image Retrieval

Search a COCO image gallery using a natural language description. This project implements the five retrieval methods described in the CS229 project by Qian and Lamberti, adds a separately authorized model improvement experiment, and provides a local Streamlit interface for demonstrating the trained models.

The project connects an image representation and a caption representation in a shared space. Given a caption, it ranks the available images by how closely their representations match the caption. It retrieves existing images from the gallery.

## Current project status

The improvement experiment completed on 5 October 2026. A new bidirectional GRU trained for 31 epochs, stopping according to the preset validation patience rule. Validation selected epoch 23. Combining it with the original GRU + GloVe using weights chosen from validation data increased measured test Recall@10 from 85.04% to 88.00% on the released evaluator protocol. This protocol contains 1,000 gallery images and 5,000 separate caption queries. On the main 1,000 query protocol, Recall@10 increased from 82.6% to 85.5%.

The 88.00% measurement is 1.20 percentage points above the paper's reported GRU + GloVe value of 86.8%. This is a result for the new model combination on our reconstructed released evaluator. The original paper's exact Table 1 caption selection and random state remain unavailable, so this is not a claim of exact experimental reproduction or a general performance advantage on every dataset.

The interface defaults to the improved combined model when its verified local result and checkpoint are available. The original GRU + GloVe remains selectable. Five application checks passed for the updated interface, including equivalent caption formatting and invalid input handling.

The final author convention training run completed on 3 October 2026 using `config_author.yaml`. GRU, GRU + GloVe, and LSTM + GloVe each completed 50 epochs. All five methods were evaluated on exactly 1,000 test images with exactly 1,000 queries per method. A separate evaluation follows the released source convention with 5,000 caption queries for each neural checkpoint.

The repository includes the source, configuration files, tests, documentation, presentation, and two page write up. Data, feature arrays, pretrained resources, trained checkpoints, and generated measurements are kept locally and are excluded from Git. A fresh clone therefore needs those resources before it can perform retrieval.

The local interface was checked with 28 input cases. Searches returned the requested number of results without app errors, and blank or space only input displayed the expected warning. This verifies that the interface executes the retrieval workflow; it does not mean every returned image is correct.

Exact numerical reproduction of the paper has not been achieved. All scores in the measured results sections below come from the local completed run.

## Project documents

The completed improvement experiment uses a bidirectional GRU, training that emphasizes confusing incorrect matches, and a combination chosen using validation data. It uses the same COCO splits and verified VGG features. See [Model improvement experiment](IMPROVEMENT.md) for the architecture, settings, measured results, comparison protocol and reproducible commands.

1. [Presentation](ML_PROJECT_review.pptx): the final review slide deck.
2. [Two page write up](ML_PROJECT_writeup.pdf): the problem, dataset, approach, implementation, and conclusions.
3. [Verification record](VERIFICATION.md): completed checks, measurements, and historical runs.
4. [Reproduction record](REPRODUCTION.md): source conventions, paper comparisons, and remaining reproduction limits.

The GitHub repository is private. The owner will provide faculty and Teaching Assistant access separately.

## Methods implemented

<table>
<tr><th>Method</th><th>Identifier</th><th>Approach</th></tr>
<tr><td>Baseline</td><td><code>baseline</code></td><td>Average GloVe word vectors and fit a linear least squares mapping to the image feature space.</td></tr>
<tr><td>Baseline + Weight</td><td><code>baseline_weighted</code></td><td>Pool the five held out caption representations for an image into one baseline query.</td></tr>
<tr><td>GRU</td><td><code>gru</code></td><td>Learn word embeddings and encode captions with a GRU.</td></tr>
<tr><td>GRU + GloVe</td><td><code>gru_glove</code></td><td>Initialize word embeddings with GloVe and encode captions with a GRU.</td></tr>
<tr><td>LSTM + GloVe</td><td><code>lstm_glove</code></td><td>Initialize word embeddings with GloVe and encode captions with an LSTM.</td></tr>
</table>

The weighted baseline receives five captions for each query. The other methods receive one caption per query in the main document protocol. This difference in available query information should be considered when comparing scores.

The Streamlit interface demonstrates the improved combined model and the original GRU + GloVe. Terminal evaluation supports the five original methods, `bigru_glove` for the new bidirectional GRU, and `ensemble_glove` for the selected combination. Interactive single caption retrieval supports Baseline, GRU, GRU + GloVe, LSTM + GloVe and both extension methods. The pooled weighted baseline is evaluated through the evaluation command because its query requires five held out captions.

## Measured model improvement

Every row below uses exactly 1,000 test gallery images. The two caption query protocols are reported separately. Recall@10 is the percentage of queries whose correct image appears in the first ten results.

<table>
<tr><th>Model</th><th>Caption queries</th><th>Recall@1</th><th>Recall@10</th><th>Mean rank</th></tr>
<tr><td>Original GRU + GloVe</td><td>1,000</td><td>34.5%</td><td>82.6%</td><td>9.699</td></tr>
<tr><td>New bidirectional GRU + GloVe</td><td>1,000</td><td>36.5%</td><td>85.0%</td><td>8.666</td></tr>
<tr><td>Combined model</td><td>1,000</td><td>38.6%</td><td>85.5%</td><td>8.174</td></tr>
<tr><td>Original GRU + GloVe</td><td>5,000</td><td>36.64%</td><td>85.04%</td><td>8.1462</td></tr>
<tr><td>New bidirectional GRU + GloVe</td><td>5,000</td><td>38.46%</td><td>87.34%</td><td>7.1098</td></tr>
<tr><td>Combined model</td><td>5,000</td><td>41.20%</td><td>88.00%</td><td>6.5698</td></tr>
</table>

The combined model uses 75% of the original model distance and 25% of the new model distance. These weights maximize validation Recall@10 over the fixed choices recorded before test evaluation. Its validation Recall@10 is 89.64%. Each query supplies one caption in both protocols. The combination does not pool the five test captions or use their target image labels at inference.

To demonstrate the improved model directly:

```powershell
.\.venv\Scripts\python.exe -m src.retrieve --config config_improved.yaml --method ensemble_glove --query "a guy riding a bike next to a train"
```

The new model reads captions in both directions, pools its word states and final states, and learns a nonlinear mapping of the existing VGG features. Its training combines a contrastive objective with a penalty for confusing incorrect matches. The image feature extractor remains the original pretrained VGG 19. The original five method measurements below remain unchanged and are labelled as the paper reproduction.

## Dataset and representations

The project uses COCO 2014 with the Karpathy split. Images marked as restval are included in training.

<table>
<tr><th>Split</th><th>Images</th><th>Captions</th></tr>
<tr><td>Training</td><td>113,287</td><td>566,435</td></tr>
<tr><td>Validation</td><td>5,000</td><td>25,000</td></tr>
<tr><td>Test</td><td>5,000</td><td>25,000</td></tr>
</table>

The final run uses the [original feature archive](https://www.cs.toronto.edu/~vendrov/order/coco.zip) linked in the [authors' repository](https://github.com/giacomolamberti90/CS229_project). The archive contains caption files and 4,096 dimensional VGG 19 fc7 features obtained from ten crops. The features are normalized. Every feature row was associated with an image by checking all five accompanying captions against the Karpathy annotations.

The archive SHA256 is:

```text
bd639ad9cefab5dd33dd79f1452fc665532b7ba17ab91a0b80cce0fc042a65a3
```

The vocabulary contains 27,011 entries including padding and unknown tokens. Following the released source, it uses all training captions and captions from the first 1,000 validation images. Caption encoding uses exact whitespace tokens and a maximum of 50 positions. GloVe initialization uses the 300 dimensional GloVe 6B vectors. Missing GloVe entries begin at zero, and the word embeddings are fine tuned during training.

Raw COCO photographs are separate from the feature archive. They are needed to display retrieved images and generate retrieval figures. Neural retrieval from a saved checkpoint uses the imported image features and the vocabulary stored inside that checkpoint.

## Repository structure

```text
ML_PROJECT/
    app.py
    download_gallery.py
    requirements.txt
    config_author.yaml
    config_improved.yaml
    config_reference.yaml
    config.yaml
    README.md
    VERIFICATION.md
    REPRODUCTION.md
    IMPROVEMENT.md
    ML_PROJECT_writeup.pdf
    ML_PROJECT_review.pptx
    src/
        prepare_data.py
        prepare_reference_data.py
        preprocessing.py
        dataset.py
        image_encoder.py
        features.py
        baseline.py
        model.py
        improved_model.py
        improve_run.py
        improve_select.py
        legacy_recurrent.py
        legacy_optimizer.py
        loss.py
        train.py
        reference_run.py
        evaluate.py
        source_evaluation.py
        retrieve.py
        experiments.py
        config.py
    tests/
        smoke_test.py
        reference_test.py
        improved_test.py
```

`app.py` displays the caption input and ranked images. It invokes the same retrieval module used by the terminal demo. `download_gallery.py` downloads photographs referenced by the prepared test manifest.

The preparation modules download or import data, verify feature identities, and create manifests and vocabulary files. The dataset module pairs captions with memory mapped feature arrays. The model and legacy modules implement the recurrent encoders, image projection, and optimizer conventions. The training module saves progress and checkpoints. The evaluation and retrieval modules rank images. The experiments module creates the comparison CSV and the required plots and retrieval examples.

Four configuration files preserve the original project stages and the model extension:

1. `config_author.yaml` reproduces the five original paper methods and the original UI model.
2. `config_reference.yaml` preserves the earlier run using the original feature archive.
3. `config.yaml` preserves the earlier local feature extraction pipeline.
4. `config_improved.yaml` trains and evaluates the new bidirectional model and the selected combination.

Use `config_author.yaml` for the original reproduction and `config_improved.yaml` for the authorized extension. Each run has separate output and checkpoint directories. IMPROVEMENT.md provides the extension training and validation selection commands.

## Environment setup

The project was run with Python 3.12. The requirements include PyTorch, torchvision, NumPy, Pillow, PyYAML, tqdm, matplotlib, and Streamlit. Streamlit is pinned to the tested version in `requirements.txt`.

The commands below use Windows PowerShell. Run project commands from the repository root. Using the virtual environment's Python directly avoids needing to activate it in PowerShell.

For a fresh clone, GitHub access to the private repository is required:

```powershell
git clone https://github.com/kishansaaai/ML_PROJECT.git
cd ML_PROJECT
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

On the existing prepared machine, open a terminal in the `ML_PROJECT` folder and use the existing virtual environment. Creating another environment and downloading the data again is unnecessary when the required files are already present.

The code uses an available CUDA device and otherwise uses the CPU. To inspect CUDA availability:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

If GPU execution is required, follow the [official PyTorch installation instructions](https://pytorch.org/) for a build compatible with the machine. CPU execution is supported, but full training can take substantial time. Training duration depends on the hardware and should not be inferred from the time needed to run a saved model.

## Run the UI on a prepared machine

For the original model, the following resources must be available at the paths expected by `config_author.yaml`:

```text
checkpoints/author/gru_glove_best.pt
data/paper_reference/manifests/test.jsonl
data/paper_reference/data/coco/images/10crop/test.npy
data/coco/val2014/
```

The improved model uses the same gallery data. To enable it in the interface, also restore the completed combined checkpoint and its measured result:

```text
checkpoints/improved/ensemble_glove_best.pt
outputs/improved/selected_comparison.json
```

These local resources can be transferred separately from this trained workspace or produced using the improvement commands in IMPROVEMENT.md. The combined checkpoint contains both encoders and their vocabulary. When the improvement resources are available, the interface selects the combined model by default. It also provides the original model option.

Keep the complete test feature array and test manifest together in their verified order. The interface searches the first 1,000 images from that pair. The corresponding photographs must be present for those results to appear in the browser. The checkpoint stores its model parameters and vocabulary, so the GloVe text file is not needed for inference with that saved neural model.

If using another machine, transfer the necessary local resources separately or prepare the data and train a model as described below. Cloning GitHub alone does not download the trained model or data.

Start the interface:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open [the local interface](http://127.0.0.1:8501/) in a browser on the same computer.

1. Choose the improved combined model or the original GRU + GloVe.
2. Enter a short English description.
3. Choose 3, 5, or 10 results.
4. Click Search Images and wait for retrieval to finish.
5. Inspect the ranked photographs, similarity scores, and dataset captions.

Useful example descriptions include:

```text
a guy riding a bike next to a train
a dog catching a frisbee in a park
a group of people skiing on snowy mountains
a surfer riding a large ocean wave
a person using a laptop computer at a desk
```

For the saved final GRU + GloVe checkpoint, the cyclist beside a train example returns COCO image 483108 first. The query `dog` returns COCO image 255209 first. Rankings for arbitrary descriptions can differ in relevance.

The displayed caption is an existing annotation associated with the retrieved image. The model does not generate it. The similarity score is negative squared Euclidean distance. A value closer to zero indicates a closer match; it is not a probability or a confidence percentage.

The server must remain running for the page to work. Press Ctrl+C in its terminal to stop it. The local browser address does not provide access from a teammate's computer.

## Terminal demonstration

The improved combined model can be demonstrated without the browser:

```powershell
.\.venv\Scripts\python.exe -m src.retrieve --config config_improved.yaml --method ensemble_glove --query "a guy riding a bike next to a train" --top-k 5
```

The original GRU + GloVe remains available:

```powershell
.\.venv\Scripts\python.exe -m src.retrieve --config config_author.yaml --method gru_glove --query "a guy riding a bike next to a train" --top-k 5
.\.venv\Scripts\python.exe -m src.retrieve --config config_author.yaml --method gru_glove --query "dog" --top-k 5
```

The terminal prints a JSON list containing the result rank, similarity score, image identifier, local image path, and dataset caption. Open a returned path to inspect the corresponding photograph.

Other trained methods can be demonstrated using their matching checkpoints:

```powershell
.\.venv\Scripts\python.exe -m src.retrieve --config config_author.yaml --method baseline --query "a dog playing outside" --top-k 5
.\.venv\Scripts\python.exe -m src.retrieve --config config_author.yaml --method gru --query "a dog playing outside" --top-k 5
.\.venv\Scripts\python.exe -m src.retrieve --config config_author.yaml --method lstm_glove --query "a dog playing outside" --top-k 5
```

These commands default to the first 1,000 test images. A single interactive query is a demonstration, rather than a measurement of Recall@1 or Recall@10.

## Prepare data and train from a fresh clone

This section is for reproducing the complete experiment when the local resources are unavailable. The full image dataset, feature archive, and GloVe resources require considerable download time and disk space. Keep the data directories and configuration paths consistent.

First download the Karpathy annotations, COCO 2014 photographs, and GloVe. This preparation command uses the earlier default configuration; the subsequent reference import creates the final manifests and vocabulary required by `config_author.yaml`.

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data --download-karpathy --download-images --download-glove
New-Item -ItemType Directory -Force data/paper_reference | Out-Null
Invoke-WebRequest -Uri https://www.cs.toronto.edu/~vendrov/order/coco.zip -OutFile data/paper_reference/coco.zip
Get-FileHash -Algorithm SHA256 data/paper_reference/coco.zip
.\.venv\Scripts\python.exe -m src.prepare_reference_data --config config_author.yaml
```

Compare the downloaded archive hash with the recorded hash in the dataset section. The import verifies archive members, caption order, feature dimensions, finite values, and nonzero feature norms. It writes the verified manifests, final vocabulary, feature arrays, and provenance record under `data/paper_reference`.

The expected main local paths after preparation are:

```text
data/coco/dataset_coco.json
data/coco/train2014/
data/coco/val2014/
data/glove/glove.6B.300d.txt
data/paper_reference/provenance.json
data/paper_reference/vocab.json
data/paper_reference/manifests/train.jsonl
data/paper_reference/manifests/val.jsonl
data/paper_reference/manifests/test.jsonl
data/paper_reference/data/coco/images/10crop/train.npy
data/paper_reference/data/coco/images/10crop/val.npy
data/paper_reference/data/coco/images/10crop/test.npy
```

The final run uses imported VGG features directly. Another image feature extraction pass is not needed. Raw training images are useful for the complete local dataset but the final neural training consumes the imported feature arrays.

Run the full experiment:

```powershell
.\.venv\Scripts\python.exe -m src.reference_run --config config_author.yaml
.\.venv\Scripts\python.exe -m src.experiments --config config_author.yaml
```

The runner fits or copies the baseline, trains the three neural variants, evaluates all five main methods, records the example retrievals, and performs the separately labelled released source evaluation. The final configuration can reuse the completed reference baseline if its checkpoint exists. On a fresh clone without that checkpoint, it fits the baseline from the imported training arrays.

To continue an interrupted run, execute the same runner command again. It uses its saved stage record and last completed neural checkpoints. A run already recorded as complete returns without retraining. To demonstrate the completed project, use the saved checkpoint and retrieval commands instead of starting a new training run.

An individual interrupted neural model can be resumed explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.train --config config_author.yaml --variant gru_glove --resume --learning-rate 0.001
```

The last checkpoint includes optimizer state, random generator state, history, and the current epoch. Do not change the optimizer convention when resuming the final run.

For an existing prepared test manifest where photographs are missing, the gallery helper can download the referenced test images:

```powershell
.\.venv\Scripts\python.exe download_gallery.py
```

This helper supplies photographs for display. It does not create feature arrays, manifests, vocabulary, or trained checkpoints.

## Original model and training details

The neural models learn an image projection from 4,096 dimensional VGG features into a shared 1,024 dimensional space. GRU or LSTM encodes the caption into the same space. Both representations are normalized before ranking by negative squared Euclidean distance.

Training uses a bidirectional margin ranking loss. For each matched image and caption, it penalizes mismatched captions and images that score too closely to the correct pair. The loss is summed over the mismatched pairs, with margin 0.1.

The final configuration preserves the released source conventions:

<table>
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Image feature dimension</td><td>4,096</td></tr>
<tr><td>Word embedding dimension</td><td>300</td></tr>
<tr><td>Shared embedding dimension</td><td>1,024</td></tr>
<tr><td>Caption positions</td><td>50</td></tr>
<tr><td>Batch size</td><td>128</td></tr>
<tr><td>Full batch updates per epoch</td><td>886</td></tr>
<tr><td>Epochs per neural model</td><td>50</td></tr>
<tr><td>Margin</td><td>0.1</td></tr>
<tr><td>Optimizer</td><td>Legacy Adam</td></tr>
<tr><td>Learning rate</td><td>0.001</td></tr>
<tr><td>Optimizer epsilon</td><td>0.0000001</td></tr>
<tr><td>Local seed</td><td>42</td></tr>
<tr><td>Validation gallery</td><td>1,000 images</td></tr>
<tr><td>Validation queries</td><td>5,000 separate captions</td></tr>
</table>

Legacy recurrent cells reconstruct the Keras GRU reset convention, hard sigmoid gates, and LSTM forget bias. All 50 caption positions, including padding, are processed as in the released source. Each shuffled caption cycle samples one of the five captions per image. Full batches wrap into a second shuffled cycle to supply the final update.

Legacy Adam applies epsilon before moment bias correction. Although the paper lists a learning rate of 0.05, the released trainer never passes that field to Adam. The final run therefore explicitly uses the effective legacy default of 0.001. The original software versions are unavailable, so this is a documented source reconstruction.

Validation selected epoch 46 for GRU, epoch 46 for GRU + GloVe, and epoch 38 for LSTM + GloVe. The final epoch checkpoints are also retained. GRU + GloVe epoch 49 and LSTM + GloVe epoch 40 were predetermined from author logs for separate comparisons. Test results were not used to select a checkpoint or tune settings.

## Evaluation protocol and commands

The main document protocol uses the first 1,000 Karpathy test images as the gallery. Each standard method uses the first held out caption of each image as one query. The weighted baseline pools all five captions for each image into one query. Every main row therefore contains exactly 1,000 queries.

The reported metrics are:

1. Recall@1: the percentage of queries with the correct image ranked first.
2. Recall@10: the percentage of queries with the correct image among the first ten results.
3. Mean rank: the average position of the correct image, counting the first position as one.

Higher recall and lower mean rank indicate better retrieval. All five methods use the same image gallery. The implementation checks actual embedding counts, including the final partial image batch.

Recompute the main measurements from saved checkpoints:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --config config_author.yaml --method baseline
.\.venv\Scripts\python.exe -m src.evaluate --config config_author.yaml --method baseline_weighted
.\.venv\Scripts\python.exe -m src.evaluate --config config_author.yaml --method gru
.\.venv\Scripts\python.exe -m src.evaluate --config config_author.yaml --method gru_glove
.\.venv\Scripts\python.exe -m src.evaluate --config config_author.yaml --method lstm_glove
```

Generate the comparison CSV, validation recall curves, caption length analysis, and short and long retrieval figures:

```powershell
.\.venv\Scripts\python.exe -m src.experiments --config config_author.yaml
```

Evaluation consumes feature arrays and checkpoints. The retrieval figure step additionally opens the raw photographs. If photographs are unavailable, the experiments command can create the comparison and plots while skipping the image examples:

```powershell
.\.venv\Scripts\python.exe -m src.experiments --config config_author.yaml --skip-examples
```

## Original paper method results

These measurements use the main 1,000 image and 1,000 query protocol. Recall values are percentages.

<table>
<tr><th>Method</th><th>Recall@1</th><th>Recall@10</th><th>Mean rank</th></tr>
<tr><td>Baseline</td><td>9.5%</td><td>45.2%</td><td>37.056</td></tr>
<tr><td>Baseline + Weight</td><td>11.0%</td><td>52.2%</td><td>25.640</td></tr>
<tr><td>GRU</td><td>34.5%</td><td>82.8%</td><td>9.127</td></tr>
<tr><td>GRU + GloVe</td><td>34.5%</td><td>82.6%</td><td>9.699</td></tr>
<tr><td>LSTM + GloVe</td><td>31.7%</td><td>83.1%</td><td>8.636</td></tr>
</table>

Among the five original paper methods, GRU and GRU + GloVe tie for the highest Recall@1. LSTM + GloVe has the highest Recall@10 and lowest mean rank in that comparison. Neural methods outperform the local baseline methods on these measurements. The weighted baseline improves on the ordinary baseline while using more caption information.

## Original model and paper comparison

The paper describes one caption per image, while its released evaluator ranks all five captions separately. The caption selection used for the original Table 1 is unspecified. To keep the two conventions clear, the project reports an additional neural evaluation with 1,000 images and 5,000 caption queries.

The following supplementary results use the checkpoints already selected by validation:

<table>
<tr><th>Method</th><th>Measured Recall@1</th><th>Measured Recall@10</th><th>Measured mean rank</th><th>Paper Recall@10</th><th>Recall@10 difference</th></tr>
<tr><td>GRU</td><td>36.36%</td><td>84.52%</td><td>8.5242</td><td>85.8%</td><td>1.28 percentage points lower</td></tr>
<tr><td>GRU + GloVe</td><td>36.64%</td><td>85.04%</td><td>8.1462</td><td>86.8%</td><td>1.76 percentage points lower</td></tr>
<tr><td>LSTM + GloVe</td><td>33.80%</td><td>84.96%</td><td>7.8108</td><td>86.2%</td><td>1.24 percentage points lower</td></tr>
</table>

These values are separate from the main table because the number of queries differs. The published GRU row is attributed to Vendrov et al. in the paper. Published baseline results also differ substantially from the local implementation. Complete comparisons are retained in `REPRODUCTION.md`.

Recompute all validation selected, final epoch, and predetermined author epoch neural comparisons:

```powershell
.\.venv\Scripts\python.exe -m src.source_evaluation --config config_author.yaml --selection all
```

The original trained checkpoints, random state, software versions, and baseline source are unavailable. Those gaps prevent a guarantee of identical scores. The project retains every predetermined comparison and does not choose whichever test score is closest to the paper.

## Tests and verification

Run the pipeline and source fidelity tests:

```powershell
.\.venv\Scripts\python.exe -m tests.smoke_test
.\.venv\Scripts\python.exe -m tests.reference_test
.\.venv\Scripts\python.exe -m unittest tests.improved_test
```

The smoke test exercises synthetic data loading, baseline fitting, neural training, resumption, checkpoint loading, evaluation, and retrieval. Source fidelity tests compare recurrent equations and Adam updates with independent NumPy calculations, check gradients and normalization, exercise padding, and verify generator batch counts and gallery boundaries. Synthetic test results establish execution correctness rather than COCO accuracy.

Final verification also checked that all 150 neural epoch records had complete updates and finite losses, every validation epoch used exactly 1,000 images and 5,000 queries, and saved model tensors were finite. Main test measurements were independently recomputed from the checkpoints. The supplementary evaluator's ranks were checked against the authors' unchanged evaluator for the saved GRU + GloVe model. Retrieval paths and example rankings were also checked.

The delivered write up has two well filled pages with black text, and the presentation has fourteen slides with white backgrounds. Both include the original measurements and the completed improvement experiment. Both artifacts were rendered and visually checked, and all reported scores were checked against the measured records. The document review checks are recorded in VERIFICATION.md.

## Local outputs and checkpoints

The final run stores the following resources outside Git:

```text
checkpoints/author/baseline.npz
checkpoints/author/gru_best.pt
checkpoints/author/gru_last.pt
checkpoints/author/gru_history.json
checkpoints/author/gru_glove_best.pt
checkpoints/author/gru_glove_last.pt
checkpoints/author/gru_glove_epoch_49.pt
checkpoints/author/gru_glove_history.json
checkpoints/author/lstm_glove_best.pt
checkpoints/author/lstm_glove_last.pt
checkpoints/author/lstm_glove_epoch_40.pt
checkpoints/author/lstm_glove_history.json
outputs/author/status.json
outputs/author/comparison.csv
outputs/author/retrieval_examples.json
outputs/author/source_protocol_audit.json
outputs/author/experiments/comparison.csv
outputs/author/experiments/recall_at_10_epochs.png
outputs/author/experiments/caption_length.png
outputs/author/experiments/retrieval_short.png
outputs/author/experiments/retrieval_long.png
```

Each method also has its own test metrics JSON under `outputs/author`. Earlier run evidence remains under `outputs/reference` and `checkpoints/reference`.

The original feature archive can be removed after a successful verified import if another download is acceptable later. Keep the extracted feature arrays and manifests for retrieval. Keep trained checkpoints to avoid retraining, and keep photographs for image display. The archive itself is not a trained model and is not stored in GitHub.

## Caption input reliability update

On 5 October 2026, interactive retrieval gained vocabulary aware input cleaning. Existing vocabulary entries are preserved. If a word is unknown, the code tries its lowercase form and forms with surrounding punctuation removed, accepting a replacement only when that replacement exists in the saved vocabulary. This lets inputs such as `Dog`, `dog!`, and `dog` use the same known word representation while retaining meaningful punctuation inside known words. The browser and terminal retrieval use this cleaning with the original whitespace vocabulary.

The browser warns about descriptions containing only punctuation, descriptions with no recognized words, and input exceeding 50 words. Unknown words are not assigned invented meanings. A partly recognized description can still be searched using its recognized words and the model's unknown token.

A controlled validation check used the existing GRU + GloVe checkpoint, the first caption of each of the first 1,000 validation images, exactly 1,000 gallery images, and exactly 1,000 queries per condition. Each altered caption was converted to title case and given a trailing exclamation mark. The same model and gallery were used before and after input cleaning. No training was performed.

<table>
<tr><th>Validation input condition</th><th>Recall@1</th><th>Recall@10</th><th>Mean rank</th></tr>
<tr><td>Clean original captions</td><td>37.5%</td><td>87.9%</td><td>6.433</td></tr>
<tr><td>Altered captions before input cleaning</td><td>0.1%</td><td>1.0%</td><td>499.904</td></tr>
<tr><td>Altered captions after input cleaning</td><td>37.5%</td><td>87.9%</td><td>6.433</td></tr>
</table>

All 1,000 altered caption encodings were restored to the clean encodings, and their image rankings matched exactly. All 1,000 original caption encodings were preserved. These measurements demonstrate recovery from the specified capitalization and punctuation changes. They are a controlled input robustness check, not a replacement for the paper comparison. The main GRU + GloVe test Recall@10 remains 82.6%, and its separately labelled released evaluator test Recall@10 remains 85.04%. Training, checkpoint selection, and benchmark evaluation tokenization are unchanged.

Reproduce this validation check without training:

```powershell
.\.venv\Scripts\python.exe -m tests.query_validation --config config_author.yaml
```

The measured conditions and individual ranks are saved locally to `outputs/input_validation/query_normalization.json` and excluded from Git. Four focused regression tests and five Streamlit UI checks passed for vocabulary preservation, restored retrieval, gallery boundaries, caption targets, and input warnings.

## Known limitations and troubleshooting

Use short English descriptions with simple wording. Only the first 50 tokens are encoded. The browser warns when an input exceeds 50 words.

Descriptions containing no recognized words are rejected. Unknown words and emoji within a partly recognized description can still reduce retrieval quality. The gallery also contains only 1,000 images, so a requested scene might not be available. Rankings are not always semantically correct; for example, a tested cat on a couch query ranked an image of dogs on a couch first.

Each UI search starts a retrieval process and loads the saved model. Observed search times on the prepared local machine were roughly 8 to 14 seconds. Speed varies with the computer. The interface can also log a Streamlit sizing deprecation warning under the currently tested version; this did not prevent the tested searches from completing.

If Python reports a missing package, install the requirements with the same virtual environment used to start the app. If it reports that a module cannot be found, run the command from the repository root.

If the model checkpoint is missing, restore `checkpoints/author/gru_glove_best.pt` for the UI or complete the training workflow. A missing feature array requires the verified archive import. A feature shape mismatch means the selected manifest and feature array are not a matching pair; restore or regenerate them together through that import.

If the UI says Image file not found, restore the corresponding photograph under `data/coco` or use the gallery download helper after preparing the test manifest. Evaluation on stored features may still work without photographs, but the browser cannot display missing files.

If the page is unavailable, confirm that the Streamlit server is running and that the browser uses the same port shown in the terminal. If port 8501 is occupied, launch the server on another port and use that address:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

## References and acknowledgements

The methodology follows Qian and Lamberti's CS229 Neural Caption Image Retrieval project and its [released repository](https://github.com/giacomolamberti90/CS229_project). The source comparison used author commit `181baae2a01dc747c8bd47de913a9e35830f1477`.

The project uses COCO 2014, the Karpathy split, VGG 19 image features, and GloVe word vectors. The original feature archive and original author implementation are external resources. The local Streamlit interface is an added demonstration interface for the existing retrieval workflow.

The separate improvement experiment uses a bidirectional GRU, a contrastive objective and a hardest negative term inspired by [VSE++](https://arxiv.org/abs/1707.05612). IMPROVEMENT.md records the architecture, training settings, validation selection and measured comparison.
