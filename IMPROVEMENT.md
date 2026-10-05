# Model improvement experiment

## Scope and status

On 5 October 2026 the user authorized a different model as a separate accuracy improvement experiment. The five original paper methods and their measured results remain available. This extension is not presented as a model from the original report or as an exact reproduction of VSE++.

Training and evaluation completed on 5 October 2026. The new encoder completed 31 epochs and stopped after eight epochs without validation improvement, following the rule fixed before training. Validation selected epoch 23. The combined model selected 75% original model distance and 25% new model distance from validation data only.

A fixed validation selection plan also compares the original and new encoders with original model weights of 0, 0.25, 0.5, 0.75 and 1. The plan is saved before the new test evaluation. The selected mixture maximizes validation Recall@10, then Recall@1, then lower mean rank. No weight is selected by proximity to a paper test score. Concatenating the two normalized representations with square root weights gives the weighted sum of their squared distances. The combined representation contains 2,048 dimensions.

## New model

1. A bidirectional GRU reads each caption in both directions. Each direction has 512 hidden units.
2. The encoder combines the mean of the word states with both final directional states and projects the result into 1,024 dimensions.
3. The image encoder maps the existing 4,096 dimensional VGG 19 features through two learned linear layers with normalization, GELU and dropout.
4. Both final representations are normalized. Retrieval continues to rank images by negative squared distance.
5. Word representations start from the same GloVe file used by the original run and are updated during training.

Training combines a symmetric contrastive objective with a margin penalty for the most confusing incorrect caption and image in each batch. Captions belonging to the same image are treated as positives. The hardest incorrect match term is inspired by [VSE++](https://arxiv.org/abs/1707.05612). Our bidirectional encoder, pooling, image projection and combined objective are a separate implementation.

## Fixed experiment settings

<table>
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Seed</td><td>42</td></tr>
<tr><td>Training images</td><td>113,287</td></tr>
<tr><td>Available training captions</td><td>566,435</td></tr>
<tr><td>Caption sampling</td><td>One randomly sampled caption per training image per epoch</td></tr>
<tr><td>Batch size</td><td>256, including the final partial batch</td></tr>
<tr><td>Optimizer</td><td>AdamW</td></tr>
<tr><td>Initial learning rate</td><td>0.0002</td></tr>
<tr><td>Weight decay</td><td>0.0001</td></tr>
<tr><td>Maximum epochs</td><td>50</td></tr>
<tr><td>Early stopping</td><td>At least 20 epochs, then stop after 8 epochs without validation improvement</td></tr>
<tr><td>Learning rate schedule</td><td>Multiply by 0.5 after epochs 15, 30 and 42</td></tr>
<tr><td>Gradient clipping</td><td>2.0</td></tr>
<tr><td>Dropout</td><td>0.1</td></tr>
<tr><td>Contrastive temperature</td><td>0.07</td></tr>
<tr><td>Hardest incorrect match margin</td><td>0.2 in cosine similarity units</td></tr>
<tr><td>Hardest incorrect match weight</td><td>0.25, enabled after three warmup epochs</td></tr>
</table>

## Data and comparison

The experiment uses the existing verified feature arrays, caption manifests and vocabulary. This vocabulary follows the original reproduction and includes training captions and captions from the first 1,000 validation images. It is reused rather than rebuilt from the test set. Features and image order are identical to the completed author convention run.

The validation gallery contains the first 1,000 validation images with 5,000 caption queries. The checkpoint is selected by validation Recall@10, followed by Recall@1 and then lower mean rank. Test scores are not used to select the checkpoint, learning rate or epoch.

After training, the selected checkpoint is evaluated on the same first 1,000 test images. Two results are recorded separately:

1. The main document protocol uses the first caption of each image, giving 1,000 queries.
2. The released evaluator protocol uses all five captions per image, giving 5,000 queries.

Both protocols use one caption per query. They are not pooled. Rankings are checked independently using direct squared distances. Image counts, caption counts, gallery identity, individual ranks and retrieval image paths are verified before completion.

The existing GRU + GloVe scores are 82.6% Recall@10 for the 1,000 query protocol and 85.04% for the 5,000 query protocol. The paper reports 86.8% for GRU + GloVe. The combined result of 88.00% exceeds that reported value by 1.20 percentage points on our reconstructed released evaluator. The original report's unresolved caption selection and random state details remain disclosed in REPRODUCTION.md. This is a separately trained extension and a measured local comparison, rather than proof of an exact Table 1 protocol match.

## Completed test measurements

<table>
<tr><th>Model</th><th>Gallery images</th><th>Caption queries</th><th>Recall@1</th><th>Recall@10</th><th>Mean rank</th></tr>
<tr><td>Original GRU + GloVe</td><td>1,000</td><td>1,000</td><td>34.5%</td><td>82.6%</td><td>9.699</td></tr>
<tr><td>New bidirectional GRU + GloVe</td><td>1,000</td><td>1,000</td><td>36.5%</td><td>85.0%</td><td>8.666</td></tr>
<tr><td>Combined model</td><td>1,000</td><td>1,000</td><td>38.6%</td><td>85.5%</td><td>8.174</td></tr>
<tr><td>Original GRU + GloVe</td><td>1,000</td><td>5,000</td><td>36.64%</td><td>85.04%</td><td>8.1462</td></tr>
<tr><td>New bidirectional GRU + GloVe</td><td>1,000</td><td>5,000</td><td>38.46%</td><td>87.34%</td><td>7.1098</td></tr>
<tr><td>Combined model</td><td>1,000</td><td>5,000</td><td>41.20%</td><td>88.00%</td><td>6.5698</td></tr>
</table>

The selected combination improves Recall@10 by 2.90 percentage points in the main protocol and 2.96 points in the released evaluator protocol. Its validation Recall@10 is 89.64%, compared with 87.38% for the original model and 87.68% for the new encoder. Validation values are not substituted for test results.

## Run and resume

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.improve_run --config config_improved.yaml
```

If interrupted, resume from the last completed epoch:

```powershell
.\.venv\Scripts\python.exe -m src.improve_run --config config_improved.yaml --resume
```

After training, choose the mixture using validation data and measure its test result:

```powershell
.\.venv\Scripts\python.exe -m src.improve_select --config config_improved.yaml
```

The operating system lock prevents duplicate runs writing the same checkpoints. Each epoch saves the model, optimizer, learning rate scheduler, history and random generator states. Resumption checks the configuration fingerprint and vocabulary.

Local checkpoints are saved under `checkpoints/improved`. Local progress and measured results are saved under `outputs/improved`. These resources remain excluded from Git.

After completion, the new model can be demonstrated from the terminal:

```powershell
.\.venv\Scripts\python.exe -m src.retrieve --config config_improved.yaml --method bigru_glove --query "a guy riding a bike next to a train"
```

After mixture selection, demonstrate the combined model:

```powershell
.\.venv\Scripts\python.exe -m src.retrieve --config config_improved.yaml --method ensemble_glove --query "a guy riding a bike next to a train"
```

## Verification before training

Seven focused tests passed. They checked padding and batch ordering, normalized representations, checkpoint restoration, loss gradients, exclusion of duplicate positives, independent ranking calculations, the exclusive writer lock and compatibility with existing retrieval and gallery boundaries. A small fixture also verified that resumed training gives exactly the same weights as uninterrupted training and rejects a changed configuration. The mixture check verified weighted distance equivalence, checkpoint restoration and validation based selection from the fixed weight grid.

Final verification recomputed all 12,000 saved ranks across the two extension methods and two protocols. Counts, rank bounds and metrics matched. Independent direct squared distance rankings agreed exactly for all 5,000 queries of each extension method. All 31 epoch records contain 443 batches, 113,287 training samples and finite losses. Actual saved representation dimensions are 1,024 for the bidirectional model and 2,048 for the combined model, with normalized outputs and finite weights.

Five real application checks passed with no errors: combined model clean input, equivalent capitals and punctuation, punctuation only input, completely unknown input, and original model clean input. Equivalent captions returned identical image captions, invalid inputs produced friendly warnings, and both models remained runnable. All saved retrieval image paths exist inside the same verified 1,000 image gallery. The original training settings and completed checkpoints remain preserved.
