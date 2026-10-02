# Reproduction of the published results

## Current status

The isolated original feature run completed on 2 October 2026. All three neural models completed 50 epochs. All five final results were recomputed and verified on the declared 1,000 query protocol. Published scores have not been reproduced exactly.

The earlier checkpoints and results remain available. Final checkpoints are under `checkpoints/reference` and measurements under `outputs/reference`. The final presentation, write up, README, and verification record now report the completed original feature run.

## Reference evidence

The reference is the [authors' repository](https://github.com/giacomolamberti90/CS229_project) at commit `181baae2a01dc747c8bd47de913a9e35830f1477`, including its report, trainer, data loader, evaluator, and published training logs. The [original COCO feature archive](https://www.cs.toronto.edu/~vendrov/order/coco.zip) is the archive linked in their README.

The downloaded archive has SHA256 `bd639ad9cefab5dd33dd79f1452fc665532b7ba17ab91a0b80cce0fc042a65a3`. Every feature row was associated with an image by comparing all five accompanying captions with the corresponding Karpathy annotations. The verified counts are 113,287 training images, 5,000 validation images, and 5,000 test images. The vocabulary has 27,011 entries including two special tokens, matching the authors' logged dimensions. Features are finite and have nonzero norms.

## Corrections for the new run

The run uses the released 4,096 dimensional features. These features have unit norm and differ from the locally extracted torchvision features. Both learned image and caption representations are also normalized to unit length before computing negative squared Euclidean similarity. The image projection includes a bias and uses Glorot initialization.

Each training epoch visits each image once, choosing one of its five captions uniformly. Incomplete final batches are dropped. The bidirectional ranking loss sums the incorrect pairs. Batch size remains 128 and the shared embedding size remains 1,024. The margin is 0.1 as specified in the report and the published experiment logs.

The dictionary uses all training captions plus the first 1,000 validation images, with whitespace tokenization, all observed words, and frequency ordering as in the released source. Missing GloVe vectors start at zero. Captions are padded to 50 positions. The padding embedding is trainable and the recurrent cell processes those positions because the released caption embedding does not enable masking.

The GRU uses the legacy Keras reset convention, hard sigmoid gates, and tanh candidate activation. The LSTM uses hard sigmoid recurrent gates and a forget gate bias of one. Tests independently compare both implementations with NumPy equations, check gradients and normalization, exercise training, and verify checkpoint loading. The original pipeline smoke test also passes.

The source lists `lrate` as 0.05 but compiles the model with Adam selected by name. The field is never supplied to the optimizer. This run explicitly uses 0.001, the legacy Adam default, rather than treating the unused field as an effective rate. The exact original dependency versions are unavailable, so this interpretation is documented rather than claimed as a verified optimizer trace.

The maximum remains 50 epochs. Checkpoint selection uses validation performance only. Test scores were first computed after training and recomputed for verification. They were not used to select settings, epochs, or random seeds. Random generator state is saved for resumable training. The local seed is 42; the authors did not publish their random seed.

## Published target values

<table>
<tr><th>Method</th><th>Paper R@1</th><th>Paper R@10</th><th>Paper mean rank</th></tr>
<tr><td>Baseline</td><td>10.3%</td><td>17.1%</td><td>176.1</td></tr>
<tr><td>Baseline + Weight</td><td>19.8%</td><td>65.5%</td><td>10.5</td></tr>
<tr><td>GRU reference</td><td>36.3%</td><td>85.8%</td><td>7.6</td></tr>
<tr><td>GRU + GloVe</td><td>37.0%</td><td>86.8%</td><td>7.3</td></tr>
<tr><td>LSTM + GloVe</td><td>35.4%</td><td>86.2%</td><td>7.5</td></tr>
</table>

These are reference values transcribed from Table 1, not local measurements. The GRU row is explicitly attributed to Vendrov et al. The authors' repository does not include the original trained checkpoints or the baseline regression implementation.

## Measured comparison

<table>
<tr><th>Method</th><th>Measured R@1</th><th>Measured R@10</th><th>Measured mean rank</th></tr>
<tr><td>Baseline</td><td>9.5%</td><td>45.2%</td><td>37.06</td></tr>
<tr><td>Baseline + Weight</td><td>11.0%</td><td>52.2%</td><td>25.64</td></tr>
<tr><td>GRU</td><td>34.9%</td><td>83.4%</td><td>9.14</td></tr>
<tr><td>GRU + GloVe</td><td>32.9%</td><td>83.1%</td><td>9.01</td></tr>
<tr><td>LSTM + GloVe</td><td>28.2%</td><td>81.7%</td><td>9.18</td></tr>
</table>

The neural Recall@10 differences from Table 1 are 2.4, 3.7, and 4.5 percentage points lower for GRU, GRU + GloVe, and LSTM + GloVe respectively. GRU has the highest measured Recall@1 and Recall@10 here, while GRU + GloVe has the lowest mean rank. This differs from the paper's model ordering. Both baseline rows also differ substantially from the published results.

## Source evaluation audit and correction

The complete author repository history was inspected. No trained HDF5 checkpoints or baseline implementation appears in that history. The earliest trainer used order embeddings; a later revision switched to normalized Euclidean embeddings and GloVe initialization. The GRU row in Table 1 is an external Vendrov reference rather than a trained model supplied in this repository.

The saved GRU + GloVe evaluation logs contain the exact published pair 86.8% Recall@10 and mean rank 7.3. The continued training log contains this pair after epoch 49. The saved LSTM + GloVe evaluation logs contain 86.2% and 7.5, with the matching training record after epoch 40. The corresponding Recall@1 logs contain 37.0% and 35.4%. This is evidence linking the paper values to those logs. It does not prove the exact original checkpoint file or validation selection.

The released loader selects the first 1000 test images and their 5000 captions. The evaluator ranks each of those captions separately and assigns target image index equal to caption index divided by five. It uses NumPy argsort on squared Euclidean distances. This differs from the report's description of one caption per image. The supplementary evaluation follows the released source and keeps the document's 1000 query comparison separate.

The local image encoder loop previously included 24 extra gallery rows in its final batch while its metadata said 1000. The loop now clamps the last slice and asserts the actual embedding dimensions. All current document protocol scores were recomputed on exactly 1000 images. The prior values are retained in the ignored status record. The historical validation checkpoint scores also used 1024 actual image embeddings; the recorded selection is preserved and is not claimed as a corrected 1000 image validation search.

The supplementary measured results from the already selected checkpoints are below. The final epoch checkpoints were also evaluated without using test results to choose between them; all six rows are retained locally in outputs/reference/source_protocol_audit.json. No test score selects a new setting or checkpoint.

<table>
<tr><th>Method</th><th>Queries</th><th>Measured R@1</th><th>Measured R@10</th><th>Mean rank</th></tr>
<tr><td>GRU</td><td>5000</td><td>37.58%</td><td>85.26%</td><td>7.9646</td></tr>
<tr><td>GRU + GloVe</td><td>5000</td><td>35.10%</td><td>85.02%</td><td>7.8240</td></tr>
<tr><td>LSTM + GloVe</td><td>5000</td><td>31.46%</td><td>83.88%</td><td>8.0992</td></tr>
</table>

The supplementary neural Recall@10 differences from the paper are 0.54, 1.78, and 2.32 percentage points lower. The protocol explains part of the earlier difference, but numerical identity remains unachieved.

The original t2i function was also executed unchanged using independently computed NumPy Euclidean norms for the GRU + GloVe checkpoint. Every one of its 5000 query ranks matched the supplementary implementation. This verifies the evaluator arithmetic and target mapping on the real gallery. The retrieval image batch now uses the same strict gallery boundary, and both existing example rankings were rechecked.

## Further source fidelity training

A separate config_author.yaml run uses the same verified features, vocabulary, recurrent cells, embeddings and loss. It corrects two remaining implementation differences: legacy Adam with epsilon 0.0000001 before moment bias correction, and 886 full batch updates per epoch. The original fit generator uses a fractional steps_per_epoch and its logs confirm 886 rather than 885 updates. Full batches wrap into a second shuffled caption cycle. As in the source, a whole caption cycle is sampled before yielding batches.

The legacy Adam convention is derived from the Keras 2.2.4 implementation. This is a documented reconstruction, not proof of the authors' dependency version. The exact asynchronous generator prefetch timing and original random state are unknown. Independent NumPy checks validate the Adam updates, tiny gradients and optimizer resumption. A generator fixture validates full batches and ceiling step counts.

The new run retains 50 epochs, seed 42, Adam learning rate 0.001, batch 128 and margin 0.1. It uses exactly 1000 validation images and 5000 validation caption queries for checkpoint selection. The author log epochs 49 for GRU + GloVe and 40 for LSTM + GloVe are saved as preset comparison checkpoints independently of local test results. All test comparisons occur after training. Progress and measurements use outputs/author, with checkpoints/author, preserving completed results.

The original checkpoints, random state, environment and baseline details remain unavailable. Further training can improve fidelity but cannot guarantee identical published scores. Current final documents retain the completed reference measurements while this isolated run is active.

## Run and resume

Download the linked archive into `data/paper_reference/coco.zip`, then run these commands from the repository root. The runner resumes saved neural progress and keeps the completed earlier run separate.

```powershell
python -m src.prepare_reference_data --config config_reference.yaml
python -m tests.reference_test
python -m src.reference_run --config config_reference.yaml
```

The status is recorded in `outputs/reference/status.json`. Training histories are in `checkpoints/reference`. The archive, feature arrays, checkpoints, and generated measurements are excluded from Git.
