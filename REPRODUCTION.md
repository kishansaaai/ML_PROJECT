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
<tr><td>GRU</td><td>34.6%</td><td>83.1%</td><td>9.36</td></tr>
<tr><td>GRU + GloVe</td><td>32.6%</td><td>82.9%</td><td>9.19</td></tr>
<tr><td>LSTM + GloVe</td><td>27.9%</td><td>81.2%</td><td>9.39</td></tr>
</table>

The neural Recall@10 differences from Table 1 are 2.7, 3.9, and 5.0 percentage points lower for GRU, GRU + GloVe, and LSTM + GloVe respectively. GRU has the highest measured Recall@1 and Recall@10 here, while GRU + GloVe has the lowest mean rank. This differs from the paper's model ordering. Both baseline rows also differ substantially from the published results.

## Evaluation ambiguity

The report says standard methods use one caption per image. Its released evaluator instead scores all five captions separately, creating 5,000 queries for a 1,000 image gallery. The exact caption selection and checkpoint used for Table 1 are not specified. The source and logs also reflect several experimental revisions.

The new final evaluation keeps the document protocol: 1,000 gallery images and 1,000 queries per method. Standard methods use the first held out caption. Baseline + Weight pools five held out captions per image. This is a declared choice, not proof that the paper used the same selection.

Exact numerical identity would require the original checkpoints, exact evaluation selection, random state, and software environment. Their absence prevents a justified promise of identical scores. The measured results above keep those differences visible.

## Run and resume

Download the linked archive into `data/paper_reference/coco.zip`, then run these commands from the repository root. The runner resumes saved neural progress and keeps the completed earlier run separate.

```powershell
python -m src.prepare_reference_data --config config_reference.yaml
python -m tests.reference_test
python -m src.reference_run --config config_reference.yaml
```

The status is recorded in `outputs/reference/status.json`. Training histories are in `checkpoints/reference`. The archive, feature arrays, checkpoints, and generated measurements are excluded from Git.
