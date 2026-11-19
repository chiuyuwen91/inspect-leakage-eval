# inspect-leakage-eval

A small [Inspect AI](https://inspect.aisi.org.uk/) eval. Each item is a short description of a machine
learning evaluation setup. The model must say whether the reported score is unreliable because of data
leakage (`ANSWER: leak`) or whether the setup is sound (`ANSWER: ok`).

All 24 cases are synthetic and were written for this project. There are 12 matched pairs: every leaky
setup has a fixed twin with the same topic and the same reported metric.

## Run

```bash
pip install -e ".[dev]"
pytest -q                                    # 10 tests, no network or API key needed
python baselines.py                          # validity checks on the dataset itself
inspect eval leakage_eval.py --model <provider/model>
```

For Bedrock, install `boto3` and use a model string like `bedrock/<model id or inference profile id>`.

## What the eval covers

Repeated rows from one unit split at random, scaler or feature selection or target encoding fit on all
data, look-ahead features, a column filled in after the outcome, duplicates across the split, random
split of time-ordered data, tuning on the test set, a photo-acquisition shortcut, benchmark
contamination, and oversampling before the split.

## Does the dataset leak its own answers? (evaluation of the evaluation)

I checked this before trusting any model score.

| Check | v1 | v2 (current) | Chance |
|---|---|---|---|
| Length baseline (leaky twin is shorter) | 0.83 | 0.50 | 0.50 |
| Word-cue baseline (leave-one-pair-out) | 0.79 | 0.67 | 0.50 |

v1 failed: the fixed twins needed extra words to explain the fix, so the shorter text was usually the
leaky one. A model could score well from length alone. v2 adds neutral filler sentences so length is
balanced, with the sign alternating across pairs.

The word-cue baseline is still above chance. The words it uses are things like "training set" and
"only", which belong to the fix itself, plus a few filler words. With 24 items the standard error is
about 0.10, so 0.67 is not far from chance, but it means this dataset is **not cue-free**. The test
suite only guards against making it worse (threshold 0.70).

## Scorer sanity check (stub models, not real results)

To check the scoring code end to end, I ran the task with three fake models that always give the same
reply. The dataset is balanced, so constant answers should score 0.50, and a reply with no answer line
should score 0.

| Fake model reply | Accuracy |
|---|---|
| `ANSWER: leak` every time | 0.50 |
| `ANSWER: ok` every time | 0.50 |
| "I am not sure." (no answer line) | 0.00 |

These runs use Inspect's `mockllm` and say nothing about any real model.

## Results with one real model (small demo)

Model: Claude Sonnet 4.5 on Amazon Bedrock, default settings, one epoch, 19 November 2026. I ran the
full task twice.

| Run | Accuracy | Standard error | Notes |
|---|---|---|---|
| 1 | 22 of 24 (0.917) | 0.058 | I did not keep the log, so I do not know which two items were missed. |
| 2 | 23 of 24 (0.958) | 0.042 | Missed `spurious_acquisition_cue-leak` (see below). 95% Wilson interval 0.80 to 0.99. |

The same model on the same items differed by one item between runs. A single run can mislead. For
comparison, the two cue baselines above score 0.67 (word cues) and 0.50 (length), so the model is well
above both, but 24 items is too few to say more. This is a demo, not a ranking.

## Error analysis (run 2)

The one miss is the skin-lesion case: most malignant photos have a ruler in the frame and most benign
photos do not, and test accuracy is 0.90. My label says `leak`. The model said `ok`, and gave this
reason: the ruler appears in both training and test photos, so the score honestly measures performance on
this test set; the real problem is that the model learns a shortcut and will fail on photos without a
ruler, which is a generalization problem and not leakage.

That argument is reasonable. In the strict sense, leakage means test information reaching training,
and a shortcut that exists in both splits is not that. Many people still call it leakage in a wider
sense. So this item is **ambiguous**, and the label depends on how "leakage" is defined. I did not change
the item after seeing the miss, because editing the data to fit the model would hide the problem.

What I would change in a next version: either ask "is this score unreliable as a guide to real-world
performance?" and keep a separate label for shortcuts, or define leakage in the system prompt and say
whether shortcut features count. The paired twin of this item (ruler in both classes) was answered
correctly.

## Limits

- 24 items is far too small to rank models. Treat any score as a demo.
- One author wrote all cases, and twins share vocabulary. A model could do well by pattern matching
  "fit on training only" without understanding leakage.
- The task checks a yes or no label. It does not check whether the model names the right reason.
- Only one model, one run. No repeated runs, so no estimate of run-to-run variation.

## Next steps

More cases with different phrasing, ask for the leak type and score the reason, run several models with
several epochs, and report confidence intervals.
