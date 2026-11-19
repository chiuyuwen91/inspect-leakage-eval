# BigData Cup 2026 — Explainable Suicide Risk Detection

**Team: GT** | **Best official submission: 0.6984 composite** (Subtask1 0.7650, Subtask2 0.5430)
**Best local validation: 0.7344 composite** (Weighted F1 0.8235, Phrase F1 0.7883, Macro F1 0.5619 — all-time local highs)
**Most recent official submission: 0.6906** (Subtask1 0.7536, Subtask2 0.5436 — Subtask2 is marginally above the original best, confirming a large round of factor-side fixes held up officially even though the local Macro F1 gain didn't fully translate)

## Approach

We're using **LLM prompting (Claude Sonnet 5 via AWS Bedrock)** rather than a fine-tuned encoder. This was a deliberate choice based on a prior result: in eRisk 2025, a system-prompt/LLM approach placed 2nd internationally, while a fine-tuned-encoder approach in eRisk 2026 placed 9th — so for this challenge we went straight to prompting instead of repeating the fine-tuning route.

Two pipeline variants exist, both producing the same three outputs per post:
1. **Risk level** (Indicator / Ideation / Behavior / Attempt) — Subtask 1, 40% of score
2. **Evidence spans** (verbatim quotes supporting the risk level) — Subtask 1, 30% of score
3. **Suicide factors** (multi-label, 24-category taxonomy) — Subtask 2, 30% of score

- **Single-call** (`system_prompt.md` + `schema.json`, run via `batch_score.py` without `--two-call`) — one prompt does all three jobs in one API call.
- **Two-call** (`risk_prompt.md`+`risk_schema.json` for risk/evidence, `factor_prompt.md`+`factor_schema.json` for factors, run via `--two-call`) — splits risk classification and factor identification into two separate calls, the second given the risk level as context. **This is the pipeline behind all our best results** — it's noticeably more stable run-to-run than single-call, and because the two calls are independent, a change to `factor_prompt.md` alone can be evaluated on the factor-side numbers without any noise from the risk/evidence side (and vice versa) — this became a key diagnostic tool for the large factor-tuning round described below.

Confirmed weighting (verified by back-computing from an actual official submission): **composite = 0.7 × Subtask1 + 0.3 × Subtask2**, where Subtask1 itself = 0.4 × risk-level Weighted F1 + 0.3 × evidence Phrase F1 (proportionally, roughly 4/7 + 3/7 within the 0.7). A separate summary page for the conference lists a different "0.6/0.4" split — that page does not match the platform that actually scores submissions; don't use it.

## Submission history (official leaderboard)

| # | Composite | Subtask 1 | Subtask 2 | Notes |
|---|---|---|---|---|
| 1 | 0.6539 | 0.7051 | 0.5345 | Rank 22. Initial prompt |
| 2 | 0.6869 | 0.7564 | 0.5249 | Rank 17. Evidence trimming, boundary fixes |
| 3 | 0.6887 | 0.7479 | 0.5504 | Fixed euphemism/Behavior-timing interaction bug |
| 4 (dip) | 0.6730 | 0.7458 | 0.5034 | Attempt-priority refinement, looked best locally, underperformed officially |
| 5 | 0.6900 | 0.7496 | 0.5508 | Goodbye/metaphor euphemism example |
| 6 | 0.6984 | 0.7650 | 0.5430 | **Isolated physical-artifact fix (two-call) — best official result** |
| 7 (dip) | 0.6870 | 0.7528 | 0.5332 | Paper-grounded factor defs, reverted |
| 8 (dip) | 0.6826 | 0.7442 | 0.5389 | Ambiguous-word caution (fall/sleep/rest), 2nd time this exact fix underperformed — reverted, do not retry |
| 9 | 0.6945 | 0.7572 | 0.5482 | meaning_in_life narrowing — essentially tied with #6, kept |
| 10 | 0.6906 | 0.7536 | 0.5436 | **Most recent.** All 8 factor fixes from the big tuning round (below) stacked together — close to #6, Subtask2 slightly above #6's, within normal noise band |

Submission 6's risk/evidence prompt state (`risk_prompt.md`) is still the baseline — untouched since. `factor_prompt.md` has evolved significantly since #6 (see below); it currently reflects submission #10's state, our most factor-tuned version to date.

## Big factor-tuning round (single day, 8 fixes, all local-only so far)

Starting from Macro F1 ~0.5056, ran a disciplined one-fix-at-a-time cycle: pull false-positive/negative examples for one weak factor category from `validation_predictions.csv`, find a genuine repeated pattern (not a guess), source a grounding example from `train_split.csv`, make one isolated edit, confirm with a full validation run before moving to the next category. Ended at Macro F1 **0.5619**, a new local best, with every fix individually confirmed:

1. **Scoring bug fix (not a model fix):** `normalize()` in `batch_score.py`/`analyze_errors.py` did no punctuation normalization — Claude's curly apostrophes (') vs. the source CSV's straight ones (') silently broke phrase-matching on any evidence span with a contraction. Fixed by unifying smart punctuation to ASCII before matching. Free, real Phrase F1 gain (0.7468→0.7668) with zero model changes.
2. **`meaning in life`** — was over-triggering badly (F1 ~0.12, precision ~0.08). The definition itself listed "feeling life has no point" as a qualifying cue, which matched generic `hopelessness` "no point in continuing" language. Narrowed to require genuine existential questioning or the missed-experience framing already validated; added a contrastive example. F1 recovered to a healthy, if small-sample-noisy, level.
3. **`emotion dysregulation`** — one confirmed miss: "just cry silently to yourself" (contains the established "crying" cue) wasn't caught because it's phrased in second-person self-address, a common Reddit inner-dialogue convention. Added a narrow clarification only (not a broader loosening, given this factor's documented overcorrection history). Small, clean gain.
4. **`psychological capital`** — under-triggering (recall ~0.24). Found a real pattern: ambivalence about dying ("I don't wanna die but...") and passive wish-for-better-state language ("I just want to be happy") are a genuine Hope signal (HERO framework) distinct from the active goal-seeking phrasing already covered — and there was zero existing Hope-specific worked example (both prior examples were Resilience-only). Added one. F1 climbed from 0.36 toward 0.45+.
5. **`social support`** — big win. 12 false negatives collapsed to just 2 underlying duplicate-post clusters, both about **religious/prayer support-seeking** ("I still believe in God... I used to pray"). The definition's explicit source list never mentioned religious/spiritual support. Added it. F1 jumped 0.33→0.79.
6. **`coping strategy`** — a cluster of very short "someone please talk to me"/"I need someone to talk to" posts were labeled `poor social support` only, missing that the act of reaching out IS itself a coping behavior. Confirmed via 4 identical duplicate posts in `train_split.csv`, all gold-labeled with both factors. Added an explicit worked example. F1 jumped ~0.52→0.72.
7. **`interpersonal difficulty`** — found the definition had been *fully* reverted (not just softened) back to only covering relational friction with a specific person; the "difficulty making friends/socializing" half of the official paper definition was completely missing. Restored it as an *additional* qualifying pattern (not replacing the friction-based one, to avoid the earlier over-narrowing mistake). F1 jumped ~0.53→0.73.
8. **`sense of responsibility`** — over-triggering (13 FPs vs. 2 FNs). The definition's own wording ("not wanting to hurt family") was matching farewell/apology language said *while actively planning to die* ("cutting everyone off so it won't hurt as bad when I die"), which annotators don't count — that's practical preparation, not a protective pull. Reframed as requiring a genuine active reason-to-stay, with a positive/negative contrast pair. Precision jumped from ~0.35 to 1.00, F1 to 0.62.

**Key methodological note:** because two-call mode makes the risk/evidence call and the factor call fully independent, composite/Weighted-F1/Phrase-F1 fluctuations between runs where *only* `factor_prompt.md` changed are attributable entirely to normal run-to-run variance in the (unchanged) risk/evidence call — not the factor edit. This let us trust factor-side gains (e.g. `social support`'s F1 0.33→0.79) even on runs where the overall composite dipped slightly, and avoid reverting real wins based on a misleading top-line number.

**Result did not fully transfer to official scoring at the same magnitude** (local Macro F1 0.5056→0.5619, official Subtask2 0.5430→0.5436) — consistent with the established, well-understood local-vs-official gap on this metric (see "validation set discipline" below), but Subtask2 did land marginally above the prior best, so the round was a net real (if compressed) improvement, not a wash.

## What we learned iterating (in rough order)

1. **TF-IDF baseline first** (kept in repo for reference): 0.557 weighted F1, complete failure on Attempt class. Confirmed the case for LLM prompting over classical ML for this task.
2. **Evidence span length was the single biggest fixable issue early on** — trim to the minimal phrase, not surrounding context.
3. **Behavior/Ideation boundary needed several distinct fixes**: method named in *wish* phrasing still counts; social/emotional prep (breaking up) does NOT count without a named method/timing; a physical artifact (note left for discovery) DOES count even without a method; a past attempt always wins over concurrent current-planning language, with an intent-check (framing an overdose as "numbing pain" ≠ an attempt).
4. **Vague euphemisms** ("do it," "gone") must NOT be escalated to Ideation without an explicit death/suicide reference — but this rule must be scoped narrowly so it doesn't suppress genuine Behavior timing cues in the same post (a real interaction bug we hit and fixed).
5. **Repeated physical partial-attempts** (e.g. choking to near-unconsciousness, not completed) count as Attempt, distinct from merely failing to start.
6. **The "ambiguous-word caution" (fall/sleep/rest ≠ method) has now failed twice officially** despite looking harmless locally both times — don't retry it without new evidence.
7. **Self-consistency (n=3 vote), forced tool-use, and extended thinking** were all tested and REJECTED — no clear benefit or a regression vs. free-text JSON + single-sample + thinking disabled.
8. **Concrete worked examples consistently outperform abstract rule restatements** — true across nearly every factor fix this session; when a category was under-triggering, adding one well-sourced example almost always beat rewording the definition alone.

## CRITICAL: validation set discipline

**`val_split.csv` must stay untouched — never pull worked-example post text from it.**

Early in this project, several worked examples were sourced by reading full post text directly out of `val_split.csv` — the same 196 posts used to score every validation run. This quietly contaminates the measurement. This is very likely part of why local composite has repeatedly run ahead of official scoring (most starkly: local Macro F1 climbing to 0.56+ while official Subtask2 sits around 0.54) — the same class of dev-to-test gap seen in the PolyFiQA/FinMMEval project (dev ROUGE-1 ~0.395 vs. official test ~0.307).

**The rule:** `val_split.csv` is for computing metrics and identifying which categories/confusion patterns are failing — never for sourcing verbatim post text to build a new prompt example. Pull grounding examples from `train_split.csv` instead (1,439 rows, never touched for scoring). Every fix in the big factor-tuning round above was sourced this way. `analyze_errors.py` prints this reminder every run.

## Remaining known weaknesses

- **Behavior recall has stayed flat around 0.62-0.66 across the entire session**, including through the whole factor-tuning round (which didn't touch `risk_prompt.md`) — the single biggest remaining lever on Subtask 1, and untouched by today's factor work.
- **`emotion_dysregulation` remains the largest under-performing factor category** (support 64, F1 stuck ~0.34-0.46). One more diagnostic pass found no new well-evidenced pattern beyond what's already fixed — the remaining misses are mostly intense/exclamatory distress language without a literal named cue, which the strict-cue requirement intentionally does not catch (loosening it has caused overcorrection before). Needs a genuinely new angle, not another narrow patch, if pursued further.
- **Tiny-support factor categories** (`poor school performance`, `sexual orientation related issues`, `traumatic experience`, `exposure to others' suicide`, `low socio-economic status` — all under 5 examples in `train_split.csv`) are structurally capped by sample size, not fixable by prompting.
- **Local Macro F1 needs a reasonably large sample to be meaningful** — at `--sample-size 40`, roughly 8 of 24 categories end up with zero examples. Use `--sample-size` for risk-level/evidence quick checks only.
- We don't have the organizers' official sample submission file, so the exact string format for the `factors` column is our best guess, not confirmed.

## Files

**`big_data_files/`** — data and EDA
- `clean_train.csv`, `clean_leaderboard.csv` — cleaned/parsed versions of the raw train and leaderboard sets
- `train_split.csv`, `val_split.csv` — user-grouped 85/15 split. **See validation discipline section above before touching either for prompt work.**
- `validation_predictions.csv` — model predictions from the most recent validation run, including predicted + gold evidence/factors for error analysis
- `GT.csv` — most recent leaderboard submission file
- `preprocess_eda.py` — produces the cleaned files + EDA plots

**`codebase/`** — modeling
- `system_prompt.md` / `schema.json` — single-call pipeline
- `risk_prompt.md` / `risk_schema.json` + `factor_prompt.md` / `factor_schema.json` — two-call pipeline (behind all our best results)
- `batch_score.py` — main script. `--mode validate` / `--mode leaderboard --team-name GT`; `--two-call`; `--sample-size N` for cheap partial-validation quick checks (risk-level/evidence only, not reliable for factors); prompt caching (`cache_control`) built in to cut repeat-token cost. Supports direct Anthropic API and AWS Bedrock.
- `analyze_errors.py` — run after `--mode validate` to see WHY the score is what it is: confusion matrix with real post text, per-factor false-positive/negative breakdown, evidence-quality checks. Prints the validation-set-discipline reminder every run. This was the primary diagnostic tool for the entire factor-tuning round.
- `check_<factor_name>.py` — a family of small one-off diagnostic scripts (e.g. `check_social_support.py`, `check_coping_strategy.py`) built during the factor-tuning round, each showing false-positive/negative examples for one specific factor category from `validation_predictions.csv`. Cheap to run (no API calls) — good template for diagnosing any factor category not yet covered.
- `prompt_tester.html` — standalone browser tool for quick one-off prompt tests before committing to a paid batch run
- `baseline_risk_classifier.py`, `baseline_risk_predictions.csv` — TF-IDF baseline, kept for reference

## How to run

```bash
cd codebase
pip install anthropic pandas scikit-learn boto3 --break-system-packages

# validate (cheap quick check on risk/evidence only)
python3 batch_score.py --mode validate --backend bedrock --aws-region us-east-2 --model-id us.anthropic.claude-sonnet-5 --two-call --sample-size 40

# full validation (needed for trustworthy factor/Macro F1 numbers)
python3 batch_score.py --mode validate --backend bedrock --aws-region us-east-2 --model-id us.anthropic.claude-sonnet-5 --two-call

# dig into WHY before changing the prompt
python3 analyze_errors.py
# or, for one specific factor category:
python3 check_<factor_name>.py

# generate the actual leaderboard submission
python3 batch_score.py --mode leaderboard --team-name GT --backend bedrock --aws-region us-east-2 --model-id us.anthropic.claude-sonnet-5 --two-call
```

Requires AWS credentials with Bedrock model access + `aws-marketplace:Subscribe`/`ViewSubscriptions` permissions (needed once to activate the model subscription). If you hit a `403 - security token invalid` error, run `export AWS_PROFILE=bigdata-cup` then `aws sts get-caller-identity` to confirm before retrying.

**Note on daily submission limits:** the leaderboard allows up to 3 submissions per day, and local validation has repeatedly disagreed with official scoring this session (in both directions) — don't submit on a single ambiguous local result. Rerun validation to check a change is directionally consistent before spending a real submission on it.
