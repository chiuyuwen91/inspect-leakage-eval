"""Validity checks for the dataset itself, no model needed.

If a dumb baseline scores well, the dataset leaks its own answers through surface cues
and a high model score would mean little.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load(path: str | None = None) -> list[dict]:
    path = path or str(Path(__file__).parent / "data" / "cases.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def _body(r: dict) -> str:
    return r["input"].split("\n\n")[0]  # drop the shared question


def word_cue_baseline(rows: list[dict]) -> float:
    """Leave-one-pair-out: train on 11 pairs, test on the held-out pair."""
    pairs = sorted({r["pair"] for r in rows})
    correct = 0
    for held in pairs:
        train = [r for r in rows if r["pair"] != held]
        test = [r for r in rows if r["pair"] == held]
        vec = TfidfVectorizer(ngram_range=(1, 2))
        x = vec.fit_transform([_body(r) for r in train])
        clf = LogisticRegression(max_iter=1000).fit(x, [r["target"] for r in train])
        pred = clf.predict(vec.transform([_body(r) for r in test]))
        correct += sum(p == r["target"] for p, r in zip(pred, test))
    return correct / len(rows)


def length_baseline(rows: list[dict]) -> float:
    """Predict 'leak' for the shorter text in each pair. Chance is 0.5."""
    correct = 0
    for p in {r["pair"] for r in rows}:
        a, b = [r for r in rows if r["pair"] == p]
        if len(_body(a)) == len(_body(b)):
            correct += 1  # a tie carries no signal: half credit for both items
            continue
        shorter, longer = (a, b) if len(_body(a)) < len(_body(b)) else (b, a)
        correct += (shorter["target"] == "leak") + (longer["target"] == "ok")
    return correct / len(rows)


def label_balance(rows: list[dict]) -> float:
    return float(np.mean([r["target"] == "leak" for r in rows]))


if __name__ == "__main__":
    rows = load()
    print("n =", len(rows))
    print("label balance (share leak):", label_balance(rows))
    print("word-cue baseline (leave-one-pair-out):", round(word_cue_baseline(rows), 3))
    print("length baseline:", round(length_baseline(rows), 3))
