"""Builds data/cases.jsonl. 12 matched pairs: each leaky setup has a fixed twin.
All cases are invented for this demo."""
import json

PAIRS = [
    ("repeated_units_random_split",
     "A team predicts which students will drop a course. Each student appears in 15 weekly rows. They shuffle all rows and put 80% in training and 20% in testing. Test AUC is 0.84.",
     "A team predicts which students will drop a course. Each student appears in 15 weekly rows. They assign whole students to training (80%) or testing (20%), so no student is on both sides. Test AUC is 0.84."),
    ("scaler_fit_on_all",
     "A model predicts hospital readmission. The team standardizes every feature with the mean and standard deviation of the full dataset, then splits into training and test sets. Test AUC is 0.79.",
     "A model predicts hospital readmission. The team splits into training and test sets, then standardizes every feature with the mean and standard deviation of the training set and applies the same values to the test set. Test AUC is 0.79."),
    ("feature_selection_on_all",
     "A gene-expression classifier has 20,000 features. The team picks the 50 features most correlated with the label using all rows, then splits into training and test sets. Accuracy is 0.91.",
     "A gene-expression classifier has 20,000 features. The team splits into training and test sets, then picks the 50 features most correlated with the label using the training rows. Accuracy is 0.91."),
    ("lookahead_feature",
     "A team predicts next-week stress for each student. One input feature is the student's average stress score over the whole semester. Test AUC is 0.66.",
     "A team predicts next-week stress for each student. One input feature is the student's average stress score over the previous four weeks. Test AUC is 0.66."),
    ("post_outcome_column",
     "A bank predicts loan default. One input column, collections_status, is filled in by the bank after a loan goes bad. It is used as a feature. Test AUC is 0.97.",
     "A bank predicts loan default. One input column, income_at_application, is recorded when the loan is approved. It is used as a feature. Test AUC is 0.97."),
    ("duplicates_across_split",
     "A classifier sorts support tickets. Many tickets are near copies of the same template. The team splits tickets at random, so copies land in both training and test. Accuracy is 0.93.",
     "A classifier sorts support tickets. Many tickets are near copies of the same template. The team groups near copies together before splitting, so each group lands on one side. Accuracy is 0.93."),
    ("random_time_split",
     "A team forecasts daily sales for a store using 2023 data. They pick a random 80% of days for training and the other 20% of days for testing. MAPE is 8%.",
     "A team forecasts daily sales for a store using 2023 data. They train on January to September and test on October to December. MAPE is 8%."),
    ("tuning_on_test",
     "A team tries 200 hyperparameter settings. They report the best score they saw on the test set. Test accuracy is 0.88.",
     "A team tries 200 hyperparameter settings. They pick the best on a validation set, then report one score on the test set. Test accuracy is 0.88."),
    ("spurious_acquisition_cue",
     "A skin lesion classifier is trained on photos. Most malignant photos have a ruler in the frame and most benign photos do not. Test accuracy is 0.90.",
     "A skin lesion classifier is trained on photos. Both classes were taken with the same camera setup, and a ruler is in the frame for about half of each class. Test accuracy is 0.90."),
    ("target_encoding_on_all",
     "A model predicts delivery delays. Zip code is replaced by the average delay for that zip, computed on all rows including test rows. Test RMSE is 4.1 minutes.",
     "A model predicts delivery delays. Zip code is replaced by the average delay for that zip, computed on training rows only. Test zips use those values. Test RMSE is 4.1 minutes."),
    ("benchmark_contamination",
     "A team tests a language model on a public math benchmark released in 2021. The model's training data was collected in 2023. The team did not check for overlap. Accuracy is 0.76.",
     "A team tests a language model on a public math benchmark released in 2021. The model's training data is public. The team removes any benchmark question found in it before testing. Accuracy is 0.76."),
    ("oversample_before_split",
     "A fraud dataset is imbalanced. The team oversamples the rare class by copying rows, then splits into training and test sets. Recall is 0.82.",
     "A fraud dataset is imbalanced. The team splits into training and test sets, then oversamples the rare class by copying rows in the training set only. Recall is 0.82."),
]

FILLERS = [
    "The data covers two years.",
    "The code is written in Python.",
    "Training takes about an hour.",
    "The team has four people.",
    "Results are averaged over one run.",
    "The data comes from one organization.",
    "Features are stored in a table.",
]


def _insert(text, filler):
    first, rest = text.split(". ", 1)
    return f"{first}. {filler} {rest}"


def balance_length(leak, ok, leak_longer):
    """Add label-neutral filler so length is not a cue. Sign alternates by pair."""
    import itertools
    best = None
    opts = [None] + FILLERS
    for fl, fo in itertools.product(opts, opts):
        a = _insert(leak, fl) if fl else leak
        b = _insert(ok, fo) if fo else ok
        diff = len(a) - len(b)
        if diff == 0 or (diff > 0) != leak_longer:
            continue
        key = (abs(diff), (fl is not None) + (fo is not None))
        if best is None or key < best[0]:
            best = (key, a, b)
    return best[1], best[2]


QUESTION = ("Does this evaluation setup contain data leakage that makes the reported score unreliable?")

rows = []
for i, (cat, leak, ok) in enumerate(PAIRS):
    leak, ok = balance_length(leak, ok, leak_longer=(i % 2 == 0))
    rows.append({"id": f"{cat}-leak", "pair": i, "category": cat, "input": f"{leak}\n\n{QUESTION}", "target": "leak"})
    rows.append({"id": f"{cat}-ok", "pair": i, "category": cat, "input": f"{ok}\n\n{QUESTION}", "target": "ok"})

with open("data/cases.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print(len(rows), "cases")
