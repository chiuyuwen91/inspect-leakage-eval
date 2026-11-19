
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

import baselines
import leakage_eval as L


def test_parse_answer_takes_last_wellformed_line():
    assert L.parse_answer("I first thought ANSWER: ok\nbut actually\nANSWER: leak") == "leak"


def test_parse_answer_rejects_loose_text():
    assert L.parse_answer("this looks like leakage to me") is None
    assert L.parse_answer("ANSWER: maybe") is None
    assert L.parse_answer("") is None


def test_parse_answer_is_case_insensitive_but_exact_word():
    assert L.parse_answer("answer: OK") == "ok"
    assert L.parse_answer("ANSWER: okay") is None  # 'okay' is not 'ok'


def test_dataset_shape():
    rows = baselines.load()
    assert len(rows) == 24
    assert len({r["id"] for r in rows}) == 24
    assert baselines.label_balance(rows) == 0.5
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], []).append(r["target"])
    assert all(sorted(v) == ["leak", "ok"] for v in pairs.values())


def test_dataset_does_not_leak_its_answer_through_length():
    assert abs(baselines.length_baseline(baselines.load()) - 0.5) < 0.1


def test_word_cue_baseline_stays_below_threshold():
    # Not chance (0.5): small data and the fixed twins naturally say "training set only".
    # This guards against making it worse.
    assert baselines.word_cue_baseline(baselines.load()) <= 0.70


def _mock_output(text):
    out = ModelOutput.from_content("mockllm/model", text)
    # Set usage so the mock never calls the tokenizer (which downloads a file).
    out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    return out


def _run(outputs):
    model = get_model("mockllm/model", custom_outputs=[_mock_output(o) for o in outputs])
    log = inspect_eval(L.leakage_spotting(), model=model, display="none", log_dir="/tmp/inspect_test_logs")[0]
    return log.results.scores[0].metrics["accuracy"].value


def test_oracle_scores_one():
    rows = baselines.load()
    assert _run([f"reasoning\nANSWER: {r['target']}" for r in rows]) == 1.0


def test_always_leak_scores_chance():
    assert _run(["ANSWER: leak"] * 24) == 0.5


def test_unparseable_scores_zero():
    assert _run(["I think it is fine."] * 24) == 0.0


def test_loads_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(L.leakage_spotting().dataset) == 24
    assert len(baselines.load()) == 24
