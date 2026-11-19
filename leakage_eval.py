"""Inspect task: can a model tell a leaky evaluation setup from its fixed twin?

Synthetic cases only. Run:  inspect eval leakage_eval.py --model <provider/model>
"""
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, system_message

SYSTEM = (
    "You review machine learning evaluation setups. Think briefly, then end your reply with "
    "exactly one line: 'ANSWER: leak' if the reported score is unreliable because of data "
    "leakage, or 'ANSWER: ok' if the setup is sound."
)

_ANSWER = re.compile(r"ANSWER:\s*(leak|ok)\b", re.IGNORECASE)


def parse_answer(text: str) -> str | None:
    """Return the LAST well-formed answer, or None. Never guess from loose text."""
    found = _ANSWER.findall(text or "")
    return found[-1].lower() if found else None


def to_sample(record: dict) -> Sample:
    return Sample(
        id=record["id"],
        input=record["input"],
        target=record["target"],
        metadata={"pair": record["pair"], "category": record["category"]},
    )


@scorer(metrics=[accuracy(), stderr()])
def answer_match():
    async def score(state: TaskState, target: Target) -> Score:
        got = parse_answer(state.output.completion)
        if got is None:
            return Score(value=INCORRECT, answer=None, explanation="no 'ANSWER: leak|ok' line found")
        return Score(
            value=CORRECT if got == target.text else INCORRECT,
            answer=got,
            explanation=f"model said {got}, target {target.text}",
        )

    return score


@task
def leakage_spotting(path: str | None = None):
    path = path or str(Path(__file__).parent / "data" / "cases.jsonl")
    return Task(
        dataset=json_dataset(path, sample_fields=to_sample),
        solver=[system_message(SYSTEM), generate()],
        scorer=answer_match(),
    )
