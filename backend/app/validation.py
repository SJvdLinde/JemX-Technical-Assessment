"""Checking whether the note sorting is any good.

There is no answer sheet, so the check had to be designed. Three strands:

  1. A hand-labelled sample. 50 notes labelled by hand, without sight of the
     code, and stratified so every category appears rather than the sample
     being three-quarters "nothing reported".

  2. Two independent methods. The template labeller and the keyword rules were
     written separately; where they disagree marks the genuinely hard notes.
     The hand labels then adjudicate the disagreements.

  3. Robustness to unseen noise. Next week's export will carry misspellings
     nobody has seen, so accuracy is measured against progressively corrupted
     text rather than only against the notes we already have.

The sample is stratified, roughly six per category, so it is NOT representative
of the corpus mix. Overall accuracy on it would be misleading, so per-category
recall is reweighted back to true corpus proportions -- see `reweighted_accuracy`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from . import config as cfg
from .notes import NoteClassifier, classify_notes, label_by_rules

VALIDATION_SAMPLE = cfg.REPO_ROOT / "labels" / "validation_sample.csv"


def load_hand_labels(path=VALIDATION_SAMPLE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["shift_id"].notna() & df["category"].notna()].copy()
    df["category"] = df["category"].str.strip()
    return df.rename(columns={"category": "truth"})


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an accuracy estimate.

    Matters because a perfect score on a small sample is not evidence of a
    perfect method. 50 out of 50 is consistent with a true accuracy as low as
    ~93%, and quoting "100%" without that caveat would be overclaiming.
    Wilson rather than normal-approximation, which degenerates at p = 1.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
    return (round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3))


@dataclass
class MethodScore:
    name: str
    n: int
    correct: int
    accuracy: float
    per_category: pd.DataFrame
    reweighted: float
    errors: pd.DataFrame

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.n)


def score_method(
    truth: pd.Series,
    predicted: pd.Series,
    corpus_mix: pd.Series,
    name: str,
    notes: pd.Series,
) -> MethodScore:
    """Score one labeller against the hand labels."""
    correct = truth == predicted
    per_category = pd.DataFrame({
        "n": truth.value_counts(),
        "correct": correct.groupby(truth).sum(),
    })
    per_category["recall"] = (per_category["correct"] / per_category["n"]).round(3)

    errors = pd.DataFrame({
        "note": notes, "truth": truth, "predicted": predicted
    }).loc[~correct]

    return MethodScore(
        name=name,
        n=len(truth),
        correct=int(correct.sum()),
        accuracy=round(float(correct.mean()), 3),
        per_category=per_category,
        reweighted=reweighted_accuracy(per_category, corpus_mix),
        errors=errors,
    )


def reweighted_accuracy(per_category: pd.DataFrame, corpus_mix: pd.Series) -> float:
    """Per-category recall, weighted by how common each category really is.

    The sample deliberately over-represents rare categories, so its raw accuracy
    is not the accuracy we would see on the whole corpus. Reweighting by the
    corpus mix gives the honest estimate.
    """
    weights = corpus_mix.reindex(per_category.index).fillna(0)
    if weights.sum() == 0:
        return float("nan")
    weights = weights / weights.sum()
    return round(float((per_category["recall"] * weights).sum()), 3)


def validate(export, path=VALIDATION_SAMPLE) -> dict:
    """Score every labeller against the hand-labelled sample."""
    hand = load_hand_labels(path)

    classified = classify_notes(export, train_classifier=True)
    joined = hand.merge(
        classified[["shift_id", "category", "rules_category", "model_category"]],
        on="shift_id", how="left", validate="one_to_one",
    )
    missing = joined["category"].isna().sum()
    if missing:
        raise ValueError(f"{missing} hand-labelled notes are not in this export.")

    corpus_mix = classified["category"].value_counts(normalize=True)

    scores = {
        name: score_method(joined["truth"], joined[column], corpus_mix, name,
                           joined["note"])
        for name, column in [
            ("template_llm", "category"),
            ("rules", "rules_category"),
            ("classifier", "model_category"),
        ]
    }
    return {
        "n": len(joined),
        "scores": scores,
        "joined": joined,
        "corpus_mix": corpus_mix,
    }


def disagreement_audit(export, path=VALIDATION_SAMPLE) -> pd.DataFrame:
    """Where the two independent methods disagree, who was right?

    This is the part worth reading. Agreement alone proves nothing -- two methods
    can share a blind spot. The hand labels adjudicate.
    """
    hand = load_hand_labels(path)
    classified = classify_notes(export)
    joined = hand.merge(
        classified[["shift_id", "category", "rules_category"]],
        on="shift_id", how="left",
    )
    conflict = joined[joined["category"] != joined["rules_category"]].copy()
    conflict["template_right"] = conflict["category"] == conflict["truth"]
    conflict["rules_right"] = conflict["rules_category"] == conflict["truth"]
    return conflict[[
        "shift_id", "note", "truth", "category", "rules_category",
        "template_right", "rules_right",
    ]]


def typo_robustness(export, edits=(0, 2, 4, 6, 10), seed: int = 0) -> pd.DataFrame:
    """Accuracy as text degrades, for the classifier and the rules.

    Requirement 4 is the reason this exists: next week's export will contain
    misspellings that are not in our training data.
    """
    from sklearn.metrics import accuracy_score

    notes = export["shift_notes"]["note"].fillna("")
    labels = classify_notes(export, train_classifier=False)["category"]
    rng = random.Random(seed)

    def corrupt(text: str, n: int) -> str:
        chars = list(text)
        for _ in range(n):
            if len(chars) < 4:
                break
            i = rng.randrange(len(chars))
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return "".join(chars)

    split = int(len(notes) * 0.7)
    model = NoteClassifier().fit(notes.iloc[:split], labels.iloc[:split])
    held_out, truth = notes.iloc[split:], labels.iloc[split:]

    rows = []
    for n in edits:
        noisy = held_out.map(lambda s, n=n: corrupt(s, n)) if n else held_out
        rows.append({
            "edits_per_note": n,
            "classifier": round(accuracy_score(truth, model.predict(noisy)), 3),
            "rules": round(accuracy_score(truth, label_by_rules(noisy)), 3),
        })
    return pd.DataFrame(rows)
