"""Sorting the supervisors' notes into reasons for the extra hours.

2,117 notes, three languages, heavy typos. Underneath they are about 35 real
sentences wearing ~700 disguises: names swapped in, times written six different
ways, and a typo injected almost everywhere.

Three independent labellers live here, which is what makes the validation
possible -- two methods that disagree tell you where the hard cases are:

  `label_by_template`  the LLM's reading. Each template below was read and
                       judged by a language model; typo variants are matched to
                       their nearest template. Committed to the repo, so every
                       decision is auditable rather than hidden in an API call.

  `label_by_rules`     keyword patterns written to generalise, deliberately NOT
                       derived from the template list. The independent check.

  NoteClassifier       character n-gram model trained on the template labels.
                       This is what runs in production: no API key, no network,
                       deterministic, and it handles typos it has never seen.

The cost split (who pays) is a separate axis, derived from the category. A note
gets a reason AND a bill, and conflating the two loses the disputed cases.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------

NOTHING_REPORTED = "nothing_reported"
CLIENT_REQUESTED = "client_requested"
DISPUTED_CLIENT_REQUEST = "disputed_client_request"
RELIEF_NO_SHOW = "relief_no_show"
COLLEAGUE_ABSENT = "colleague_absent"
LATE_HANDOVER = "late_handover"
EQUIPMENT_FAILURE = "equipment_failure"
UNCLEAR = "unclear"

CATEGORIES = [
    NOTHING_REPORTED,
    CLIENT_REQUESTED,
    DISPUTED_CLIENT_REQUEST,
    RELIEF_NO_SHOW,
    COLLEAGUE_ABSENT,
    LATE_HANDOVER,
    EQUIPMENT_FAILURE,
    UNCLEAR,
]

# Who carries the cost. This is the answer to "which overtime is fixable?".
#
# `disputed_client_request` is deliberately billed to the company: the paperwork
# says the client asked, but the note itself names a rostering failure as the
# real cause. Filing those as billable would tell the contract manager to relax
# about the exact thing costing him money.
CLIENT_PAYS = "client_pays"
COMPANY_PAYS = "company_pays"
NO_COST_SIGNAL = "no_signal"

COST_OF_CATEGORY = {
    NOTHING_REPORTED: NO_COST_SIGNAL,
    UNCLEAR: NO_COST_SIGNAL,
    CLIENT_REQUESTED: CLIENT_PAYS,
    DISPUTED_CLIENT_REQUEST: COMPANY_PAYS,
    RELIEF_NO_SHOW: COMPANY_PAYS,
    COLLEAGUE_ABSENT: COMPANY_PAYS,
    LATE_HANDOVER: COMPANY_PAYS,
    EQUIPMENT_FAILURE: COMPANY_PAYS,
}

# What a manager could actually do about each one.
FIX_FOR_CATEGORY = {
    RELIEF_NO_SHOW: "Relief roster — shifts are not being handed over.",
    COLLEAGUE_ABSENT: "Absence cover — no standby for sick and family leave.",
    LATE_HANDOVER: "Handover process — keys and OB book holding shifts open.",
    EQUIPMENT_FAILURE: "Equipment maintenance — breakdowns forcing manual work.",
    DISPUTED_CLIENT_REQUEST: "Billing dispute — logged as client-approved, caused by a no-show.",
    CLIENT_REQUESTED: "Billable to the client. Check it was invoiced.",
}


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z0-9\s]")
_SPACE = re.compile(r"\s+")

# Times are written six ways for the same thing: 6, six, 6am, 0600, 06h00, 06 00.
_TIME = re.compile(
    r"\b(0?6\s?h\s?00|0?6\s?00|0600|06h00|6\s?am|six|0?6)\b"
)


def normalise(text: str | float | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace and time formats."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = _PUNCT.sub(" ", str(text).lower())
    s = _SPACE.sub(" ", s).strip()
    return _TIME.sub("TIME", s)


def strip_names(text: str, surnames: set[str]) -> str:
    """Replace colleague surnames with a placeholder.

    `covering Wyk post` and `covering Ndlovu post` are one sentence, not two.
    isiZulu prefixes the surname (uMahlangu), so we strip a leading u- too.
    """
    out = []
    for word in text.split():
        bare = word[1:] if word.startswith("u") and word[1:] in surnames else word
        out.append("NAME" if bare in surnames else word)
    return " ".join(out)


# --------------------------------------------------------------------------
# The LLM's labels
# --------------------------------------------------------------------------
#
# Each entry was read and judged by a language model, then committed here so the
# decision is inspectable. Typo variants are matched to their nearest template at
# runtime, so `srubber broke down` lands on the `scrubber` entry.
#
# Where the judgement was genuinely difficult it is noted inline.

TEMPLATES: list[tuple[str, str]] = [
    # ---- nothing useful in the note -------------------------------------
    ("", NOTHING_REPORTED),
    ("ok", NOTHING_REPORTED),
    ("okay", NOTHING_REPORTED),
    ("fine", NOTHING_REPORTED),
    ("all fine", NOTHING_REPORTED),
    ("all good", NOTHING_REPORTED),
    ("all quiet", NOTHING_REPORTED),
    ("sharp", NOTHING_REPORTED),
    ("quiet shift", NOTHING_REPORTED),
    ("ntr", NOTHING_REPORTED),
    ("nothing to report", NOTHING_REPORTED),
    ("no incidents", NOTHING_REPORTED),
    ("no issues on site", NOTHING_REPORTED),
    ("as per normal", NOTHING_REPORTED),
    ("akukho lutho", NOTHING_REPORTED),            # isiZulu: nothing
    ("kwakuhle", NOTHING_REPORTED),                # isiZulu: it was fine
    ("niks om te rapporteer nie", NOTHING_REPORTED),  # Afrikaans: nothing to report
    ("alles reg", NOTHING_REPORTED),               # Afrikaans: all good

    # ---- the client asked for the hours, and they pay -------------------
    ("requested by centre management for load in signed off", CLIENT_REQUESTED),
    ("client requested deep clean before the audit approved", CLIENT_REQUESTED),
    ("client asked us to stay for the delivery ok d by centre mgmt", CLIENT_REQUESTED),
    ("client wanted extra man on the gate for the event approved", CLIENT_REQUESTED),
    ("client asked for extra patrol after the break in tuesday", CLIENT_REQUESTED),
    ("stocktake ran over client asked us to remain they know they pay for it",
     CLIENT_REQUESTED),
    ("extra patrol per client email approved by office", CLIENT_REQUESTED),
    ("extra hours approved by client for the event setup", CLIENT_REQUESTED),
    ("additional cover requested by site manager signed off", CLIENT_REQUESTED),
    ("centre manager requested additional cover for stocktake", CLIENT_REQUESTED),
    ("klient het ekstra ure gevra vir stocktake", CLIENT_REQUESTED),  # Afrikaans

    # ---- claims client approval, but names a real operational cause -----
    # The most consequential category. Keyword matching sees "client signed"
    # and files these as billable, which is exactly backwards.
    ("client signed for the extra hours but real reason is relief no show again",
     DISPUTED_CLIENT_REQUEST),

    # ---- supervisor does not know if it was authorised ------------------
    # Genuinely undecidable from the note. Not client_requested (unconfirmed),
    # not a failure either. Honest answer is that we cannot tell.
    ("client says stay till TIME dont know if office approved", UNCLEAR),

    # ---- the incoming shift never arrived -------------------------------
    ("next shift guard did not pitch had to cover", RELIEF_NO_SHOW),
    ("control room says relief coming nobody came", RELIEF_NO_SHOW),
    ("no replacement sent ngicela sort this out", RELIEF_NO_SHOW),  # mixed isiZulu
    ("relief no show again", RELIEF_NO_SHOW),
    ("waited for relief nobody came through", RELIEF_NO_SHOW),
    ("no relief stayed someone must please sort the roster", RELIEF_NO_SHOW),
    ("double shift because relief never pitched", RELIEF_NO_SHOW),
    ("still on site relief was suppose to come TIME", RELIEF_NO_SHOW),
    ("relief only arrived TIME stayed until then", RELIEF_NO_SHOW),
    ("relief never arrived stayed on till TIME", RELIEF_NO_SHOW),
    ("aflos het nie opgedaag nie moes aanbly", RELIEF_NO_SHOW),  # Afrikaans
    ("next shift akafikanga ngihlale kuze kube TIME", RELIEF_NO_SHOW),  # isiZulu

    # ---- a named colleague was absent; this person covered --------------
    # Boundary note: "covering NAME post no show no call" is about covering
    # SOMEONE ELSE'S post, so it sits here rather than under relief_no_show,
    # which is about the person due to take over at the end of the shift.
    ("covering NAME post no show no call", COLLEAGUE_ABSENT),
    ("covering for NAME booked off sick", COLLEAGUE_ABSENT),
    ("NAME didnt come in covered the post", COLLEAGUE_ABSENT),
    ("NAME absent took her rounds as well", COLLEAGUE_ABSENT),
    ("NAME off sick again covered", COLLEAGUE_ABSENT),
    ("double duty today NAME on family responsibility leave", COLLEAGUE_ABSENT),
    ("stood in for NAME", COLLEAGUE_ABSENT),
    ("took NAME shift as well 2 posts 1 guard", COLLEAGUE_ABSENT),
    ("covered for NAME again 3rd time this month", COLLEAGUE_ABSENT),
    ("worked through NAME at the clinic", COLLEAGUE_ABSENT),
    ("gedek vir NAME siek gemeld", COLLEAGUE_ABSENT),        # Afrikaans
    ("uNAME akezanga namhlanje ngimele yena", COLLEAGUE_ABSENT),  # isiZulu

    # ---- the shift ran over because the handover dragged ----------------
    ("late handover waiting on paperwork", LATE_HANDOVER),
    ("handover late again keys missing", LATE_HANDOVER),
    ("shift handover delayed by 30 min", LATE_HANDOVER),
    ("waited 25 min for handover ob book not signed", LATE_HANDOVER),
    ("oorhandiging was laat gewag vir sleutels", LATE_HANDOVER),  # Afrikaans

    # ---- something broke and the work took longer -----------------------
    ("generator fault stayed to monitor", EQUIPMENT_FAILURE),
    ("buffer machine kaput did the floor by hand", EQUIPMENT_FAILURE),
    ("scrubber broke down had to do the floor manually", EQUIPMENT_FAILURE),
    ("lift out of order everything carried up stairs took long", EQUIPMENT_FAILURE),
    ("machine down again took twice as long", EQUIPMENT_FAILURE),
    ("gate motor failed manned it by hand till TIME", EQUIPMENT_FAILURE),
    ("masjien is stukkend alles met die hand gedoen", EQUIPMENT_FAILURE),  # Afrikaans
]

# Below this similarity to the nearest template, we decline to guess.
TEMPLATE_MATCH_THRESHOLD = 0.62


@dataclass
class TemplateLabeller:
    """Match a note to the nearest template the LLM has already judged."""

    surnames: set[str]
    templates: list[tuple[str, str]]

    def _skeleton(self, text: str) -> str:
        return strip_names(normalise(text), self.surnames)

    def label_one(self, text: str) -> tuple[str, float, str]:
        skeleton = self._skeleton(text)
        if not skeleton:
            return NOTHING_REPORTED, 1.0, ""

        best_score, best_cat, best_template = 0.0, UNCLEAR, ""
        for template, category in self.templates:
            score = difflib.SequenceMatcher(None, skeleton, template).ratio()
            if score > best_score:
                best_score, best_cat, best_template = score, category, template

        if best_score < TEMPLATE_MATCH_THRESHOLD:
            return UNCLEAR, best_score, best_template
        return best_cat, best_score, best_template

    def label(self, notes: pd.Series) -> pd.DataFrame:
        """Label a series of notes, caching by skeleton so typo families
        resolve once rather than once per row."""
        skeletons = notes.map(self._skeleton)
        cache = {s: self.label_one(s) for s in skeletons.unique()}
        rows = [cache[s] for s in skeletons]
        return pd.DataFrame(rows, columns=["category", "match_score", "matched_template"],
                            index=notes.index)


_PLACEHOLDER = re.compile(r"\b(name|time)\b")


def _normalise_template(text: str) -> str:
    """Normalise a template the same way as a note, keeping the placeholders.

    `normalise` lowercases, which would turn NAME and TIME into ordinary words,
    so they are restored afterwards. Both sides must end up in the same shape or
    the similarity match is comparing different alphabets.
    """
    return _PLACEHOLDER.sub(lambda m: m.group(1).upper(), normalise(text))


def build_template_labeller(employees: pd.DataFrame) -> TemplateLabeller:
    surnames = {
        str(name).split()[-1].lower()
        for name in employees["full_name"].dropna()
        if str(name).split()
    }
    templates = [(_normalise_template(text), category) for text, category in TEMPLATES]
    return TemplateLabeller(surnames=surnames, templates=templates)


# --------------------------------------------------------------------------
# The independent rules labeller
# --------------------------------------------------------------------------
#
# Written from the categories, not from the template list, so that disagreement
# with the template labeller is informative rather than circular. Order matters:
# the disputed pattern must be tested before the plain client pattern.

RULES: list[tuple[str, str]] = [
    # A claim of client approval AND a stated operational cause. Checked first.
    (DISPUTED_CLIENT_REQUEST,
     r"(?:client|klient|kli.nt).*(?:but|maar|real reason|regte rede)"
     r"|(?:but|real reason).*(?:no.?show|relief|aflos)"),

    # Explicit uncertainty about authorisation.
    (UNCLEAR, r"(?:dont|don't|dont|do not|nie)\s*k?n?o?w?.*(?:approv|apprv|authoris|gemagtig)"
              r"|not sure.*(?:approv|authoris)"),

    (NOTHING_REPORTED,
     r"^(?:ok|okay|fine|all fine|all good|all quiet|sharp|quiet shift|ntr|nothing to report"
     r"|no incidents|no issues on site|as per normal|akukho lutho|kwakuhle"
     r"|niks om te rapporteer nie|alles reg|n/?a|none|nil|[-.\s])*$"),

    # Extra staffing the client asked for and signed off. Must be tested before
    # `colleague_absent`, because "additional COVER requested by site manager,
    # signed off" contains the word "cover" and would otherwise be filed as
    # somebody standing in for an absent colleague. Found by comparing against
    # the template labeller; it is an ordering error, not a tuning tweak.
    (CLIENT_REQUESTED,
     r"(?:requested|rquested|request|gevra|asked|wanted|per client email)"
     r".*(?:sign(?:ed)? off|siged off|approv|appoved|approoved|ok.?d by)"
     r"|(?:centre|cnetre|site) m(?:gr|anager|gmt|ngr).*(?:request|cover|stocktake)"
     r"|additional cover request"),

    (EQUIPMENT_FAILURE,
     r"machine|masjien|scrubber|buffer|generator|lift|gate motor|compactor"
     r"|kaput|stukkend|broke|broken|breakdown|out of order|fault|failed|down agn|down again"),

    (LATE_HANDOVER,
     r"handover|hand over|hanodver|handoevr|oorhandig|ob book|paperwork"
     r"|keys missing|sleutels"),

    # Relief = the person due to take over. Tested before colleague_absent,
    # since both mention someone not arriving.
    (RELIEF_NO_SHOW,
     r"relief|relieef|relif|aflos|replacement|next shift|nobody came|nobdoy came"
     r"|no.?show no.?call$|did not pitch|didnt pitch|never pitched|akafikanga"
     r"|sort the roster|still on site"),

    (COLLEAGUE_ABSENT,
     r"cover|covreed|coevred|gedek|stood in|took .* shift|2 posts|double duty"
     r"|absent|off sick|booked off|siek|clinic|family responsibility"
     r"|akezanga|ngimele|took her rounds|took his rounds"),

    (CLIENT_REQUESTED,
     r"client|klient|kli.nt|clinet|clent|cliennt|centre m|cnetre|site m"
     r"|approved|appoved|approoved|signed off|siged off|ok.?d by|per client email"
     r"|requested|rquested|gevra"),
]


def label_by_rules(notes: pd.Series) -> pd.Series:
    text = notes.map(normalise)
    out = pd.Series(UNCLEAR, index=notes.index, dtype=object)
    assigned = pd.Series(False, index=notes.index)
    for category, pattern in RULES:
        hit = text.str.contains(pattern, regex=True, na=False) & ~assigned
        out[hit] = category
        assigned |= hit
    return out


# --------------------------------------------------------------------------
# The production classifier
# --------------------------------------------------------------------------
#
# Distillation: the language model's judgement is frozen into the committed
# TEMPLATES list, and this model learns from those labels. What ships is a few
# kilobytes of coefficients -- no API key, no network call, no rate limit, and
# the same input always gives the same output.
#
# Character n-grams rather than words, because the noise here is spelling.
# `handover` and `hanodver` share almost every 3-gram, so a character model sees
# through typos that a word model treats as unrelated vocabulary. It also means
# the model generalises to misspellings that appear for the first time in next
# week's export.

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


class NoteClassifier:
    """Character n-gram classifier trained on the LLM's labels."""

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=3000,
                C=5.0,
                class_weight="balanced",
            )),
        ])
        self.classes_: list[str] = []

    def fit(self, notes: pd.Series, labels: pd.Series) -> "NoteClassifier":
        text = notes.map(normalise)
        keep = labels.notna()
        self.pipeline.fit(text[keep], labels[keep])
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, notes: pd.Series) -> pd.Series:
        return pd.Series(
            self.pipeline.predict(notes.map(normalise)),
            index=notes.index,
            dtype=object,
        )

    def predict_confidence(self, notes: pd.Series) -> pd.Series:
        proba = self.pipeline.predict_proba(notes.map(normalise))
        return pd.Series(proba.max(axis=1), index=notes.index)


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------

# Below this the classifier's answer is not trusted on a note the template
# labeller could not place. Chosen from the novel-sentence probe: correct
# answers on unseen wordings scored 0.90, wrong ones mostly 0.25-0.49.
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.60


def classify_notes(
    export,
    train_classifier: bool = True,
    use_llm: bool = True,
) -> pd.DataFrame:
    """Label every note, through the cascade.

    Each gate handles only what the one above could not:

      1. Template match (>= 0.62 similarity) -- a sentence we already know.
         Covers 100% of the shipped corpus, and is fully auditable.
      2. Classifier (>= 0.60 confidence) -- a new wording of a known cause.
         Catches things like "my relief bunked" that templates score at 0.52.
      3. LLM, if a key is present -- a genuinely new cause. Cached to disk.
      4. `unclear`, surfaced on the dashboard for a human to look at.

    Gate 4 is the point. Left to itself the classifier files anything unfamiliar
    as `nothing_reported` with no warning, which would record a real operational
    failure as a normal shift.
    """
    notes = export["shift_notes"].copy()
    labeller = build_template_labeller(export["employees"])

    out = notes[["shift_id", "logged_by", "note"]].copy()
    template = labeller.label(notes["note"])
    out["category"] = template["category"]
    out["match_score"] = template["match_score"].round(3)
    out["rules_category"] = label_by_rules(notes["note"])
    out["methods_agree"] = out["category"] == out["rules_category"]
    out["source"] = "template"
    out["suggested_category"] = None

    unplaced = out["match_score"] < TEMPLATE_MATCH_THRESHOLD

    if train_classifier:
        # Train ONLY on notes the template placed confidently. Including the
        # unplaced ones would teach the model that they are `unclear`, and it
        # would then predict `unclear` for them with high confidence -- a
        # self-fulfilling loop that silently disables gate 2.
        placed = ~unplaced
        model = NoteClassifier().fit(notes["note"][placed], out["category"][placed])
        out["model_category"] = model.predict(notes["note"])
        out["model_confidence"] = model.predict_confidence(notes["note"]).round(3)

        # Gate 2: the template could not place it, but the classifier is sure.
        rescued = unplaced & (out["model_confidence"] >= CLASSIFIER_CONFIDENCE_THRESHOLD)
        out.loc[rescued, "category"] = out.loc[rescued, "model_category"]
        out.loc[rescued, "source"] = "classifier"

    # Gate 3: still unplaced. Ask the LLM if we can, else leave it `unclear`.
    stranded = out["match_score"] < TEMPLATE_MATCH_THRESHOLD
    if train_classifier:
        stranded &= out["source"] != "classifier"

    if stranded.any():
        from .llm import build_labeller

        llm = build_labeller(enabled=use_llm)
        texts = [normalise(t) for t in out.loc[stranded, "note"]]
        for idx, result in zip(out.index[stranded], llm.label(texts)):
            known = result.category in CATEGORIES
            out.at[idx, "category"] = result.category if known else UNCLEAR
            out.at[idx, "suggested_category"] = (
                result.suggested_name or (None if known else result.category)
            )
            out.at[idx, "source"] = result.source

    out["who_pays"] = out["category"].map(COST_OF_CATEGORY).fillna(NO_COST_SIGNAL)
    out["suggested_fix"] = out["category"].map(FIX_FOR_CATEGORY)
    return out


def to_note_classifications_csv(classified: pd.DataFrame) -> pd.DataFrame:
    """The submission file: exactly `shift_id, category, note`."""
    out = classified[["shift_id", "category", "note"]].copy()
    out["note"] = out["note"].fillna("")
    return out
