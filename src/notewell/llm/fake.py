"""The offline `fake` provider: a deterministic, rule-based extractive scribe.

It reads the same masked prompt a cloud model would get and answers with the same JSON shape, so the whole pipeline
(masking, egress check, grounding, guards, review UI, FHIR export, MCP) runs in tests, CI and the demo with no key and
no network. It is also the evaluation's baseline: it copies sentences instead of understanding them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from notewell.guards.text import AFFIRM, DENY
from notewell.kb import drug_aliases, drug_pattern, find_concepts
from notewell.llm.base import Completion, Message

_LINE = re.compile(r"^\[(\d+)\]\s+(clinician|patient|caregiver|unknown):\s+(.*)$", re.MULTILINE)
_SENT = re.compile(r"[^.?!]+[.?!]?")
_VITALS = re.compile(
    r"\b(?:blood pressure|pulse|heart rate|temperature|temp|oxygen|o2|weight|weigh|bmi|respiratory rate|a1c|score|"
    r"phq|gad|audit|ldl|tsh|ejection fraction|ecg|ekg|rapid strep|urine|dip|level|results?|came back)\b",
    re.I,
)
_EXAM = re.compile(
    r"\b(?:i see|i can see|looks|look|sounds|sound|clear|tender|tenderness|swelling|swollen|red|bulging|lesion|mole|"
    r"rash|plaques?|exam|on exam|lungs|heart|abdomen|eardrum|throat|skin|border|millimeters?|centimeters?|normal|"
    r"no murmur|regular|irregular|reflexes|range of motion|strength)\b",
    re.I,
)
_ASSESS = re.compile(
    r"\b(?:i think|this is|this looks like|looks like|consistent with|my impression|diagnosis|so you have|you have|"
    r"it's an?|this sounds like|likely|most likely|i'm confident|it is)\b",
    re.I,
)
_PLAN = re.compile(
    r"\b(?:start|starting|i'd like|i would like|we'll|we will|i'm going to|i am going to|prescrib\w*|refer\w*|"
    r"follow up|follow-up|come back|see you|order|increase|decrease|stop|continue|keep taking|check|schedule|"
    r"recheck|let's|plan|take|use|apply|call us|go to the|avoid|return)\b",
    re.I,
)
_DOSE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(milligrams?|micrograms?|mg|mcg|units|percent|grams?|milliliters?|puffs?)", re.I
)
_FREQ = re.compile(
    r"\b(?:once a day|once daily|twice a day|twice daily|three times a day|four times a day|every \d+ hours|"
    r"every morning|every night|at bedtime|daily|a day|weekly|as needed|nightly)\b",
    re.I,
)
_ALLERGY = re.compile(r"\ballergic to ([a-z][a-z -]+?)(?:[,.]| and | it | they |$)", re.I)
_REACTION = re.compile(r"\b(hives|rash|swelling|vomiting|itching|anaphylaxis|trouble breathing|nausea)\b", re.I)
_PROCEDURE_CUES = [
    (re.compile(r"\b(?:ecg|ekg|electrocardiogram)\b.*\b(?:today|shows|showed|did)\b|\bwe did an? (?:ecg|ekg)\b", re.I), "93000"),
    (re.compile(r"\b(?:phq-9|phq 9|gad-7|gad 7|vanderbilt|audit-c)\b.*\bscore\b|\bscore\b.*\b(?:phq|gad)\b", re.I), "96127"),
    (re.compile(r"\brapid strep\b", re.I), "87880"),
    (re.compile(r"\b(?:urine dip|dipstick|urine test)\b", re.I), "81002"),
    (re.compile(r"\bnebulizer\b.*\b(?:now|today|right now)\b", re.I), "94640"),
    (re.compile(r"\b(?:freeze|freezing|cryotherapy|liquid nitrogen)\b.*\bwarts?\b|\bwarts?\b.*\b(?:freeze|cryo)", re.I), "17110"),
    (re.compile(r"\bshave biopsy\b", re.I), "11102"),
    (re.compile(r"\bpunch biopsy\b", re.I), "11104"),
    (re.compile(r"\bflu shot\b.*\btoday\b|\bflu vaccine\b.*\btoday\b", re.I), "90686"),
]


def _transcript(prompt: str) -> list[tuple[int, str, str]]:
    return [(int(m.group(1)), m.group(2), m.group(3).strip()) for m in _LINE.finditer(prompt)]


def _third_person(sentence: str) -> str:
    s = sentence.strip()
    replacements = [
        (r"\bI've been\b", "has been"), (r"\bI have been\b", "has been"), (r"\bI've\b", "has"),
        (r"\bI have\b", "has"), (r"\bI'm\b", "is"), (r"\bI am\b", "is"), (r"\bI was\b", "was"),
        (r"\bI get\b", "gets"), (r"\bI take\b", "takes"), (r"\bI feel\b", "feels"), (r"\bI think\b", "thinks"),
        (r"\bI don't\b", "does not"), (r"\bI do\b", "does"), (r"\bI\b", "they"), (r"\bmy\b", "their"),
        (r"\bMy\b", "Their"), (r"\bme\b", "them"),
    ]  # fmt: skip
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)
    s = re.sub(r"^(?:So|Well|Um|Uh|Yeah|Yes|Okay|Oh|And|But),?\s+", "", s, flags=re.I)
    return s[:1].upper() + s[1:] if s else s


def _ev(turn: int, sentence: str) -> dict[str, Any]:
    words = sentence.split()
    return {"turn": turn, "quote": " ".join(words[:18])}


def draft_note(prompt: str) -> dict[str, Any]:
    lines = _transcript(prompt)
    sections: dict[str, list[dict[str, Any]]] = {"subjective": [], "objective": [], "assessment": [], "plan": []}
    seen: set[str] = set()
    meds: dict[str, dict[str, Any]] = {}
    allergies: list[dict[str, Any]] = []

    def add(section: str, text: str, evidence: list[dict[str, Any]]) -> None:
        key = text.lower()
        if text and key not in seen and len(text.split()) >= 2:
            seen.add(key)
            sections[section].append({"text": text, "evidence": evidence})

    for i, (turn, role, text) in enumerate(lines):
        prev = lines[i - 1] if i > 0 else None
        for sm in _SENT.finditer(text):
            sentence = sm.group(0).strip()
            if not sentence:
                continue
            if role == "clinician":
                if sentence.endswith("?"):
                    continue
                if _ASSESS.search(sentence) and not _VITALS.search(sentence):
                    add("assessment", sentence.rstrip("."), [_ev(turn, sentence)])
                elif _VITALS.search(sentence) or (_EXAM.search(sentence) and re.search(r"\d|no |normal|clear", sentence, re.I)):
                    add("objective", sentence.rstrip("."), [_ev(turn, sentence)])
                elif _PLAN.search(sentence):
                    add("plan", sentence.rstrip("."), [_ev(turn, sentence)])
            else:
                if prev and prev[1] == "clinician" and prev[2].strip().endswith("?"):
                    question = prev[2]
                    concepts = find_concepts(question.split(".")[-1])
                    if concepts and DENY.match(text) and sm.start() == 0:
                        for _, s, e in concepts:
                            add("subjective", f"Denies {question.split('.')[-1][s:e].lower()}", [_ev(prev[0], question), _ev(turn, sentence)])
                        continue
                    if concepts and AFFIRM.match(text) and sm.start() == 0 and len(sentence.split()) <= 3:
                        for _, s, e in concepts:
                            add("subjective", f"Reports {question.split('.')[-1][s:e].lower()}", [_ev(prev[0], question), _ev(turn, sentence)])
                        continue
                if find_concepts(sentence) or re.search(r"\b\d+\s*(?:days?|weeks?|months?|years?)\b|\bsince\b", sentence):
                    add("subjective", _third_person(sentence).rstrip("."), [_ev(turn, sentence)])
            for m in drug_pattern().finditer(sentence):
                name = drug_aliases()[m.group(1).lower()]
                window = sentence[m.start() : m.start() + 80]
                dose = _DOSE.search(window) or _DOSE.search(sentence)
                if name in meds and not dose:
                    continue
                status = "active"
                if role == "clinician" and re.search(r"\b(start|prescrib|begin|try)\w*", sentence, re.I):
                    status = "new"
                elif re.search(r"\b(increase|raise|bump|up the|decrease|lower|reduce|change)\w*", sentence, re.I):
                    status = "changed"
                elif re.search(r"\b(stop|stopped|quit|discontinu)\w*", sentence, re.I):
                    status = "stopped"
                freq = _FREQ.search(sentence)
                meds[name] = {
                    "name": name,
                    "dose": f"{dose.group(1)} {dose.group(2)}" if dose else "",
                    "frequency": freq.group(0) if freq else "",
                    "route": "",
                    "status": status,
                    "evidence": [_ev(turn, sentence)],
                }
            am = _ALLERGY.search(sentence)
            if am and not re.search(r"\b(no|not)\b", sentence[: am.start()], re.I):
                reaction = _REACTION.search(text)
                allergies.append(
                    {"substance": am.group(1).strip(), "reaction": reaction.group(1) if reaction else "", "evidence": [_ev(turn, sentence)]}
                )
    plan_lines = [p["text"] for p in sections["plan"]][:6]
    avs = "Thank you for coming in today. Here is what we talked about. " + " ".join(
        re.sub(r"\bwe'll\b", "we will", p, flags=re.I).rstrip(".") + "." for p in plan_lines
    )
    if not plan_lines:
        avs += "Please follow the plan we discussed and call the clinic with any questions."
    return {**sections, "medications": list(meds.values()), "allergies": allergies, "after_visit_summary": avs}


def suggest_codes(prompt: str) -> dict[str, Any]:
    note_part = prompt.split("Candidate ICD-10-CM codes", 1)[0]
    assessment = " ".join(re.findall(r"^A: (.*)$", note_part, re.MULTILINE)) or note_part
    cand_block = prompt.split("Candidate ICD-10-CM codes retrieved from the official code list for this note:", 1)
    candidates: list[tuple[str, str]] = []
    if len(cand_block) > 1:
        for line in cand_block[1].split("\n\n", 1)[0].strip().splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2 and re.match(r"^[A-Z]\d", parts[0]):
                candidates.append((parts[0], parts[1]))
    words = set(re.findall(r"[a-z]{4,}", assessment.lower()))
    scored = sorted(
        candidates,
        key=lambda c: (-len(words & set(re.findall(r"[a-z]{4,}", c[1].lower()))), candidates.index(c)),
    )
    icd = [{"code": code, "rationale": f"Matches the assessment: {title}", "evidence": []} for code, title in scored[:3]]
    new = "Visit type: new" in prompt
    rx = re.search(r"\b(start|increase|decrease|prescrib|switch|stop)\w*", note_part, re.I)
    level = "4" if rx else "3"
    em = {"code": f"9920{level}" if new else f"9921{level}", "rationale": "Rule-based: prescription drug management" if rx else "Rule-based: low complexity"}
    transcript = prompt.split("Transcript:", 1)[-1]
    procedures = []
    for pattern, code in _PROCEDURE_CUES:
        for turn, role, text in _transcript(transcript):
            if role == "clinician" and pattern.search(text):
                procedures.append({"code": code, "rationale": "Performed today", "evidence": [_ev(turn, text)]})
                break
    return {"icd10": icd, "em": em, "procedures": procedures}


class FakeScribe:
    label = "fake/fake-scribe"
    is_local = True

    def complete(self, messages: Sequence[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        data = suggest_codes(user) if "medical coder" in system else draft_note(user)
        text = json.dumps(data, ensure_ascii=False)
        return Completion(text=text, model="fake-scribe", input_tokens=len(user) // 4, output_tokens=len(text) // 4)
