"""Prompts for the two model steps. Both only ever see the masked transcript (placeholders instead of identifiers)."""

from __future__ import annotations

from collections.abc import Sequence

from notewell.llm.base import Message

NOTE_SYSTEM = """\
You are a careful medical scribe. You turn the transcript of an outpatient visit into a draft SOAP note for the \
clinician to review. The clinician signs the note, so every statement must be traceable to the conversation.

Identifiers in the transcript are replaced by placeholders such as [PATIENT], [NAME_2], [DATE_1], [PHONE_1]. Copy a \
placeholder exactly when you need it; never guess what it stands for. The note rarely needs names at all.

Rules:
1. Use only what was said in the transcript. Never add findings, vitals, doses, history or plans that were not said. \
If something is unclear, leave it out.
2. One fact per statement, short and clinical ("Headaches for 2 weeks, worse on waking"). Use standard clinical \
wording, but do not upgrade what was said (a patient's "a bit tired" is not "severe fatigue").
3. Pertinent negatives are statements too: "Denies chest pain". A negative answer to the clinician's question \
("Any fever?" "No.") means the patient denies it.
4. Family history is not the patient's history: write "Family history: father had MI at 50". Things about other \
people are not about the patient. Hypotheticals and return precautions ("if the rash spreads, come back") go in the \
plan as precautions, never as current findings. When a value is corrected in the conversation, use the corrected one.
5. Sections: subjective = history the patient or caregiver gave (complaint, history of present illness, pertinent \
negatives, relevant history, social history, current medications in words); objective = vitals, exam findings, test \
results and scores the clinician stated; assessment = the clinician's diagnoses or impressions as stated; plan = \
tests, medications started/changed/stopped with doses, referrals, counseling, precautions, follow-up.
6. Evidence for every statement: the turn number(s) and a short verbatim quote (at most 20 words, copied exactly \
from that turn, placeholders included). A negative from a question-and-answer needs both turns.
7. Medications: every medication discussed for the patient, with dose, frequency, route and status \
(active, new, changed, stopped, prn) as said. Allergies with the reaction, or none if the patient said none.
8. After-visit summary: plain language for the patient, second person, about a 6th-grade reading level, \
120-200 words: what we found, what to do (medications with the exact doses from the plan), warning signs, follow-up. \
No new information.

Answer with one JSON object only, no markdown, exactly this shape:
{"subjective": [{"text": "...", "evidence": [{"turn": 4, "quote": "..."}]}],
 "objective": [...], "assessment": [...], "plan": [...],
 "medications": [{"name": "lisinopril", "dose": "20 mg", "frequency": "once daily", "route": "oral", \
"status": "active", "evidence": [{"turn": 7, "quote": "..."}]}],
 "allergies": [{"substance": "penicillin", "reaction": "hives", "evidence": [{"turn": 9, "quote": "..."}]}],
 "after_visit_summary": "..."}"""


def note_messages(*, transcript: str, specialty: str, visit_type: str) -> list[Message]:
    user = (
        f"Visit: {specialty.replace('_', ' ')}, {visit_type} patient.\n"
        "Transcript (turn number, speaker role, words; roles come from diarization and may occasionally be wrong):\n\n"
        f"{transcript}\n\nWrite the JSON note now."
    )
    return [{"role": "system", "content": NOTE_SYSTEM}, {"role": "user", "content": user}]


CODES_SYSTEM = """\
You are a certified outpatient medical coder. From a visit note draft and its transcript, suggest ICD-10-CM \
diagnosis codes, the office E/M code and CPT codes for procedures performed today. A human coder and the clinician \
review every suggestion.

Rules:
1. ICD-10-CM: code only conditions the clinician assessed in this visit, most specific billable code the \
documentation supports, primary (reason for the visit) first, at most 5. Do not code symptoms that are part of a \
confirmed diagnosis, conditions the patient denied, family members' conditions (use a Z code for family history only \
if it was relevant), or hypotheticals. Prefer a code from the candidate list when one fits; a code that is not in the \
list must be a real, billable ICD-10-CM code.
2. E/M: choose by medical decision making under the 2021 AMA office visit guidelines: number and complexity of \
problems, data reviewed/ordered (tests, independent historian, independent interpretation), and risk of management \
(prescription drug management is moderate risk). New patient: 99202-99205; established: 99212-99215. A \
psychotherapy-only visit uses 90791/90832/90834/90837 instead. Explain the level in one sentence.
3. Procedures: only CPT codes from the allowed list, only for services clearly performed today in clinic (tests \
ordered for later are not procedures).
4. Evidence: for each code, the turn number and a short verbatim quote from the transcript.

Answer with one JSON object only, no markdown:
{"icd10": [{"code": "I10", "rationale": "...", "evidence": [{"turn": 12, "quote": "..."}]}],
 "em": {"code": "99214", "rationale": "..."},
 "procedures": [{"code": "93000", "rationale": "...", "evidence": [{"turn": 20, "quote": "..."}]}]}"""


def codes_messages(
    *,
    note_text: str,
    transcript: str,
    visit_type: str,
    candidates: Sequence[tuple[str, str]],
    allowed_cpt: Sequence[tuple[str, str]],
) -> list[Message]:
    cand = "\n".join(f"{code} {title}" for code, title in candidates) or "(none retrieved)"
    cpt = "\n".join(f"{code} {summary}" for code, summary in allowed_cpt)
    user = (
        f"Visit type: {visit_type} patient.\n\nNote draft:\n{note_text}\n\n"
        f"Candidate ICD-10-CM codes retrieved from the official code list for this note:\n{cand}\n\n"
        f"Allowed CPT codes for procedures:\n{cpt}\n\nTranscript:\n{transcript}\n\nWrite the JSON now."
    )
    return [{"role": "system", "content": CODES_SYSTEM}, {"role": "user", "content": user}]


def repair_message(error: str) -> Message:
    return {
        "role": "user",
        "content": f"Your answer could not be parsed as the required JSON object ({error}). "
        "Reply again with only the JSON object, no markdown fences, no comments.",
    }
