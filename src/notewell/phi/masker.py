"""Mask identifiers with stable placeholders before anything leaves the process, and restore them locally afterwards.

The registered patient becomes ``[PATIENT]`` (and the date of birth ``[DOB]``); everything else gets a numbered
placeholder by type (``[NAME_2]``, ``[PHONE_1]``, ``[DATE_3]``). The same value always gets the same placeholder within a
visit, and partial names are linked to the full name ("Marisol" and "Mrs. Ferreira" are the same ``[PATIENT]``), so the
model can still follow who is who. The mapping (the vault) never leaves the process; it is stored encrypted.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from notewell.models import Patient, Turn
from notewell.phi.detector import PHIDetector, Registered, Span, parse_date

PLACEHOLDER_RE = re.compile(r"\[(PATIENT|DOB|AGE_90_PLUS|(?:NAME|DATE|PHONE|ADDRESS|EMAIL|ID|ORG|URL|SSN)_\d+)\]")
_TYPE_PREFIX = {
    "NAME": "NAME",
    "DATE": "DATE",
    "PHONE": "PHONE",
    "ADDRESS": "ADDRESS",
    "EMAIL": "EMAIL",
    "ID": "ID",
    "ORG": "ORG",
    "URL": "URL",
    "SSN": "SSN",
}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().removesuffix("'s"))


@dataclass
class Vault:
    """placeholder -> original surface form (the canonical one for restoring) and every surface form seen."""

    originals: dict[str, str] = field(default_factory=dict)
    surfaces: dict[str, set[str]] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)
    _by_value: dict[str, str] = field(default_factory=dict)
    _counters: Counter[str] = field(default_factory=Counter)

    def placeholder_for(self, span: Span, registered: Registered) -> str:
        value = _norm(span.text)
        if span.type == "DOB":
            return self._remember("[DOB]", span.text, "DOB")
        if span.type == "AGE":
            return self._remember("[AGE_90_PLUS]", span.text, "AGE")
        if span.type == "NAME" and span.who == "patient":
            return self._remember("[PATIENT]", span.text, "NAME")
        if span.type == "NAME":
            linked = self._linked_name(value, registered)
            if linked:
                return self._remember(linked, span.text, "NAME")
        if value in self._by_value:
            return self._remember(self._by_value[value], span.text, span.type)
        prefix = _TYPE_PREFIX.get(span.type, "ID")
        self._counters[prefix] += 1
        placeholder = f"[{prefix}_{self._counters[prefix]}]"
        return self._remember(placeholder, span.text, span.type)

    def _linked_name(self, value: str, registered: Registered) -> str | None:
        tokens = {t for t in re.findall(r"[a-z][a-z'-]+", value) if t not in {"dr", "mr", "mrs", "ms", "miss"}}
        if not tokens:
            return None
        patient = registered.patient_tokens()
        if patient and (tokens <= patient or (tokens & patient and len(tokens) == 1)):
            return "[PATIENT]"
        for placeholder, forms in self.surfaces.items():
            if not placeholder.startswith("[NAME_"):
                continue
            for form in forms:
                known = set(re.findall(r"[a-z][a-z'-]+", form.lower()))
                if tokens <= known or known <= tokens:
                    return placeholder
        return None

    def add(self, surface: str, kind: str) -> str:
        """A placeholder for a value found outside the detector (the egress check's residue)."""
        existing = self._by_value.get(_norm(surface))
        if existing:
            return existing
        prefix = _TYPE_PREFIX.get(kind, "NAME" if kind.startswith("NAME") else "ID")
        self._counters[prefix] += 1
        return self._remember(f"[{prefix}_{self._counters[prefix]}]", surface, kind)

    def _remember(self, placeholder: str, surface: str, kind: str) -> str:
        forms = self.surfaces.setdefault(placeholder, set())
        forms.add(surface)
        current = self.originals.get(placeholder)
        # The canonical form is the longest one seen ("Marisol Ferreira" rather than "Marisol").
        if current is None or len(surface) > len(current):
            self.originals[placeholder] = surface
        self.types[placeholder] = kind
        self._by_value[_norm(surface)] = placeholder
        return placeholder

    def restore(self, text: str) -> str:
        return PLACEHOLDER_RE.sub(lambda m: self.originals.get(m.group(0), m.group(0)), text)

    def unknown_placeholders(self, text: str) -> list[str]:
        return [m.group(0) for m in PLACEHOLDER_RE.finditer(text) if m.group(0) not in self.originals]

    def counts(self) -> dict[str, int]:
        return dict(Counter(self.types.values()))

    def sensitive_values(self) -> list[str]:
        values: set[str] = set()
        for forms in self.surfaces.values():
            values.update(f for f in forms if len(f.strip()) >= 3)
        return sorted(values, key=len, reverse=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "originals": self.originals,
            "surfaces": {k: sorted(v) for k, v in self.surfaces.items()},
            "types": self.types,
            "counters": dict(self._counters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Vault:
        vault = cls()
        vault.originals = dict(data.get("originals", {}))  # type: ignore[call-overload]
        vault.surfaces = {k: set(v) for k, v in dict(data.get("surfaces", {})).items()}  # type: ignore[call-overload]
        vault.types = dict(data.get("types", {}))  # type: ignore[call-overload]
        vault._counters = Counter(dict(data.get("counters", {})))  # type: ignore[call-overload]
        for placeholder, forms in vault.surfaces.items():
            for form in forms:
                vault._by_value[_norm(form)] = placeholder
        return vault


def registered_from(patient: Patient, clinician_name: str = "") -> Registered:
    dob: date | None = None
    if patient.dob:
        dob = parse_date(patient.dob)
    names = [_strip_title(clinician_name)] if clinician_name else []
    return Registered(
        names=[n for n in names if n],
        patient_name=patient.name,
        dob=dob,
        phones=[patient.phone] if patient.phone else [],
        ids=[patient.mrn] if patient.mrn else [],
        emails=[patient.email] if patient.email else [],
        addresses=[patient.address] if patient.address else [],
    )


def _strip_title(name: str) -> str:
    return re.sub(r"^(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Mx|Nurse|Prof)\.?\s+", "", name.strip())


@dataclass
class MaskedTranscript:
    turns: list[Turn]
    vault: Vault
    spans: list[list[Span]]

    def text(self) -> str:
        return "\n".join(f"[{t.idx}] {t.speaker}: {t.text}" for t in self.turns)


class Masker:
    def __init__(self, detector: PHIDetector | None = None) -> None:
        self.detector = detector or PHIDetector()

    def mask_turns(self, turns: Sequence[Turn], registered: Registered, vault: Vault | None = None) -> MaskedTranscript:
        vault = vault or Vault()
        all_spans = self.detector.detect_many([t.text for t in turns], registered)
        texts = [self._apply(t.text, spans, registered, vault) for t, spans in zip(turns, all_spans, strict=True)]
        texts = self._propagate(texts, vault)
        masked = [turn.model_copy(update={"text": text}) for turn, text in zip(turns, texts, strict=True)]
        return MaskedTranscript(masked, vault, all_spans)

    def _propagate(self, texts: list[str], vault: Vault) -> list[str]:
        """A value masked once is masked everywhere ("November" at the start of a sentence too). Values that are
        ordinary words are matched only with the same capitalization ("May" the month, not "may" the verb)."""
        common = self.detector.english | self.detector.ambiguous
        values = sorted(
            ((v, p) for p, forms in vault.surfaces.items() for v in forms if len(v.strip()) >= 3),
            key=lambda x: -len(x[0]),
        )
        out: list[str] = []
        for text in texts:
            for value, placeholder in values:
                flags = 0 if all(w in common for w in value.lower().split()) else re.IGNORECASE
                text = re.sub(rf"(?<![\w\[]){re.escape(value)}(?![\w\]])", placeholder, text, flags=flags)
            out.append(text)
        return out

    def mask_text(self, text: str, registered: Registered, vault: Vault) -> str:
        known = [f for p, forms in vault.surfaces.items() if p.startswith("[NAME") or p == "[PATIENT]" for f in forms]
        spans = self.detector.detect(text, registered, known_names=[w for f in known for w in f.split()])
        return self._apply(text, spans, registered, vault)

    @staticmethod
    def _apply(text: str, spans: Sequence[Span], registered: Registered, vault: Vault) -> str:
        out: list[str] = []
        cursor = 0
        for span in sorted(spans, key=lambda s: s.start):
            if span.start < cursor:
                continue
            out.append(text[cursor : span.start])
            out.append(vault.placeholder_for(span, registered))
            cursor = span.end
        out.append(text[cursor:])
        return "".join(out)
