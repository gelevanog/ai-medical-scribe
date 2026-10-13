"""The egress check: the last line of defence before a payload leaves the process.

It does not trust the masker. It searches the final outgoing messages (system prompt included) for every value the
visit's vault holds and every registered identifier of the patient, in several normal forms: case-insensitive text,
digit runs with any separators ("614 555 0187" = "614-555-0187"), and close misspellings of registered names. A hit
raises PolicyViolationError with counts by type, never the values, and nothing is sent.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from notewell.llm.base import Message, PolicyViolationError
from notewell.phi.detector import Registered, _close, lexicons
from notewell.phi.masker import Vault

_WORD = re.compile(r"[a-z][a-z'-]+")
_DIGITS = re.compile(r"\d(?:[\s().,/-]{0,3}\d)+")


@dataclass
class EgressReport:
    hits: Counter[str] = field(default_factory=Counter)
    found: set[str] = field(default_factory=set, repr=False)
    """The matched surface forms, for masking the residue in-process. Never logged or returned to a client."""

    @property
    def clean(self) -> bool:
        return not self.hits

    def summary(self) -> str:
        return ", ".join(f"{kind} x{n}" for kind, n in sorted(self.hits.items()))


class EgressGuard:
    def __init__(self, registered: Registered, vault: Vault | None = None, *, extra_values: Sequence[str] = ()) -> None:
        _, _, ambiguous, english, _ = lexicons()
        self._common = english | ambiguous
        self.texts: list[tuple[str, str]] = []
        self.cased: list[tuple[str, str]] = []
        """Values that are also ordinary words ("Will", "Grace"): only their capitalized form counts."""
        self.digits: list[tuple[str, str]] = []
        self.name_tokens: set[str] = set()
        for value, kind in self._values(registered, vault, extra_values):
            value = value.strip()
            digits = re.sub(r"\D", "", value)
            if len(digits) >= 6 and len(digits) >= len(re.sub(r"\s", "", value)) * 0.6:
                self.digits.append((digits, kind))
            elif len(value) >= 3 and all(w in self._common for w in value.lower().split()):
                if value[:1].isupper():
                    self.cased.append((value, kind))  # "Will Hope" the person, not "will hope" the words
            elif len(value) >= 3:
                self.texts.append((value.lower(), kind))
        for name in [registered.patient_name, *registered.names]:
            self.name_tokens.update(t for t in _WORD.findall(name.lower()) if len(t) >= 6 and t not in self._common)

    @staticmethod
    def _values(registered: Registered, vault: Vault | None, extra: Sequence[str]) -> list[tuple[str, str]]:
        values: list[tuple[str, str]] = []
        if registered.patient_name:
            values.append((registered.patient_name, "NAME"))
            values.extend((t, "NAME") for t in registered.patient_name.split() if len(t) >= 3)
        for name in registered.names:
            values.append((name, "NAME"))
        if registered.dob:
            d = registered.dob
            values.extend(
                (form, "DOB")
                for form in (
                    d.strftime("%B %-d, %Y"),
                    d.strftime("%m/%d/%Y"),
                    f"{d.month}/{d.day}/{d.year}",
                    f"{d.month}/{d.day}/{d.strftime('%y')}",
                    d.isoformat(),
                )
            )
        values.extend((p, "PHONE") for p in registered.phones)
        values.extend((i, "ID") for i in registered.ids)
        values.extend((e, "EMAIL") for e in registered.emails)
        for address in registered.addresses:
            values.append((address.split(",")[0], "ADDRESS"))
        if vault is not None:
            for placeholder, forms in vault.surfaces.items():
                kind = vault.types.get(placeholder, "ID")
                values.extend((f, kind) for f in forms)
        values.extend((v, "EXTRA") for v in extra)
        return values

    def scan(self, messages: Sequence[Message]) -> EgressReport:
        report = EgressReport()
        for message in messages:
            text = message["content"]
            lower = text.lower()
            for value, kind in self.texts:
                if m := re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", lower):
                    report.hits[kind] += 1
                    report.found.add(text[m.start() : m.end()])
            for value, kind in self.cased:
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", text):
                    report.hits[kind] += 1
                    report.found.add(value)
            runs = [(re.sub(r"\D", "", m.group(0)), m.group(0)) for m in _DIGITS.finditer(text)]
            for digits, kind in self.digits:
                for run, surface in runs:
                    if digits in run:
                        report.hits[kind] += 1
                        report.found.add(surface)
            for word in set(_WORD.findall(lower)) - self._common:
                if any(word != token and _close(word, token) for token in self.name_tokens):
                    report.hits["NAME_FUZZY"] += 1
                    report.found.update(m.group(0) for m in re.finditer(rf"\b{re.escape(word)}\b", text, re.I))
        return report

    def check(self, messages: Sequence[Message]) -> None:
        report = self.scan(messages)
        if not report.clean:
            raise PolicyViolationError(
                f"egress check: identifiers found in the outgoing payload ({report.summary()}); nothing was sent"
            )
