"""Tolerant parsing of the model's JSON answers into note objects (models wrap JSON in fences, add trailing commas,
return numbers as strings, or name fields slightly differently)."""

from __future__ import annotations

import json
import re
from typing import Any

from notewell.models import SECTIONS, Allergy, Evidence, Medication, Section, Statement


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ParseError("no JSON object in the answer")
    candidate = cleaned[start : end + 1]
    no_trailing = re.sub(r",\s*([}\]])", r"\1", candidate)
    for attempt in (candidate, no_trailing, balance_brackets(no_trailing), balance_brackets(cleaned[start:])):
        try:
            data = json.loads(attempt)
        except json.JSONDecodeError as exc:
            error = exc
            continue
        if not isinstance(data, dict):
            raise ParseError("the answer is not a JSON object")
        return data
    raise ParseError(f"invalid JSON: {error.msg} at line {error.lineno}")


def balance_brackets(text: str) -> str:
    """Repair the most common structural slip of small models: a missing closing brace or bracket. Closers are
    inserted where the nesting requires them (outside strings), missing ones are appended, extra ones dropped."""
    out: list[str] = []
    stack: list[str] = []
    in_string = escaped = False
    pairs = {"{": "}", "[": "]"}
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if ch not in stack:
                continue  # an extra closer
            while stack and stack[-1] != ch:
                out.append(stack.pop())
            stack.pop()
        out.append(ch)
        if not stack and ch in "}]" and text:
            break
    out.extend(reversed(stack))
    return "".join(out)


def _int(value: Any) -> int | None:
    try:
        return int(str(value).strip().lstrip("#").split()[0])
    except (ValueError, IndexError):
        return None


def parse_evidence(raw: Any) -> list[Evidence]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[Evidence] = []
    for item in items:
        if isinstance(item, dict):
            turn = _int(item.get("turn", item.get("turn_id", item.get("idx"))))
            quote = str(item.get("quote", item.get("text", "")) or "").strip()
            if turn is not None or quote:
                out.append(Evidence(turn=turn if turn is not None else -1, quote=quote))
        elif isinstance(item, int | str):
            turn = _int(item)
            if turn is not None:
                out.append(Evidence(turn=turn, quote=""))
    return out


def _statement_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [i if isinstance(i, dict) else {"text": str(i)} for i in raw]
    if isinstance(raw, dict):  # {"chief_complaint": [...], "hpi": [...]} -> flatten
        items: list[dict[str, Any]] = []
        for value in raw.values():
            items.extend(_statement_items(value))
        return items
    if isinstance(raw, str) and raw.strip():
        return [{"text": s} for s in re.split(r"(?<=[.!?])\s+", raw.strip()) if s]
    return []


def parse_note(data: dict[str, Any]) -> tuple[list[Statement], list[Medication], list[Allergy], str]:
    statements: list[Statement] = []
    lowered = {str(k).lower(): v for k, v in data.items()}
    for section in SECTIONS:
        raw = lowered.get(section, lowered.get(section[0], []))
        for i, item in enumerate(_statement_items(raw)):
            text = str(item.get("text", item.get("statement", "")) or "").strip()
            if not text:
                continue
            sec: Section = section
            statements.append(
                Statement(
                    id=f"{section[0]}{i + 1}",
                    section=sec,
                    text=text,
                    evidence=parse_evidence(item.get("evidence", item.get("turns"))),
                )
            )
    meds: list[Medication] = []
    for item in lowered.get("medications", []) or []:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        meds.append(
            Medication(
                name=str(item.get("name", "")).strip(),
                dose=str(item.get("dose", "") or "").strip(),
                frequency=str(item.get("frequency", "") or "").strip(),
                route=str(item.get("route", "") or "").strip(),
                status=str(item.get("status", "active") or "active").strip().lower(),
                evidence=parse_evidence(item.get("evidence", {"turn": item.get("turn"), "quote": item.get("quote")})),
            )
        )
    allergies: list[Allergy] = []
    for item in lowered.get("allergies", []) or []:
        if isinstance(item, dict) and str(item.get("substance", "")).strip():
            substance = str(item["substance"]).strip()
            lowered_substance = substance.lower().strip(" .")
            if lowered_substance in {"none", "nkda", "nka", "n/a", "unknown"} or lowered_substance.startswith(("no ", "none", "denies")):
                continue
            allergies.append(
                Allergy(
                    substance=substance,
                    reaction=str(item.get("reaction", "") or "").strip(),
                    evidence=parse_evidence(item.get("evidence")),
                )
            )
    avs = str(lowered.get("after_visit_summary", lowered.get("avs", "")) or "").strip()
    return statements, meds, allergies, avs


def parse_codes(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    lowered = {str(k).lower(): v for k, v in data.items()}
    icd: list[dict[str, Any]] = []
    for item in lowered.get("icd10", lowered.get("icd10cm", lowered.get("diagnoses", []))) or []:
        if isinstance(item, dict) and item.get("code"):
            icd.append(item)
        elif isinstance(item, str):
            icd.append({"code": item})
    em = lowered.get("em") or lowered.get("e/m") or lowered.get("em_code")
    if isinstance(em, str):
        em = {"code": em}
    procedures = [p for p in lowered.get("procedures", []) or [] if isinstance(p, dict) and p.get("code")]
    return icd, em if isinstance(em, dict) else None, procedures
