"""The official ICD-10-CM code list: lookup, validation and candidate retrieval for the coding step.

The full FY2027 order file comes from CDC/NCHS (`notewell codes download`, verified by SHA-256). Without it, a subset
shipped with the package is used (some chapters only) and validation says so: a code outside the subset is reported
as "not verifiable offline", never as invalid.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

FULL_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/icd10cm-code-descriptions-2027.zip"
FULL_SHA256 = "93e3ad6004badf470c55bfe679b748ae88fd9b2b421851e409eec382c7713b9a"
FULL_MEMBER = "icd10cm-order-2027.txt"
SUBSET_FILE = "icd10cm-order-2027-subset.txt"
FISCAL_YEAR = "FY2027 (October 1, 2026 - September 30, 2027)"

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "of",
        "the",
        "and",
        "or",
        "in",
        "to",
        "for",
        "by",
        "due",
        "other",
        "specified",
        "not",
        "elsewhere",
        "classified",
        "type",
        "nos",
        "a",
        "an",
        "as",
        "on",
        "at",
        "site",
    ]
)
_SYNONYMS = {
    "high blood pressure": "hypertension",
    "heart attack": "myocardial infarction",
    "afib": "atrial fibrillation",
    "a-fib": "atrial fibrillation",
    "sugar": "diabetes",
    "underactive thyroid": "hypothyroidism",
    "cholesterol": "hyperlipidemia",
    "high cholesterol": "hyperlipidemia",
    "uti": "urinary tract infection cystitis",
    "bladder infection": "cystitis",
    "ear infection": "otitis media",
    "strep throat": "streptococcal pharyngitis",
    "eczema": "atopic dermatitis",
    "warts": "verruca",
    "wart": "verruca",
    "mole": "nevus",
    "depression": "depressive episode",
    "anxiety": "anxiety disorder",
    "adhd": "attention deficit hyperactivity",
    "copd": "chronic obstructive pulmonary",
    "chf": "heart failure",
    "hfref": "systolic heart failure",
    "sprain": "sprain",
    "low back pain": "low back pain",
    "checkup": "general adult medical examination",
    "physical": "general medical examination",
}


@dataclass(frozen=True)
class ICDCode:
    code: str
    """Without the dot, as in the CDC files (E119)."""
    billable: bool
    short: str
    title: str

    @property
    def dotted(self) -> str:
        return dotted(self.code)


def dotted(code: str) -> str:
    c = code.replace(".", "").strip().upper()
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


def undotted(code: str) -> str:
    return code.replace(".", "").strip().upper()


Status = Literal["valid", "not_billable", "invalid", "unverifiable"]


@dataclass(frozen=True)
class Validation:
    status: Status
    code: str
    title: str = ""
    children: tuple[str, ...] = ()
    message: str = ""


def default_full_path(codes_dir: Path) -> Path:
    return codes_dir / "full" / FULL_MEMBER


def download_full(codes_dir: Path, *, url: str = FULL_URL, sha256: str = FULL_SHA256) -> Path:
    """Download the official code descriptions zip from CDC, verify its checksum, extract the order file."""
    target = default_full_path(codes_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"checksum mismatch for {url}: {digest} (expected {sha256})")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(n for n in archive.namelist() if n.endswith(FULL_MEMBER))
        tmp = target.with_suffix(".part")
        tmp.write_bytes(archive.read(member))
        tmp.replace(target)
    return target


class CodeSet:
    def __init__(self, lines: list[str], *, source: str, complete: bool) -> None:
        self.source = source
        self.complete = complete
        self.codes: dict[str, ICDCode] = {}
        for line in lines:
            if not line.strip() or line.startswith("#") or len(line) < 16:
                continue
            code = line[6:13].strip()
            billable = line[14] == "1"
            short = line[16:76].strip()
            title = line[77:].strip() or short
            self.codes[code] = ICDCode(code, billable, short, title)
        self.categories = {c[:3] for c in self.codes}
        self._index: _BM25 | None = None

    @classmethod
    def load(cls, codes_dir: Path | None = None) -> CodeSet:
        if codes_dir is not None:
            full = default_full_path(codes_dir)
            if full.exists() and full.stat().st_size > 1_000_000:
                return cls(full.read_text(encoding="utf-8").splitlines(), source=str(full), complete=True)
        text = resources.files("notewell.codes").joinpath("data", SUBSET_FILE).read_text(encoding="utf-8")
        return cls(text.splitlines(), source=f"package subset ({SUBSET_FILE})", complete=False)

    def get(self, code: str) -> ICDCode | None:
        return self.codes.get(undotted(code))

    def children(self, code: str, *, limit: int = 12) -> tuple[str, ...]:
        prefix = undotted(code)
        kids = [c.dotted for k, c in self.codes.items() if k.startswith(prefix) and k != prefix and c.billable]
        return tuple(kids[:limit])

    def validate(self, code: str) -> Validation:
        raw = undotted(code)
        if not re.fullmatch(r"[A-Z][0-9][0-9A-Z](?:[0-9A-Z]{1,4})?", raw):
            return Validation("invalid", code, message=f"{code!r} is not an ICD-10-CM code format")
        entry = self.codes.get(raw)
        if entry is None:
            if not self.complete and raw[:3] not in self.categories:
                return Validation(
                    "unverifiable",
                    dotted(raw),
                    message="outside the offline subset; run `notewell codes download` to validate against the full list",
                )
            return Validation("invalid", dotted(raw), message=f"{dotted(raw)} is not in the ICD-10-CM {FISCAL_YEAR}")
        if not entry.billable:
            kids = self.children(raw)
            return Validation(
                "not_billable",
                entry.dotted,
                entry.title,
                kids,
                f"{entry.dotted} is a category header, not billable; choose a more specific code",
            )
        return Validation("valid", entry.dotted, entry.title)

    def search(self, query: str, *, limit: int = 20, billable_only: bool = True) -> list[tuple[ICDCode, float]]:
        if self._index is None:
            self._index = _BM25([(k, f"{c.title} {c.short}") for k, c in self.codes.items()])
        expanded = query.lower()
        for phrase, extra in _SYNONYMS.items():
            if phrase in expanded:
                expanded += " " + extra
        hits = self._index.search(expanded, limit=limit * 4)
        out: list[tuple[ICDCode, float]] = []
        for key, score in hits:
            code = self.codes[key]
            if billable_only and not code.billable:
                continue
            out.append((code, score))
            if len(out) >= limit:
                break
        return out


class _BM25:
    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.2, b: float = 0.6) -> None:
        self.k1, self.b = k1, b
        self.keys = [k for k, _ in docs]
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, (_, text) in enumerate(docs):
            tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
            self.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.n = len(docs)
        self.avg = sum(self.lengths) / max(1, self.n)

    def search(self, query: str, *, limit: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in {t for t in _TOKEN.findall(query.lower()) if t not in _STOP}:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc, tf in plist:
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[doc] / self.avg))
                scores[doc] += idf * norm
        # Shorter titles first among equals: the less specific "unspecified" code is the safer suggestion.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.lengths[kv[0]]))
        return [(self.keys[i], s) for i, s in ranked[:limit]]


@lru_cache(maxsize=4)
def load_codeset(codes_dir: str | None) -> CodeSet:
    return CodeSet.load(Path(codes_dir) if codes_dir else None)
