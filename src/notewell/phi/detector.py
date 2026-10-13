"""Local PHI detection for visit transcripts: the HIPAA Safe Harbor identifiers that occur in clinic conversations.

Runs entirely in-process (no model, no network), in layers:

1. **Registered identifiers**: the patient's demographics from the schedule (name, date of birth, phone, MRN, address,
   email), matched exactly and, for names, fuzzily (speech recognition misspells names: "Ferreira" -> "Ferrera").
2. **Patterns with validation**: phone numbers (also read digit by digit), e-mail (also "at gmail dot com"), SSN,
   record and member numbers next to a keyword (also spelled "H as in hotel 4 4 7"), dates (spoken and numeric, the
   date of birth is recognized as such), ages over 89, street addresses, ZIP codes, URLs.
3. **Names**: titles ("Dr. Hale", "Mrs. Okafor"), cue phrases ("my name is", "my daughter Emma", "this is"),
   gazetteers (US Census first names and surnames, public domain, plus a curated international list), and an
   aggressive layer that treats an unknown capitalized word in mid-sentence (not English, not a drug, not a clinical
   term) as a name.
4. **Organizations tied to the patient**: employers and schools ("I work at ...", "... Elementary").
5. **Propagation**: a name found once is masked everywhere in the visit, in every casing.

Precision is deliberately traded for recall: a masked word costs the model a little context and is restored locally
afterwards; a leaked identifier cannot be taken back.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from importlib import resources
from typing import Literal

PHIType = Literal["NAME", "DATE", "DOB", "PHONE", "ADDRESS", "EMAIL", "ID", "AGE", "ORG", "URL", "SSN"]

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip
_MONTH_NAMES = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_RE = rf"(?:{_MONTH_NAMES}|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?"
_ORD = r"(?:st|nd|rd|th)?"
_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}  # fmt: skip
_ORD_WORD = r"(?:(?:twenty|thirty)[- ])?(?:" + "|".join(ORDINAL_WORDS) + r")"

TITLES = r"(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Mx|Nurse|Professor|Prof|Coach|Pastor|Father|Sister|Officer)"
RELATIONS = (
    r"(?:husband|wife|son|daughter|mom|mother|mum|dad|father|brother|sister|partner|boyfriend|girlfriend|fiance|"
    r"fiancee|grandson|granddaughter|grandma|grandmother|grandpa|grandfather|aunt|uncle|cousin|niece|nephew|friend|"
    r"neighbor|neighbour|roommate|coworker|co-worker|boss|manager|caregiver|stepson|stepdaughter|child|kid|baby|"
    r"therapist|counselor|sponsor|teacher|nanny|landlord|ex|ex-wife|ex-husband)"
)
STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Way|Place|Pl|Terrace|Ter|Circle|Cir|"
    r"Parkway|Pkwy|Highway|Hwy|Trail|Trl|Loop|Square|Sq|Pike|Row|Run|Crescent|Alley|Path|Point|Ridge|Hollow)"
)
ORG_SUFFIX = (
    r"(?:Elementary|Elementary School|Middle School|High School|High|Academy|Preschool|Pre-K|Daycare|Day Care|"
    r"Montessori|University|College|Community College|Hospital|Medical Center|Clinic|Church|Industries|Inc|LLC|"
    r"Corporation|Corp|Company|Logistics|Foods|Bank|Insurance|Group|Partners|Labs|Systems|Construction|Motors|"
    r"Manufacturing|Warehouse|Distribution|Services|Solutions|Pharmacy)"
)
_ID_KEYWORDS = (
    r"(?:MRN|medical record(?: number)?|record number|chart number|member (?:ID|number)|member I\.?D\.?|"
    r"policy (?:number|ID)|subscriber (?:ID|number)|insurance (?:ID|number)|group number|account number|"
    r"ID number|I\.?D\.? number|patient ID|Medicare (?:number|ID)|Medicaid (?:number|ID)|license number|"
    r"case number|claim number)"
)
_NATO = (
    r"(?:alpha|alfa|bravo|charlie|delta|echo|foxtrot|golf|hotel|india|juliet|juliett|kilo|lima|mike|november|oscar|"
    r"papa|quebec|romeo|sierra|tango|uniform|victor|whiskey|x-ray|xray|yankee|zulu|[a-z]+)"
)
_DIGIT_WORDS = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_NUMBER_WORDS = {
    "ninety": 90, "ninety-one": 91, "ninety-two": 92, "ninety-three": 93, "ninety-four": 94, "ninety-five": 95,
    "ninety-six": 96, "ninety-seven": 97, "ninety-eight": 98, "ninety-nine": 99, "a hundred": 100,
    "one hundred": 100,
}  # fmt: skip

# Capitalized words that are never identifiers on their own in a clinic conversation.
NEVER_NAMES = frozenset(
    """i i'm i've i'll i'd okay ok yes yeah no nope um uh hmm mm oh hi hello hey thanks thank please sorry well so and
    but or if then now right sure alright all god jesus lord christmas thanksgiving easter halloween monday tuesday
    wednesday thursday friday saturday sunday january february march april may june july august september october
    november december dr doctor mr mrs ms miss nurse covid medicare medicaid tylenol advil motrin aleve benadryl zyrtec
    claritin allegra flonase sudafed mucinex nyquil dayquil tums pepto excedrin aspirin vaseline aquaphor cerave
    eucerin cetaphil neosporin epipen ozempic wegovy mounjaro jardiance farxiga eliquis xarelto coumadin lipitor
    crestor zocor synthroid adderall ritalin concerta vyvanse focalin strattera zoloft prozac lexapro celexa paxil
    wellbutrin effexor cymbalta trazodone ambien xanax ativan klonopin valium lamictal depakote seroquel abilify
    lithium flovent albuterol proair ventolin symbicort advair singulair qvar pulmicort xopenex augmentin keflex
    bactrim cipro macrobid zithromax z-pak amoxil prednisone medrol humira dupixent stelara cosentyx skyrizi otezla
    taltz tremfya enbrel accutane absorica differin retin-a tretinoin epiduo aczone benzoyl doxycycline minocycline
    kenalog elidel protopic eucrisa opzelura lasix toprol lopressor norvasc zestril prinivil cozaar diovan entresto
    plavix brilinta nitrostat nitro metformin glucophage januvia lantus humalog novolog basaglar tresiba victoza
    trulicity rybelsus gabapentin neurontin lyrica flexeril mobic celebrex ultram norco percocet narcan naloxone
    vivitrol suboxone naltrexone antabuse campral chantix nicorette nicoderm pap mri ct ekg ecg echo a1c ldl hdl tsh
    bmi phq gad audit emr ehr er ed icu ob gyn ent pt ot ups fedex amazon google facebook instagram tiktok youtube
    netflix zoom bluetooth wifi walgreens cvs rite aid walmart target costco kroger publix safeway english spanish
    american african asian hispanic latino latina european christian muslim jewish catholic baptist""".split()
)

_WORD_RE = re.compile(r"[^\W\d_][\w'\-]*")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: PHIType
    text: str
    source: str
    who: Literal["patient", "other"] = "other"
    priority: int = 0


@dataclass
class Registered:
    """What the clinic already knows about this visit's patient (and clinician): matched before anything else."""

    names: list[str] = field(default_factory=list)
    patient_name: str = ""
    dob: date | None = None
    phones: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    addresses: list[str] = field(default_factory=list)

    def patient_tokens(self) -> set[str]:
        return {t.lower() for t in _WORD_RE.findall(self.patient_name) if len(t) >= 2}

    def name_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for name in [self.patient_name, *self.names]:
            tokens.update(t.lower() for t in _WORD_RE.findall(name) if len(t) >= 2 and t.lower() not in _TITLE_WORDS)
        return tokens


_TITLE_WORDS = {"dr", "doctor", "mr", "mrs", "ms", "miss", "mx", "nurse", "professor", "prof"}


def _load(name: str) -> frozenset[str]:
    text = resources.files("notewell.phi").joinpath("data", name).read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))


@lru_cache(maxsize=1)
def lexicons() -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    """(first names, surnames, ambiguous name-words, English words, clinical and drug terms)."""
    first = _load("first_names_census1990.txt") | _load("first_names_extra.txt")
    surnames = _load("surnames_census2010.txt")
    ambiguous = _load("name_words_ambiguous.txt")
    english = _load("english_words.txt")
    from notewell.kb import clinical_vocabulary

    return first, surnames, ambiguous, english, clinical_vocabulary()


# ------------------------------------------------------------------------------------------------- date parsing
def parse_date(text: str) -> date | None:
    """Parse a spoken or written date ("March 22nd, 1971", "the 22nd of March 1971", "3/22/71"); None if partial."""
    t = text.strip().lower().replace(",", " ")
    t = re.sub(r"\s+", " ", t)
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", t)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + (1900 if y > 30 else 2000) if y < 100 else y
        return _safe_date(y, mo, d)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    month = next((MONTHS[w.rstrip(".")] for w in t.split() if w.rstrip(".") in MONTHS), None)
    if month is None:
        return None
    year_m = re.search(r"\b(19\d{2}|20\d{2})\b", t)
    day = None
    dm = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b(?!\d)", re.sub(r"\b(19|20)\d{2}\b", " ", t))
    if dm:
        day = int(dm.group(1))
    else:
        for word, value in ORDINAL_WORDS.items():
            if re.search(rf"\b(?:(twenty|thirty)[- ])?{word}\b", t):
                prefix = re.search(rf"\b(twenty|thirty)[- ]{word}\b", t)
                day = value + (20 if prefix and prefix.group(1) == "twenty" else 30 if prefix else 0)
                break
    if day is None or year_m is None:
        return None
    return _safe_date(int(year_m.group(1)), month, day)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def dob_matches(text: str, dob: date | None) -> bool:
    if dob is None:
        return False
    parsed = parse_date(text)
    if parsed == dob:
        return True
    # Month and day without a year, or "1971" alone next to the month: still the date of birth.
    t = text.lower()
    month = next((MONTHS[w.strip(".,")] for w in t.split() if w.strip(".,") in MONTHS), None)
    day = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", t)
    return month == dob.month and day is not None and int(day.group(1)) == dob.day


# ------------------------------------------------------------------------------------------------- patterns
_PATTERNS: list[tuple[PHIType, str, re.Pattern[str], int]] = [
    ("EMAIL", "email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), 9),
    (
        "EMAIL",
        "email_spoken",
        re.compile(
            r"\b[\w.+-]+(?:\s(?:dot|underscore)\s[\w+-]+)*\s(?:at)\s[\w-]+(?:\s?(?:dot|\.)\s?(?:com|org|net|edu|gov|io|"
            r"co|us|me))+\b",
            re.IGNORECASE,
        ),
        8,
    ),
    ("URL", "url", re.compile(r"\b(?:https?://|www\.)[^\s,;]+", re.IGNORECASE), 9),
    ("SSN", "ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 9),
    (
        "PHONE",
        "phone",
        re.compile(r"(?<![\d-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\d-])"),
        8,
    ),
    ("PHONE", "phone7", re.compile(r"(?<![\d-])\d{3}-\d{4}(?![\d-])"), 7),
    ("PHONE", "phone_digits", re.compile(r"(?<![\d])(?:\d[\s,.-]{1,2}){6,10}\d(?![\d])"), 6),
    (
        "ID",
        "id_keyword",
        re.compile(
            rf"\b(?i:{_ID_KEYWORDS})\b(?i:\s+(?:is|was|of|it's|its|number|reads|should be|ends in))*[\s:,#-]*"
            rf"((?:[A-Z](?:\s+(?i:as in)\s+[A-Za-z-]+)?[\s,-]+)?(?:[A-Z]{{1,3}}[\s-]?)?(?:\d|(?i:{_DIGIT_WORDS}))"
            rf"(?:[\s,.-]{{0,2}}(?:\d|[A-Z]\b|(?i:{_DIGIT_WORDS})\b))*)"
        ),
        8,
    ),
    ("ID", "id_alnum", re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z]{1,4}-?\d{5,}[A-Z0-9]*\b"), 7),
    ("ID", "id_long_digits", re.compile(r"(?<![\d.])\d{7,}(?![\d.])"), 6),
    (
        "DATE",
        "date_month_day",
        re.compile(
            rf"\b(?:{_WEEKDAY},?\s+)?(?:the\s+)?(?:{_MONTH_RE}\s+(?:\d{{1,2}}{_ORD}|the\s+{_ORD_WORD}|{_ORD_WORD})(?:,?\s+(?:19|20)\d{{2}})?|"
            rf"(?:\d{{1,2}}{_ORD}|{_ORD_WORD})\s+of\s+{_MONTH_RE}(?:,?\s+(?:19|20)\d{{2}})?)\b"
        ),
        7,
    ),
    ("DATE", "date_day_only", re.compile(r"\b(?:on|by|until|till|before|after|since) (the \d{1,2}(?:st|nd|rd|th))\b"), 6),
    ("DATE", "month_alone", re.compile(rf"(?<=[a-z,] )(?:{_MONTH_NAMES})\b(?! I\b)"), 4),
    ("DATE", "date_month_year", re.compile(rf"\b{_MONTH_RE}\s+(?:of\s+)?(?:19|20)\d{{2}}\b"), 6),
    ("DATE", "date_numeric", re.compile(r"\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\b"), 7),
    ("DATE", "date_iso", re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"), 7),
    (
        "AGE",
        "age_over_89",
        re.compile(
            r"\b(?:(?:9\d|1[0-1]\d)|ninety(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|(?:a|one) hundred)"
            r"(?:\s*(?:years?(?:\s|-)old|years? of age|yo\b)|(?=\s*(?:this (?:year|spring|summer|fall|winter|month)|"
            r"next (?:month|week|year)|on (?:my|her|his) (?:birthday))))",
            re.IGNORECASE,
        ),
        7,
    ),
    (
        "AGE",
        "age_im",
        re.compile(
            r"\b(?:I'm|I am|she's|he's|she is|he is|turned|turning|age|aged)\s"
            r"(9\d|1[0-1]\d|ninety(?:-(?:one|two|three|four|five|six|seven|eight|nine))?)\b",
            re.IGNORECASE,
        ),
        7,
    ),
    (
        "ADDRESS",
        "street",
        re.compile(
            rf"\b\d{{1,6}}(?:\s+(?:[NSEW]\.?|North|South|East|West))?(?:\s+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+){{1,4}}\s+{STREET_SUFFIX}\b\.?"
            rf"(?:,?\s+(?:Apartment|Apt\.?|Unit|Suite|#)\s*[\dA-Z]+)?"
            rf"(?:,?\s+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+)?(?:,?\s+(?:[A-Z]{{2}}|[A-Z][a-z]+))?"
            rf"(?:,?\s+\d{{5}}(?:-\d{{4}})?)?)?"
        ),
        8,
    ),
    (
        "ADDRESS",
        "street_named",
        re.compile(rf"\b(?:on|at|off|near)\s+((?:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+\s+){{1,3}}{STREET_SUFFIX})\b"),
        6,
    ),
    ("ADDRESS", "zip", re.compile(r"(?<=\b[A-Z]{2}\s)\d{5}(?:-\d{4})?\b|\bzip(?: code)?(?: is)?\s+\d{5}\b"), 7),
    (
        "ADDRESS",
        "po_box",
        re.compile(r"\b(?:P\.?\s?O\.?\s?Box|post office box)\s+\d+\b", re.IGNORECASE),
        8,
    ),
]

_CUE_NAME = re.compile(
    r"\b(?i:my name is|my name's|name is|name's|i'm|i am|this is|call me|called|named|goes by|it's|its)\s+"
    r"((?:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+)(?:\s+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+){0,2})",
)
_TITLE_NAME = re.compile(rf"\b{TITLES}\.?\s+((?:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+)(?:[\s-]+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+){{0,2}})")
_RELATION_NAME = re.compile(
    rf"\b(?:my|our|her|his|your|the)\s+(?:\w+\s+)?{RELATIONS}(?:'s)?,?\s+((?:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+)(?:\s+[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+)?)"
)
_NAME_RELATION = re.compile(rf"\b([A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'-]+),?\s+(?:is\s+)?(?:my|our|her|his)\s+{RELATIONS}\b")
_ORG = re.compile(
    rf"\b((?:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'&-]*[ ]){{1,4}}{ORG_SUFFIX})\b|"
    rf"\b(?:work|works|working|worked|employed|job|teacher|nurse|cashier|manager)\s+(?:at|for|with)[ ](?:the[ ])?"
    rf"((?:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'&-]*)(?:[ ](?:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ'&-]*|of|and|&)){{0,4}})"
)


class PHIDetector:
    def __init__(self, *, aggressive: bool = True) -> None:
        self.aggressive = aggressive
        self.first, self.surnames, self.ambiguous, self.english, self.clinical = lexicons()

    # ------------------------------------------------------------------------------------------------- public
    def detect(self, text: str, registered: Registered | None = None, known_names: Iterable[str] = ()) -> list[Span]:
        reg = registered or Registered()
        spans: list[Span] = []
        spans += self._registered(text, reg)
        spans += self._patterns(text, reg)
        spans += self._names(text, reg, known_names)
        spans += self._orgs(text)
        return _join_adjacent_names(text, resolve(spans))

    def detect_many(self, texts: Sequence[str], registered: Registered | None = None) -> list[list[Span]]:
        """Detect over all turns of a visit, then propagate every name found to every turn."""
        first_pass = [self.detect(t, registered) for t in texts]
        names: set[str] = set()
        for spans in first_pass:
            for s in spans:
                if s.type == "NAME":
                    names.update(
                        w for w in _WORD_RE.findall(s.text) if len(w) >= 3 and w.lower() not in _TITLE_WORDS
                    )
        if not names:
            return first_pass
        return [self.detect(t, registered, known_names=names) for t in texts]

    # ------------------------------------------------------------------------------------------------- layers
    def _registered(self, text: str, reg: Registered) -> list[Span]:
        spans: list[Span] = []
        patient_tokens = reg.patient_tokens()
        for name in [reg.patient_name, *reg.names]:
            if not name.strip():
                continue
            who: Literal["patient", "other"] = "patient" if name == reg.patient_name else "other"
            for m in re.finditer(rf"(?<![\w]){re.escape(name)}(?![\w])", text, re.IGNORECASE):
                spans.append(Span(m.start(), m.end(), "NAME", m.group(0), "registered", who, 10))
        for token in reg.name_tokens():
            who = "patient" if token in patient_tokens else "other"
            for m in _WORD_RE.finditer(text):
                word = m.group(0)
                lw = word.lower().removesuffix("'s")
                if token in self.english or token in self.ambiguous:
                    hit = word[0].isupper() and lw == token  # "Will" the patient, not "will" the verb
                else:
                    hit = lw == token or (word[0].isupper() and lw not in self.english and _close(lw, token))
                if hit:
                    end = m.start() + len(lw) if word.lower().endswith("'s") else m.end()
                    spans.append(Span(m.start(), end, "NAME", text[m.start() : end], "registered_token", who, 9))
        for phone in reg.phones:
            digits = re.sub(r"\D", "", phone)
            for tail in {digits, digits[-7:]}:
                if len(tail) < 7:
                    continue
                pattern = r"[\s().,-]{0,3}".join(tail)
                for m in re.finditer(pattern, text):
                    spans.append(Span(m.start(), m.end(), "PHONE", m.group(0), "registered", "patient", 10))
        for ident in reg.ids:
            digits = re.sub(r"\D", "", ident)
            if len(digits) >= 4:
                pattern = r"[\s,.-]{0,3}".join(digits)
                for m in re.finditer(pattern, text):
                    spans.append(Span(m.start(), m.end(), "ID", m.group(0), "registered", "patient", 10))
        for email in reg.emails:
            for m in re.finditer(re.escape(email), text, re.IGNORECASE):
                spans.append(Span(m.start(), m.end(), "EMAIL", m.group(0), "registered", "patient", 10))
        for address in reg.addresses:
            street = address.split(",")[0].strip()
            for part in {address, street}:
                if len(part) >= 6:
                    for m in re.finditer(re.escape(part), text, re.IGNORECASE):
                        spans.append(Span(m.start(), m.end(), "ADDRESS", m.group(0), "registered", "patient", 10))
        return spans

    def _patterns(self, text: str, reg: Registered) -> list[Span]:
        spans: list[Span] = []
        for kind, source, pattern, priority in _PATTERNS:
            for m in pattern.finditer(text):
                group = 1 if pattern.groups and m.group(1) else 0
                start, end = m.start(group), m.end(group)
                value = text[start:end].rstrip(" ,.-")
                end = start + len(value)
                if not value:
                    continue
                if source == "phone_digits" and not self._looks_like_spoken_number(text, start, end):
                    continue
                if source == "id_long_digits" and _is_measurement(text, end):
                    continue
                if source == "id_keyword" and not re.search(rf"\d|{_DIGIT_WORDS}", value[-12:], re.IGNORECASE):
                    continue
                if source == "zip" and value.lower().startswith("zip"):
                    start += value.lower().rfind(" ") + 1
                    value = text[start:end]
                ptype: PHIType = kind
                who: Literal["patient", "other"] = "other"
                if kind == "DATE" and dob_matches(value, reg.dob):
                    ptype, who = "DOB", "patient"
                spans.append(Span(start, end, ptype, value, source, who, priority))
        return spans

    def _looks_like_spoken_number(self, text: str, start: int, end: int) -> bool:
        """Digits read one by one ("5 5 5, 0 1 4 2") form a phone or ID; vitals and doses ("98 6") do not."""
        digits = re.sub(r"\D", "", text[start:end])
        groups = re.split(r"[\s,.-]+", text[start:end].strip())
        singles = sum(1 for g in groups if len(g) == 1)
        before = text[max(0, start - 40) : start].lower()
        cue = re.search(r"(number|phone|cell|call|reach|text|fax|id|mrn|record|member|policy|at)\W*$", before)
        return len(digits) >= 7 and (singles >= 4 or cue is not None)

    def _names(self, text: str, reg: Registered, known_names: Iterable[str]) -> list[Span]:
        spans: list[Span] = []
        for m in _TITLE_NAME.finditer(text):
            name = self._trim_name(m.group(1))
            if name:
                spans.append(Span(m.start(1), m.start(1) + len(name), "NAME", name, "title", "other", 8))
        for regex, source in ((_CUE_NAME, "cue"), (_RELATION_NAME, "relation"), (_NAME_RELATION, "relation")):
            for m in regex.finditer(text):
                name = self._trim_name(m.group(1), strict=source == "cue")
                if name:
                    spans.append(Span(m.start(1), m.start(1) + len(name), "NAME", name, source, "other", 7))
        known = {k.lower() for k in known_names}
        sentence_starts = _sentence_starts(text)
        tokens = list(_WORD_RE.finditer(text))
        for i, m in enumerate(tokens):
            word = m.group(0)
            core = word.removesuffix("'s")
            lw = core.lower()
            end = m.start() + len(core)
            if lw in known and len(lw) >= 3:
                spans.append(Span(m.start(), end, "NAME", core, "propagated", "other", 7))
                continue
            if not word[0].isupper() or lw in NEVER_NAMES or lw in self.clinical or len(lw) < 2:
                continue
            if core.isupper() and len(core) <= 5:  # acronyms: MRN, ECG, PHQ
                continue
            at_start = m.start() in sentence_starts
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            followed_by_surname = (
                nxt is not None
                and nxt.group(0)[0].isupper()
                and text[m.end() : nxt.start()] == " "
                and nxt.group(0).lower().removesuffix("'s") in self.surnames
                and nxt.group(0).lower() not in self.clinical
            )
            vocative = text[end : end + 1] in {",", "."} and (at_start or text[max(0, m.start() - 2) : m.start()] == ", ")
            if lw in self.first and (lw not in self.ambiguous or followed_by_surname or (vocative and not at_start)):
                if at_start and not (followed_by_surname or vocative):
                    continue
                spans.append(Span(m.start(), end, "NAME", core, "gazetteer", "other", 5))
                if followed_by_surname and nxt is not None:
                    ncore = nxt.group(0).removesuffix("'s")
                    spans.append(
                        Span(m.start(), nxt.start() + len(ncore), "NAME", text[m.start() : nxt.start() + len(ncore)],
                             "gazetteer", "other", 6)
                    )  # fmt: skip
                continue
            prev = tokens[i - 1] if i > 0 else None
            if (
                prev is not None
                and lw in self.surnames
                and lw not in self.ambiguous
                and prev.group(0)[0].isupper()
                and prev.group(0).lower() in self.first
                and text[prev.end() : m.start()] == " "
            ):
                spans.append(Span(prev.start(), end, "NAME", text[prev.start() : end], "gazetteer", "other", 6))
                continue
            unknown = lw not in self.english and lw not in self.ambiguous and lw.rstrip("s") not in self.english
            if self.aggressive and at_start and unknown and nxt is not None and text[m.end() : nxt.start()] == " ":
                nlw = nxt.group(0).lower().removesuffix("'s")
                if nxt.group(0)[0].isupper() and nlw not in self.english and nlw not in self.clinical:
                    spans.append(Span(m.start(), end, "NAME", core, "unknown_capitalized", "other", 3))
                    continue
            if self.aggressive and not at_start and unknown:
                if lw.rstrip("s") in self.english or (lw.endswith("'s") and lw[:-2] in self.english):
                    continue
                spans.append(Span(m.start(), end, "NAME", core, "unknown_capitalized", "other", 3))
        return spans

    def _trim_name(self, candidate: str, *, strict: bool = False) -> str:
        """Keep the leading run of name-like words ("Emma and" -> "Emma"); empty if the first word is not a name."""
        words = candidate.split()
        kept: list[str] = []
        for w in words:
            lw = w.lower().removesuffix("'s").strip(".,")
            if not w[0].isupper() or lw in NEVER_NAMES or lw in self.clinical:
                break
            if strict and not kept and lw in self.english and lw not in self.first and lw not in self.surnames:
                break
            kept.append(w.removesuffix("'s").strip(".,"))
        return " ".join(kept)

    def _orgs(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for m in _ORG.finditer(text):
            group = 1 if m.group(1) else 2
            start = m.start(group)
            words = m.group(group).strip().split()
            # "The Lincoln Elementary" -> "Lincoln Elementary"; "I work at Home Depot" is kept, "at night" never matches.
            while words and words[0].lower() in {"the", "a", "an", "and", "at", "my", "our"}:
                start = text.index(words[1], start + 1) if len(words) > 1 else start
                words = words[1:]
            if group == 1:
                while words and (words[0].lower() in NEVER_NAMES or words[0].lower() in self.clinical):
                    start = text.index(words[1], start + 1) if len(words) > 1 else start
                    words = words[1:]
            if not words or (group == 1 and len(words) < 2):
                continue
            value = " ".join(words)
            spans.append(Span(start, start + len(value), "ORG", value, "org", "other", 6))
        return spans


def _close(a: str, b: str) -> bool:
    """Speech-recognition misspellings of a registered name: edit distance 1 (or 2 for long names), same initial."""
    if len(b) < 4 or abs(len(a) - len(b)) > 2 or a[:1] != b[:1]:
        return False
    limit = 1 if len(b) < 7 else 2
    return _levenshtein(a, b, limit) <= limit


def _levenshtein(a: str, b: str, limit: int) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _is_measurement(text: str, end: int) -> bool:
    return re.match(r"\s*(?:mg|milligrams?|mcg|micrograms?|units?|ml|steps|calories|dollars|pounds)\b", text[end:], re.I) is not None


def _sentence_starts(text: str) -> set[int]:
    starts = {0}
    starts.update(m.start(1) for m in re.finditer(r"[.!?\n]\s*[\"“]?\s*([A-Za-z])", text))
    starts.update(m.start(1) for m in re.finditer(r"^\s*[\"“]?\s*([A-Za-z])", text))
    return starts


def _join_adjacent_names(text: str, spans: list[Span]) -> list[Span]:
    """"[NAME_3] [NAME_4]" for "Zebulon Quarrington" becomes one name."""
    out: list[Span] = []
    for span in spans:
        prev = out[-1] if out else None
        if prev and prev.type == span.type == "NAME" and text[prev.end : span.start] == " ":
            who = "patient" if "patient" in (prev.who, span.who) else "other"
            out[-1] = Span(prev.start, span.end, "NAME", text[prev.start : span.end], prev.source, who,
                           max(prev.priority, span.priority))  # fmt: skip
        else:
            out.append(span)
    return out


_CONTAINERS = frozenset({"EMAIL", "URL", "ADDRESS", "ID", "PHONE", "SSN", "DOB", "DATE", "ORG"})


def resolve(spans: list[Span]) -> list[Span]:
    """Merge overlapping spans into a non-overlapping list.

    Higher priority wins, then the longer span. Two exceptions keep whole identifiers whole: a structured span
    (e-mail, address, ID, phone, date) that strictly contains other spans absorbs them ("marisol.f at gmail dot com"
    is one e-mail, not a name plus text), and a span of the same type that extends a chosen one grows it.
    """
    ordered = sorted(spans, key=lambda s: (-s.priority, -(s.end - s.start), s.start))
    chosen: list[Span] = []
    for span in ordered:
        if span.end <= span.start:
            continue
        overlaps = [c for c in chosen if span.start < c.end and c.start < span.end]
        if not overlaps:
            chosen.append(span)
            continue
        contains_all = all(span.start <= c.start and span.end >= c.end for c in overlaps)
        longer = (span.end - span.start) > max(c.end - c.start for c in overlaps)
        if contains_all and longer and (span.type in _CONTAINERS or all(c.type == span.type for c in overlaps)):
            for c in overlaps:
                chosen.remove(c)
            same = next((c for c in overlaps if c.type == span.type), None)
            who = "patient" if any(c.who == "patient" for c in overlaps if c.type == span.type) else span.who
            chosen.append(
                Span(span.start, span.end, span.type, span.text, same.source if same else span.source, who,
                     max([span.priority, *(c.priority for c in overlaps)]))
            )  # fmt: skip
    return sorted(chosen, key=lambda s: s.start)
