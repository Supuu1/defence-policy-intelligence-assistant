"""Deterministic extraction and chronological ordering of explicitly dated events."""

from calendar import month_abbr, month_name
from datetime import datetime
import re


MONTHS = {
    name.lower(): number
    for number in range(1, 13)
    for name in (month_name[number], month_abbr[number])
}
MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
DATE_PATTERNS = (
    ("exact", re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")),
    (
        "exact",
        re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTH_PATTERN})[,]?\s+(?:19|20)\d{{2}}\b", re.I),
    ),
    (
        "exact",
        re.compile(rf"\b(?:{MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+(?:19|20)\d{{2}}\b", re.I),
    ),
    ("fiscal", re.compile(r"\b(?:FY|Fiscal Year)\s*(?:19|20)\d{2}(?:\s*[-–/]\s*\d{2,4})?\b", re.I)),
    ("month", re.compile(rf"\b(?:{MONTH_PATTERN})\s+(?:19|20)\d{{2}}\b", re.I)),
    (
        "relative",
        re.compile(
            r"\b(?:within|no later than|not later than)\s+\d+\s+"
            r"(?:calendar\s+|business\s+|working\s+)?(?:day|days|week|weeks|month|months|year|years)\b",
            re.I,
        ),
    ),
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _event_type(text: str) -> str:
    lowered = text.lower()
    categories = (
        ("Publication", ("publish", "publication", "issued", "issuance")),
        ("Effective date", ("effective", "takes effect", "commences")),
        ("Application deadline", ("application", "apply", "submission", "deadline", "submit")),
        ("Review date", ("review", "assessment", "evaluate")),
        ("Implementation milestone", ("implement", "milestone", "phase", "rollout")),
        ("Amendment", ("amend", "revision", "revised")),
        ("Expiration", ("expire", "expiration", "sunset", "valid until")),
    )
    for label, keywords in categories:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "Other dated event"


def _sentence_containing(text: str, start: int, end: int) -> str:
    """Return the local sentence containing a matched date expression."""
    cursor = 0
    for sentence in SENTENCE_BOUNDARY.split(text):
        sentence_start = text.find(sentence, cursor)
        sentence_end = sentence_start + len(sentence)
        cursor = sentence_end
        if sentence_start <= start and end <= sentence_end:
            return " ".join(sentence.split())[:900]
    return " ".join(text.split())[:900]


def _sort_key(date_text: str, precision: str) -> tuple:
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", date_text, flags=re.I).replace(",", "")
    cleaned = re.sub(r"\bSept\b", "Sep", cleaned, flags=re.I)
    if precision == "exact":
        for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
            try:
                parsed = datetime.strptime(cleaned, date_format)
                return parsed.year, parsed.month, parsed.day, 0, date_text.lower()
            except ValueError:
                continue
    if precision == "month":
        parts = cleaned.split()
        month = MONTHS.get(parts[0].lower(), 0)
        return int(parts[-1]), month, 0, 1, date_text.lower()
    if precision == "fiscal":
        year_match = re.search(r"(?:19|20)\d{2}", date_text)
        return int(year_match.group(0)) if year_match else 9998, 1, 0, 2, date_text.lower()
    return 9999, 12, 31, 3, date_text.lower()


def extract_timeline_events(
    chunks: list[dict], document_id: str | None = None
) -> list[dict]:
    """Extract explicitly dated chunk passages and return them chronologically."""
    events = []
    seen = set()
    for chunk in chunks:
        if document_id and chunk.get("document_id") != document_id:
            continue
        text = chunk.get("text", "")
        occupied_spans = []
        for precision, pattern in DATE_PATTERNS:
            for match in pattern.finditer(text):
                if any(match.start() < end and match.end() > start for start, end in occupied_spans):
                    continue
                occupied_spans.append(match.span())
                event_text = _sentence_containing(text, match.start(), match.end())
                date_text = match.group(0)
                identity = (
                    chunk.get("document_id"),
                    chunk.get("page_number"),
                    date_text.lower(),
                    event_text.lower(),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                events.append(
                    {
                        "date": date_text,
                        "date_precision": precision,
                        "event_type": _event_type(event_text),
                        "event": event_text,
                        "source_filename": chunk.get("source_filename", "Unknown document"),
                        "document_id": chunk.get("document_id", ""),
                        "page_number": chunk.get("page_number", 0),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "extraction_method": chunk.get("extraction_method", "Unknown"),
                        "text": event_text,
                        "sort_key": _sort_key(date_text, precision),
                    }
                )

    events.sort(
        key=lambda item: (
            item["sort_key"],
            item["source_filename"],
            item["page_number"],
            item["chunk_id"],
        )
    )
    for position, event in enumerate(events, start=1):
        event["evidence_id"] = f"T{position}"
        event.pop("sort_key", None)
    return events
