"""Evidence-grounded entity and factual-value extraction from existing chunks."""

from collections import Counter
import re


ENTITY_CATEGORIES = (
    "Ministries",
    "Government departments",
    "Defence organizations",
    "Agencies",
    "Committees",
    "Policies",
    "Acts",
    "Rules/regulations",
    "Programs/schemes",
    "Locations",
    "Dates",
    "Monetary amounts",
    "Percentages",
    "Organizations",
    "Equipment/platform names",
    "People/designations",
)
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
CAPITALIZED_WORD = r"(?:[A-Z][A-Za-z0-9&'’.-]*|of|the|and|for|to|in)"
CAPITALIZED_PHRASE = rf"[A-Z][A-Za-z0-9&'’.-]*(?:\s+{CAPITALIZED_WORD}){{0,8}}"


PATTERNS = (
    (
        "Dates",
        re.compile(
            rf"\b(?:"
            rf"(?:19|20)\d{{2}}[-/]\d{{1,2}}[-/]\d{{1,2}}|"
            rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})[,]?\s+(?:19|20)\d{{2}}|"
            rf"(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+(?:19|20)\d{{2}}|"
            rf"(?:{MONTHS})\s+(?:19|20)\d{{2}}|"
            rf"(?:FY|Fiscal Year)\s*(?:19|20)\d{{2}}(?:\s*[-–/]\s*\d{{2,4}})?|"
            rf"(?:within|no later than|not later than)\s+\d+\s+"
            rf"(?:calendar\s+|business\s+|working\s+)?(?:day|days|week|weeks|month|months|year|years)"
            rf")\b",
            re.I,
        ),
    ),
    (
        "Monetary amounts",
        re.compile(
            r"(?<!\w)(?:₹|Rs\.?|INR|US\$|USD|\$|£|€)\s*\d[\d,]*(?:\.\d+)?"
            r"(?:\s*(?:thousand|million|billion|crore|lakh))?\b|"
            r"\b\d[\d,]*(?:\.\d+)?\s*(?:thousand|million|billion|crore|lakh)\s*"
            r"(?:rupees?|dollars?|USD|INR|pounds?|euros?)\b",
            re.I,
        ),
    ),
    ("Percentages", re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)\b", re.I)),
    ("Ministries", re.compile(rf"\bMinistry\s+(?:of\s+)?{CAPITALIZED_PHRASE}\b")),
    ("Government departments", re.compile(rf"\bDepartment\s+(?:of\s+)?{CAPITALIZED_PHRASE}\b")),
    ("Agencies", re.compile(rf"\b(?:National\s+)?Agency\s+(?:for\s+|of\s+)?{CAPITALIZED_PHRASE}\b")),
    ("Agencies", re.compile(rf"\b{CAPITALIZED_PHRASE}\s+Agency\b")),
    ("Committees", re.compile(rf"\b{CAPITALIZED_PHRASE}\s+Committee\b")),
    ("Acts", re.compile(rf"\b{CAPITALIZED_PHRASE}\s+Act(?:,?\s+(?:19|20)\d{{2}})?\b")),
    ("Rules/regulations", re.compile(rf"\b{CAPITALIZED_PHRASE}\s+(?:Rules?|Regulations?)(?:,?\s+(?:19|20)\d{{2}})?\b")),
    ("Policies", re.compile(rf"\b{CAPITALIZED_PHRASE}\s+(?:Policy|Directive|Doctrine|Strategy)\b")),
    ("Programs/schemes", re.compile(rf"\b{CAPITALIZED_PHRASE}\s+(?:Programme|Program|Scheme|Initiative|Mission)\b")),
    (
        "Defence organizations",
        re.compile(
            rf"\b{CAPITALIZED_PHRASE}\s+(?:Command|Army|Navy|Air Force|Armed Forces)\b"
        ),
    ),
    (
        "Equipment/platform names",
        re.compile(
            rf"\b{CAPITALIZED_PHRASE}\s+(?:Aircraft|Platform|System|Tank|Missile|Radar|"
            rf"Drone|UAV|Vessel|Ship|Helicopter|Rifle|Satellite)\b"
        ),
    ),
    (
        "People/designations",
        re.compile(
            rf"\b(?:Minister|Secretary|Chief|Director|General|Admiral|Marshal|Commander|"
            rf"Chairperson|Chairman|Chairwoman)\s+(?:of\s+)?{CAPITALIZED_PHRASE}\b"
        ),
    ),
    (
        "Locations",
        re.compile(
            rf"\b(?:located|based|headquartered|deployed)\s+(?:in|at)\s+"
            rf"([A-Z][A-Za-z'’-]*(?:\s+[A-Z][A-Za-z'’-]*){{0,4}})\b"
        ),
    ),
    (
        "Organizations",
        re.compile(
            rf"\b{CAPITALIZED_PHRASE}\s+(?:Authority|Commission|Council|Board|"
            rf"Organisation|Organization|Directorate|Secretariat)\b"
        ),
    ),
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _context_for_match(text: str, start: int, end: int) -> str:
    cursor = 0
    for sentence in SENTENCE_BOUNDARY.split(text):
        sentence_start = text.find(sentence, cursor)
        sentence_end = sentence_start + len(sentence)
        cursor = sentence_end
        if sentence_start <= start and end <= sentence_end:
            return " ".join(sentence.split())[:700]
    return " ".join(text.split())[:700]


def _clean_entity(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" \t\n,.;:()[]")
    return re.sub(r"^The\s+", "", cleaned)


def _canonical_entity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _refine_category(category: str, entity: str) -> str:
    lowered = entity.lower()
    if category == "Organizations" and any(
        keyword in lowered
        for keyword in ("defence", "defense", "armed", "army", "navy", "air force", "command")
    ):
        return "Defence organizations"
    return category


def extract_entities(
    chunks: list[dict], document_id: str | None = None
) -> list[dict]:
    """Extract explicit entities, merge duplicates, and retain every source reference."""
    merged: dict[tuple[str, str], dict] = {}
    seen_references = set()
    for chunk in chunks:
        if document_id and chunk.get("document_id") != document_id:
            continue
        text = chunk.get("text", "")
        occupied = set()
        for category, pattern in PATTERNS:
            for match in pattern.finditer(text):
                entity = _clean_entity(match.group(1) if category == "Locations" else match.group(0))
                if category == "Organizations" and re.search(r"\s+and\s+(?:the\s+)?", entity):
                    entity = re.split(r"\s+and\s+(?:the\s+)?", entity)[-1]
                if not entity or len(entity) > 180:
                    continue
                refined_category = _refine_category(category, entity)
                canonical = _canonical_entity(entity)
                if not canonical:
                    continue
                occurrence = (match.start(), match.end(), canonical)
                if occurrence in occupied:
                    continue
                occupied.add(occurrence)
                context = _context_for_match(text, match.start(), match.end())
                reference_key = (
                    refined_category,
                    canonical,
                    chunk.get("document_id"),
                    chunk.get("page_number"),
                    chunk.get("chunk_id"),
                    context,
                )
                if reference_key in seen_references:
                    continue
                seen_references.add(reference_key)
                key = refined_category, canonical
                record = merged.setdefault(
                    key,
                    {
                        "entity": entity,
                        "category": refined_category,
                        "references": [],
                    },
                )
                record["references"].append(
                    {
                        "context": context,
                        "source_filename": chunk.get("source_filename", "Unknown document"),
                        "document_id": chunk.get("document_id", ""),
                        "page_number": chunk.get("page_number", 0),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "extraction_method": chunk.get("extraction_method", "Unknown"),
                        "text": context,
                    }
                )

    records = sorted(
        merged.values(), key=lambda item: (ENTITY_CATEGORIES.index(item["category"]), item["entity"].lower())
    )
    evidence_number = 1
    for record in records:
        record["references"].sort(
            key=lambda item: (
                item["source_filename"], item["page_number"], item["chunk_id"]
            )
        )
        for reference in record["references"]:
            reference["evidence_id"] = f"F{evidence_number}"
            evidence_number += 1
    return records


def entity_counts(records: list[dict]) -> dict[str, int]:
    """Count merged entities, rather than repeated source mentions, by category."""
    counts = Counter(record["category"] for record in records)
    return {category: counts[category] for category in ENTITY_CATEGORIES if counts[category]}
