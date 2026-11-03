"""Citation and numeric-claim checks for generated answers."""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal

from .retriever import Hit

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CITATION_RE = re.compile(r"\[(\d+)\]|【(\d+)】")
_SUBJECT_PATTERNS = {
    "home loan": re.compile(r"\bhome[- ]loans?\b", re.I),
    "personal loan": re.compile(r"\bpersonal[- ]loans?\b", re.I),
    "education loan": re.compile(r"\beducation[- ]loans?\b", re.I),
    "vehicle loan": re.compile(r"\bvehicle[- ]loans?\b", re.I),
    "fixed deposit": re.compile(r"\b(?:fixed[- ]deposits?|fds?)\b", re.I),
    "credit card": re.compile(r"\bcredit[- ]cards?\b", re.I),
    "savings": re.compile(r"\bsavings?\b", re.I),
    "current account": re.compile(r"\bcurrent accounts?\b", re.I),
    "loan": re.compile(r"\bloans?\b", re.I),
    "card": re.compile(r"\bcards?\b", re.I),
    "atm": re.compile(r"\batms?\b", re.I),
    "upi": re.compile(r"\bupi\b", re.I),
    "neft": re.compile(r"\bneft\b", re.I),
    "rtgs": re.compile(r"\brtgs\b", re.I),
    "imps": re.compile(r"\bimps\b", re.I),
    "locker": re.compile(r"\blockers?\b", re.I),
}
_CONDITION_PATTERNS = {
    "senior citizens": re.compile(r"\bsenior citizens?\b", re.I),
    "women": re.compile(r"\bwomen\b", re.I),
    "rural": re.compile(r"\brural\b", re.I),
    "semi-urban": re.compile(r"\bsemi[- ]urban\b", re.I),
    "metro": re.compile(r"\bmetro(?:politan)?\b", re.I),
    "urban": re.compile(r"\burban\b", re.I),
    "premature": re.compile(r"\bpremature\b", re.I),
    "annual": re.compile(r"\bannual(?:ly)?\b", re.I),
    "monthly": re.compile(r"\bmonthly\b", re.I),
    "maximum": re.compile(
        r"\b(?:maximum|max(?:imum)?|up to|not more than|no older than|or less|or below)\b",
        re.I,
    ),
    "minimum": re.compile(
        r"\b(?:minimum|min(?:imum)?|at least|or more|or above)\b", re.I
    ),
}
_AMOUNT_DUE_RE = re.compile(r"\bamount\s+due\b", re.I)
_MAGNITUDES = {
    "crore": Decimal("10000000"),
    "crores": Decimal("10000000"),
    "cr": Decimal("10000000"),
    "lakh": Decimal("100000"),
    "lakhs": Decimal("100000"),
    "lac": Decimal("100000"),
    "lacs": Decimal("100000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "thousand": Decimal("1000"),
}


def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        "-" if unicodedata.category(character) == "Pd" else character
        for character in text
    ).translate(str.maketrans({"−": "-"}))
    text = "".join(
        " " if character.isspace() and character not in "\r\n" else character
        for character in text
    )
    return re.sub(r"[*_`~]", "", text)


def _norm_num(token: str) -> str:
    return token.replace(",", "").rstrip(".")


def _numeric_value(text: str, match: re.Match[str]) -> Decimal:
    value = Decimal(_norm_num(match.group()))
    magnitude = re.match(r"\s*(crores?|cr|lakhs?|lacs?|millions?|billion|thousand)\b",
                         text[match.end():], re.I)
    if magnitude:
        value *= _MAGNITUDES[magnitude.group(1).casefold()]
    return value


def _citation_ids(text: str) -> list[int]:
    return [int(square or wide) for square, wide in _CITATION_RE.findall(text)]


def _claims(answer: str) -> list[str]:
    split = re.compile(
        r"\n+|(?<!p\.a\.)(?<!p\.m\.)(?<=[.!?])\s+(?!\[\d+\]|【\d+】)"
    )
    return [part.strip() for part in split.split(answer) if part.strip()]


def _number_unit(text: str, start: int, end: int) -> str | None:
    prefix = text[max(0, start - 12):start].lower()
    suffix = text[end:end + 48].lower()
    base = None
    unit_end = 0
    if "₹" in prefix or re.search(r"\brs\.?\s*$", prefix):
        base = "currency"
    elif re.match(r"\s*(?:%|percent\b|per cent\b)", suffix):
        unit_match = re.match(r"\s*(?:%|percent\b|per cent\b)", suffix)
        base = "percent"
        unit_end = unit_match.end() if unit_match else 0
    else:
        quantity = re.match(
            r"\s*(working\s+days?|years?|yrs?|months?|days?|weeks?)\b", suffix
        )
        if quantity:
            unit = " ".join(quantity.group(1).split())
            base = "working_day" if unit.startswith("working") else unit.rstrip("s")
            unit_end = quantity.end()
        elif re.match(r"\s*(?:crores?|cr|laks?|lacs?|millions?|billion)\b", suffix):
            base = "currency"

    period = re.match(
        r"\s*(?:(?:per|/)\s*(annum|years?|months?|days?|weeks?)\b|p\.(a|m)\.)",
        suffix[unit_end:],
    )
    if period:
        period_unit = period.group(1)
        if period.group(2):
            period_unit = "year" if period.group(2) == "a" else "month"
        else:
            period_unit = "year" if period_unit == "annum" else period_unit.rstrip("s")
        return f"{base}:{period_unit}" if base else f"per_{period_unit}"
    return base


def _table_number_segment(text: str, start: int) -> str | None:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    if not text[line_start:line_end].lstrip().startswith("|"):
        return None

    lines = text.splitlines()
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1
    row_index = next(
        (index for index, row_offset in enumerate(line_offsets)
         if row_offset <= start <= row_offset + len(lines[index])),
        None,
    )
    if row_index is None:
        return None
    first = row_index
    last = row_index
    while first > 0 and lines[first - 1].lstrip().startswith("|"):
        first -= 1
    while last + 1 < len(lines) and lines[last + 1].lstrip().startswith("|"):
        last += 1

    block = lines[first:last + 1]
    separator = next((index for index, line in enumerate(block)
                      if re.search(r"-{3,}", line)), None)
    header_index = first + separator - 1 if separator is not None else first
    headers = [cell.strip() for cell in lines[header_index].strip().strip("|").split("|")]
    cells = [cell.strip() for cell in lines[row_index].strip().strip("|").split("|")]

    target_offset = start - line_offsets[row_index]
    raw = lines[row_index]
    cursor = raw.find("|") + 1
    column = 0
    for column, cell in enumerate(raw.strip().strip("|").split("|")):
        cell_start = raw.find(cell, cursor)
        if cell_start <= target_offset <= cell_start + len(cell):
            break
        cursor = cell_start + len(cell) + 1
    else:
        column = min(column, len(cells) - 1)

    previous = lines[first - 1].strip() if first > 0 else ""
    row_label = cells[0] if cells else ""
    if column == 0:
        column_label = " ".join(headers)
        value = " ".join(cells[1:])
    else:
        column_label = headers[column] if column < len(headers) else ""
        value = cells[column] if column < len(cells) else ""
    return " ".join(part for part in (previous, row_label, column_label, value) if part)


def _number_segment(text: str, start: int, end: int) -> str:
    table_context = _table_number_segment(text, start)
    if table_context is not None:
        return table_context
    delimiters = []
    for index, character in enumerate(text):
        if character in "\n.!?;":
            if character == "." and index > 0 and index + 1 < len(text) \
                    and text[index - 1].isdigit() and text[index + 1].isdigit():
                continue
            delimiters.append(index)
    left = max((index for index in delimiters if index < start), default=-1) + 1
    right = min((index for index in delimiters if index >= end), default=len(text))
    sentence = text[left:right]
    relative = start - left
    comma_before = sentence.rfind(",", 0, relative)
    comma_after = sentence.find(",", end - left)
    clause_start = comma_before + 1
    clause_end = comma_after if comma_after >= 0 else len(sentence)
    clause = sentence[clause_start:clause_end]
    clause_relative = relative - clause_start
    parts = list(re.finditer(r"\b(?:and|but|whereas)\b", clause, re.I))
    boundaries = [0] + [part.end() for part in parts] + [len(clause)]
    starts = [0] + [part.start() for part in parts]
    for index, boundary in enumerate(boundaries[:-1]):
        next_boundary = boundaries[index + 1]
        if boundary <= clause_relative <= next_boundary:
            local = clause[starts[index]:next_boundary].strip()
            if (
                any(pattern.search(local) for pattern in _SUBJECT_PATTERNS.values())
                or any(pattern.search(local) for pattern in _CONDITION_PATTERNS.values())
                or re.search(r"\b(?:age|tenure|fee|amount|balance|limit|rate|term)\b", local, re.I)
            ):
                return local
            return sentence.strip()
    return sentence.strip()


def _supports_number(claim: str, token: str, source: str, source_metadata: str = "") -> bool:
    claim_without_citations = _CITATION_RE.sub(" ", claim)
    claim_match = next(
        (match for match in _NUM_RE.finditer(claim_without_citations)
         if _norm_num(match.group()) == _norm_num(token)),
        None,
    )
    if claim_match is None:
        return False
    claim_unit = _number_unit(claim_without_citations, claim_match.start(), claim_match.end())
    claim_subjects = [name for name, pattern in _SUBJECT_PATTERNS.items()
                      if pattern.search(claim_without_citations)]
    claim_segment = _number_segment(
        claim_without_citations, claim_match.start(), claim_match.end()
    )
    claim_conditions = [name for name, pattern in _CONDITION_PATTERNS.items()
                        if pattern.search(claim_segment)]
    if "maximum" in claim_conditions:
        segment_offset = claim_without_citations.find(claim_segment)
        relative_number = claim_match.start() - max(segment_offset, 0)
        maximum_match = _CONDITION_PATTERNS["maximum"].search(claim_segment)
        if maximum_match and maximum_match.start() > relative_number \
                and re.search(r"\bup to\b", maximum_match.group(), re.I):
            claim_conditions.remove("maximum")

    for source_match in _NUM_RE.finditer(source):
        if _numeric_value(source, source_match) != _numeric_value(claim_without_citations, claim_match):
            continue
        evidence = _number_segment(source, source_match.start(), source_match.end())
        source_unit = _number_unit(source, source_match.start(), source_match.end())
        if source_unit == "percent" and re.search(
            r"(?:\bp\.a\.|\bper annum\b|\bper year\b|\bannual(?:ly)?\b)",
            evidence, re.I,
        ):
            source_unit = "percent:year"
        elif source_unit == "percent" and re.search(
            r"(?:\bp\.m\.|\bper month\b|\bmonthly\b)", evidence, re.I,
        ):
            source_unit = "percent:month"
        implicit_daily_amount = (
            claim_unit == "currency"
            and source_unit == "currency:day"
            and re.search(r"\b(?:daily|per day|each day)\b", claim_segment, re.I)
        )
        unspecified_percent_period = (
            claim_unit == "percent" and source_unit in {"percent:year", "percent:month"}
        )
        if claim_unit and source_unit != claim_unit and not (
            implicit_daily_amount or unspecified_percent_period
        ):
            continue
        if claim_subjects:
            supported_subject = any(_SUBJECT_PATTERNS[name].search(evidence)
                                    for name in claim_subjects)
            conflicting_subject = any(
                pattern.search(evidence) and name not in claim_subjects
                for name, pattern in _SUBJECT_PATTERNS.items()
            )
            metadata_support = any(
                _SUBJECT_PATTERNS[name].search(source_metadata) for name in claim_subjects
            )
            if not supported_subject and (conflicting_subject or not metadata_support):
                continue
        def condition_is_supported(name: str) -> bool:
            if _CONDITION_PATTERNS[name].search(evidence):
                return True
            if name == "minimum":
                after = source[source_match.end():source_match.end() + 40]
                return bool(re.match(
                    r"\s*(?:(?:working\s+)?(?:days?|years?|months?|weeks?)\s*)?"
                    r"(?:to|[-])\s*\d",
                    after,
                    re.I,
                ))
            if name == "maximum":
                before = source[max(0, source_match.start() - 40):source_match.start()]
                return bool(re.search(
                    r"\d\s*(?:(?:working\s+)?(?:days?|years?|months?|weeks?)\s*)?"
                    r"(?:to|[-])\s*$",
                    before,
                    re.I,
                ))
            return False

        if claim_conditions and not all(condition_is_supported(name) for name in claim_conditions):
            continue
        return True
    return False


def _is_customer_amount_input(
    token: str, claim: str, question: str, claim_match: re.Match[str]
) -> bool:
    if not _AMOUNT_DUE_RE.search(question):
        return False
    claim_without_citations = _CITATION_RE.sub(" ", claim)
    if _number_unit(claim_without_citations, claim_match.start(), claim_match.end()) != "currency":
        return False
    if not _AMOUNT_DUE_RE.search(_number_segment(
        claim_without_citations, claim_match.start(), claim_match.end()
    )):
        return False
    return any(
        _numeric_value(question, question_match) == _numeric_value(claim_without_citations, claim_match)
        for question_match in _NUM_RE.finditer(question)
        if _norm_num(question_match.group()) == _norm_num(token)
    )


def verification_issues(answer: str, hits: list[Hit], question: str = "") -> dict[str, list[str]]:
    """Return citation omissions, invalid citations, and unsupported figures separately."""
    answer = _normalise_text(answer)
    question = _normalise_text(question)
    by_rank = {hit.rank: hit for hit in hits}
    issues = {
        "missing_citations": [],
        "invalid_citations": [],
        "unsupported_numbers": [],
    }

    for claim in _claims(answer):
        citations = _citation_ids(claim)
        for citation in citations:
            if citation not in by_rank:
                warning = f"Citation [{citation}] does not match a retrieved source."
                if warning not in issues["invalid_citations"]:
                    issues["invalid_citations"].append(warning)
        claim_text = _CITATION_RE.sub(" ", claim)
        if not citations:
            warning = "A response claim is missing a source citation."
            if warning not in issues["missing_citations"]:
                issues["missing_citations"].append(warning)

        cited_hits = [by_rank[citation] for citation in citations if citation in by_rank]
        for match in _NUM_RE.finditer(claim_text):
            token = match.group()
            if _is_customer_amount_input(token, claim, question, match):
                continue
            if any(
                _supports_number(
                    claim, token, hit.chunk.text,
                    f"{hit.chunk.doc_title} {hit.chunk.section}",
                )
                for hit in cited_hits
            ):
                continue
            unit = _number_unit(claim_text, match.start(), match.end())
            prefix = claim_text[max(0, match.start() - 12):match.start()]
            base_unit = unit.split(":", 1)[0] if unit else None
            figure = f"₹{token}" if base_unit == "currency" and "₹" in prefix else token
            if base_unit == "percent":
                figure += "%"
            elif base_unit in {"year", "yr", "month", "day", "week", "working_day"}:
                figure += f" {base_unit.replace('_', ' ')}"
            warning = f"{figure} is not verified by the cited policy passage(s)."
            if warning not in issues["unsupported_numbers"]:
                issues["unsupported_numbers"].append(warning)
    return issues


def verification_warnings(answer: str, hits: list[Hit], question: str = "") -> list[str]:
    """Flatten categorized verification issues for existing UI callers."""
    issues = verification_issues(answer, hits, question)
    return [
        warning
        for category in ("missing_citations", "invalid_citations", "unsupported_numbers")
        for warning in issues[category]
    ]


def unverified_numbers(answer: str, hits: list[Hit]) -> list[str]:
    """Return numeric claims unsupported by their own cited passages."""
    answer = _normalise_text(answer)
    by_rank = {hit.rank: hit for hit in hits}
    missing: list[str] = []
    for claim in _claims(answer):
        citations = _citation_ids(claim)
        cited_hits = [by_rank[citation] for citation in citations if citation in by_rank]
        for match in _NUM_RE.finditer(_CITATION_RE.sub(" ", claim)):
            token = match.group()
            if not any(
                _supports_number(claim, token, hit.chunk.text,
                                 f"{hit.chunk.doc_title} {hit.chunk.section}")
                for hit in cited_hits
            ) and token not in missing:
                missing.append(token)
    return missing
