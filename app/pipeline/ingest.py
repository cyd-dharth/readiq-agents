from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from app.config import get_settings
from app.llm import LLMClient

logger = logging.getLogger(__name__)

_CHAPTER_DETECT_SYSTEM = "You are a book structure analyst. Return JSON only, no other text."

_ROMAN_NUMERAL_RE = re.compile(
    r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)[.:]?$", re.IGNORECASE
)
_CHAPTER_WORD_RE = re.compile(r"^(chapter|part|book|section)\b", re.IGNORECASE)
_DIGIT_HEADING_RE = re.compile(r"^\d+[.\s]")


class _TocPageEntry(BaseModel):
    """A single table of contents entry with its chapter title and printed page number."""

    chapter_title: str
    toc_page: int


class _TocPageEntries(BaseModel):
    """Wrapper for a list of TOC entries returned by the Tier 1 LLM call."""

    entries: list[_TocPageEntry]


class _TocTitles(BaseModel):
    """Wrapper for a list of chapter titles returned by the Tier 2 LLM call, with no page numbers."""

    titles: list[str]


class _PageBoundary(BaseModel):
    """A single detected chapter start, identified by pypdf page index and title."""

    page: int
    chapter_title: str


class _PageBoundaries(BaseModel):
    """Wrapper for a list of page boundaries returned by the Tier 4 LLM call."""

    boundaries: list[_PageBoundary]


def _toc_page_numbers_prompt(toc_text: str) -> str:
    """Build the Tier 1 prompt asking the LLM to extract chapter titles and page numbers from TOC text."""
    return (
        "Here is the table of contents of a book. Extract chapter titles and "
        "their page numbers. Ignore entries that are not chapters or major "
        "sections (ignore preface, acknowledgements, bibliography, index, "
        "appendix, notes, introduction unless it is a numbered chapter). "
        "Return the chapters found in the 'entries' array. "
        "If no page numbers are present in the TOC, return an empty array.\n\n"
        f"TOC text:\n{toc_text}"
    )


def _toc_titles_prompt(pages_text: str) -> str:
    """Build the Tier 2 prompt asking the LLM to extract chapter titles only, when the TOC has no page numbers."""
    return (
        "Here are the first pages of a book which may contain a table of "
        "contents. Extract chapter and section titles in order. "
        "Ignore page numbers, dots, leader lines, and non-chapter entries "
        "such as bibliography, index, and appendix. "
        "If no table of contents is present, return an empty array.\n\n"
        f"Pages:\n{pages_text}"
    )


def _page_analysis_prompt(page_lines: str) -> str:
    """Build the Tier 4 prompt asking the LLM to pick out chapter-start pages from each page's first two lines."""
    return (
        "Here are the first two lines of each page of a book PDF. "
        "Identify which pages start a new chapter or major section. "
        "Ignore table of contents, copyright, preface, and index pages.\n\n"
        f"{page_lines}"
    )


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace so titles can be compared loosely."""
    lowered = text.lower()
    stripped = re.sub(r"[^a-z0-9\s]", "", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def _non_empty_lines(page_text: str) -> list[str]:
    """Split a page's text into lines, dropping blank ones."""
    return [line.strip() for line in page_text.splitlines() if line.strip()]


async def ingest_pdf(file_path: str, llm: LLMClient, max_tokens: int) -> list[dict]:
    """
    Parse a PDF into chapters via pypdf and a 4-tier chapter detection cascade.
    Falls back to the whole book as a single chapter (chapter_title=None) if the
    PDF looks image-based (too little extracted text) or if detection fails or
    raises for any reason. Never raises.
    """
    settings = get_settings()
    try:
        pages = _extract_pages(file_path)
    except Exception:
        logger.warning("PDF text extraction failed for %s", file_path, exc_info=True)
        return [{"chapter_number": 1, "chapter_title": None, "raw_text": ""}]

    full_text = "".join(pages)
    if len(full_text.strip()) < settings.unreadable_pdf_min_chars:
        logger.warning(
            "PDF appears to be image-based or has no extractable text. "
            "pypdf returned fewer than %d characters. "
            "The file may be a scanned document without an OCR text layer. "
            "Returning whole-book fallback. File: %s",
            settings.unreadable_pdf_min_chars,
            file_path,
        )
        return [{"chapter_number": 1, "chapter_title": None, "raw_text": full_text}]

    try:
        boundaries = await _detect_boundaries(pages, llm, max_tokens, settings)
        return _split_by_page_boundaries(pages, boundaries)
    except Exception as exc:
        logger.warning("Chapter detection failed: %s. Using whole-book fallback.", exc, exc_info=True)
        return [{"chapter_number": 1, "chapter_title": None, "raw_text": full_text}]


def _extract_pages(file_path: str) -> list[str]:
    """Extract raw text for each page of the PDF via pypdf, one string per page."""
    from pypdf import PdfReader  # lazy: only needed for the pdf path

    reader = PdfReader(file_path)
    return [page.extract_text() or "" for page in reader.pages]


async def _detect_boundaries(pages: list[str], llm: LLMClient, max_tokens: int, settings) -> list[dict]:
    """
    Run the 4-tier chapter boundary detection cascade in order, falling through
    to the next tier only when the current one finds fewer than 2 boundaries.
    Raises if even Tier 4 fails to find 2 or more boundaries, so the caller
    can fall back to a whole-book chapter.
    """
    # Tier 1: TOC page number extraction.
    boundaries = await _tier1_toc_page_numbers(pages, llm, max_tokens, settings)
    if boundaries is not None and len(boundaries) >= 2:
        logger.info("Chapter detection: Tier 1 succeeded, %d chapters found", len(boundaries))
        return boundaries
    logger.info("Tier 1 found fewer than 2 boundaries, trying Tier 2")

    # Tier 2: TOC title extraction without page numbers.
    boundaries = await _tier2_toc_titles(pages, llm, max_tokens, settings)
    if len(boundaries) >= 2:
        logger.info("Chapter detection: Tier 2 succeeded, %d chapters found", len(boundaries))
        return boundaries
    logger.info("Tier 2 found fewer than 2 boundaries, trying Tier 3")

    # Tier 3: generic heuristic patterns.
    boundaries = _tier3_generic_heuristics(pages)
    if len(boundaries) >= 2:
        logger.info("Chapter detection: Tier 3 succeeded, %d chapters found", len(boundaries))
        return boundaries
    logger.info("Tier 3 found fewer than 2 boundaries, trying Tier 4")

    # Tier 4: LLM full page analysis.
    boundaries = await _tier4_llm_page_analysis(pages, llm, max_tokens, settings)
    if len(boundaries) < 2:
        raise ValueError("no tier found 2 or more chapter boundaries")
    logger.info("Chapter detection: Tier 4 succeeded, %d chapters found", len(boundaries))
    return boundaries


def _is_toc_page(page_text: str) -> bool:
    """Heuristically detect a table of contents page: starts with 'contents' or has 3+ lines ending in a digit."""
    lines = _non_empty_lines(page_text)
    if not lines:
        return False
    if lines[0].lower().startswith("contents") or "contents" in lines[0].lower():
        return True
    digit_ending_lines = sum(1 for line in lines if line.rstrip()[-1:].isdigit())
    return digit_ending_lines >= 3


def _collect_toc_page_indices(pages: list[str], scan_pages: int) -> list[int]:
    """Scan the first scan_pages pages and return the indices of a contiguous run of TOC pages, if any."""
    toc_indices = []
    started = False
    for i, page_text in enumerate(pages[:scan_pages]):
        if _is_toc_page(page_text):
            started = True
            toc_indices.append(i)
        elif started:
            break
    return toc_indices


def _collect_toc_text(pages: list[str], scan_pages: int) -> str:
    """Concatenate the text of the detected TOC pages into one string for the LLM prompt."""
    toc_indices = _collect_toc_page_indices(pages, scan_pages)
    return "\n".join(pages[i] for i in toc_indices)


async def _tier1_toc_page_numbers(pages: list[str], llm: LLMClient, max_tokens: int, settings) -> list[dict] | None:
    """
    Tier 1: extract {chapter_title, toc_page} pairs from the TOC via LLM, then
    calibrate the offset between printed TOC page numbers and actual pypdf page
    indices and map each entry to the closest matching page. Returns None if no
    TOC text or no entries were found.
    """
    toc_indices = _collect_toc_page_indices(pages, settings.toc_scan_pages)
    toc_text = "\n".join(pages[i] for i in toc_indices)
    if not toc_text.strip():
        return None

    try:
        result = await llm.complete_json(
            _CHAPTER_DETECT_SYSTEM, _toc_page_numbers_prompt(toc_text), max_tokens, _TocPageEntries
        )
        entries = _TocPageEntries.model_validate(result).entries
    except Exception:
        logger.warning("Tier 1 LLM parse failed", exc_info=True)
        return None

    if not entries:
        return None

    toc_index_set = set(toc_indices)
    offset = _calibrate_offset(pages, entries[0], toc_index_set)

    confirmed = []
    for entry in entries:
        mapped = max(0, min(entry.toc_page + offset, len(pages) - 1))
        best_page = _best_matching_page_near(pages, mapped, entry.chapter_title, toc_index_set)
        confirmed.append({"page": best_page, "chapter_title": entry.chapter_title})

    confirmed.sort(key=lambda item: item["page"])
    return confirmed


def _calibrate_offset(pages: list[str], first_entry: _TocPageEntry, toc_index_set: set[int]) -> int:
    """
    Find the pypdf page index where the first TOC entry's title actually
    appears, near its printed page number, and return the offset between the
    two. Front matter shifts printed page numbers relative to pypdf's 0-based
    index, so this offset is needed to map every other TOC entry. Defaults to
    0 if no match is found nearby.
    """
    n = first_entry.toc_page
    norm_title = _normalise(first_entry.chapter_title)
    lo = max(0, n - 5)
    hi = min(len(pages) - 1, n + 10)
    for candidate in range(lo, hi + 1):
        if candidate in toc_index_set:
            continue
        lines = _non_empty_lines(pages[candidate])[:3]
        norm_page_text = _normalise(" ".join(lines))
        if norm_title and norm_title in norm_page_text:
            offset = candidate - n
            logger.info(
                "TOC page offset detected: %d (toc_page %d found at pdf index %d)", offset, n, n + offset
            )
            return offset
    logger.warning("TOC offset not found for toc_page %d, defaulting to 0", n)
    return 0


def _best_matching_page_near(pages: list[str], mapped_index: int, title: str, toc_index_set: set[int]) -> int:
    """Given a calibrated page estimate, check the page itself and its neighbors for the best title match."""
    norm_title = _normalise(title)
    candidates = [
        i for i in (mapped_index - 1, mapped_index, mapped_index + 1)
        if 0 <= i < len(pages) and i not in toc_index_set
    ]
    if not candidates:
        candidates = [mapped_index]

    best_index = mapped_index
    best_score = -1
    for i in candidates:
        lines = _non_empty_lines(pages[i])[:3]
        norm_page_text = _normalise(" ".join(lines))
        score = 1 if norm_title and norm_title in norm_page_text else 0
        if score > best_score:
            best_score = score
            best_index = i
    return best_index


async def _tier2_toc_titles(pages: list[str], llm: LLMClient, max_tokens: int, settings) -> list[dict]:
    """
    Tier 2: when the TOC has no page numbers, ask the LLM to extract just the
    chapter titles, then heuristically match each page's first line against
    that title list to locate chapter starts.
    """
    toc_text = _collect_toc_text(pages, settings.toc_scan_pages)
    if not toc_text.strip():
        toc_text = "\n".join(pages[: settings.toc_titles_scan_pages])

    toc_titles: list[str] = []
    try:
        result = await llm.complete_json(
            _CHAPTER_DETECT_SYSTEM, _toc_titles_prompt(toc_text), max_tokens, _TocTitles
        )
        toc_titles = _TocTitles.model_validate(result).titles
    except Exception:
        logger.warning("Tier 2 LLM parse failed", exc_info=True)
        toc_titles = []

    if not toc_titles:
        return []

    norm_titles = [_normalise(t) for t in toc_titles if _normalise(t)]

    boundaries = []
    for page_index, page_text in enumerate(pages):
        lines = _non_empty_lines(page_text)
        if not lines:
            continue
        candidate = lines[0]
        if len(candidate) >= 80 or len(lines) <= 3:
            continue
        norm_candidate = _normalise(candidate)
        if not norm_candidate:
            continue
        matched = any(
            norm_title in norm_candidate or norm_candidate in norm_title
            for norm_title in norm_titles
        )
        if matched:
            boundaries.append({"page": page_index, "chapter_title": candidate})

    return boundaries


def _tier3_generic_heuristics(pages: list[str]) -> list[dict]:
    """Tier 3: pure regex heuristics, no LLM call. Matches chapter/part keywords, roman numerals, digit headings, or short all-caps lines as the first line of a page."""
    boundaries = []
    for page_index, page_text in enumerate(pages):
        lines = _non_empty_lines(page_text)
        if not lines:
            continue
        candidate = lines[0]
        if len(candidate) >= 60 or len(lines) <= 3:
            continue
        stripped = candidate.strip()
        is_match = (
            bool(_CHAPTER_WORD_RE.match(stripped))
            or bool(_ROMAN_NUMERAL_RE.match(stripped))
            or bool(_DIGIT_HEADING_RE.match(stripped))
            or (stripped.isupper() and len(stripped) < 40)
        )
        if is_match:
            boundaries.append({"page": page_index, "chapter_title": candidate})

    return boundaries


async def _tier4_llm_page_analysis(pages: list[str], llm: LLMClient, max_tokens: int, settings) -> list[dict]:
    """
    Tier 4: send the LLM the first two lines of every page (sampling every
    other page if the combined text exceeds the configured char limit) and
    ask it to identify which pages start a new chapter directly.
    """
    entries = []
    for page_index, page_text in enumerate(pages):
        lines = _non_empty_lines(page_text)[:2]
        entries.append((page_index, " | ".join(lines)))

    page_lines_text = "\n".join(f"Page {i}: {text}" for i, text in entries)
    if len(page_lines_text) > settings.tier4_page_lines_char_limit:
        sampled = entries[::2]
        page_lines_text = "\n".join(f"Page {i}: {text}" for i, text in sampled)

    try:
        result = await llm.complete_json(
            _CHAPTER_DETECT_SYSTEM, _page_analysis_prompt(page_lines_text), max_tokens, _PageBoundaries
        )
        page_boundaries = _PageBoundaries.model_validate(result).boundaries
    except Exception:
        logger.warning("Tier 4 LLM parse failed", exc_info=True)
        return []

    return [{"page": b.page, "chapter_title": b.chapter_title} for b in page_boundaries]


def _split_by_page_boundaries(pages: list[str], boundaries: list[dict]) -> list[dict]:
    """Slice the page list into chapters using each boundary's start page through the next boundary (or end of book)."""
    boundaries = sorted(boundaries, key=lambda item: item["page"])

    chapters = []
    for i, boundary in enumerate(boundaries):
        start = boundary["page"]
        end = boundaries[i + 1]["page"] if i + 1 < len(boundaries) else len(pages)
        raw_text = "".join(pages[start:end])
        chapters.append({
            "chapter_number": i + 1,
            "chapter_title": boundary["chapter_title"],
            "raw_text": raw_text,
        })
    return chapters
