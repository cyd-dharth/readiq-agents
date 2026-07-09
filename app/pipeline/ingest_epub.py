import logging
import zipfile
from io import BytesIO
from pathlib import Path

log = logging.getLogger(__name__)

# Module-level set of patterns that identify non-chapter content.
# Lowercase, checked as substring or full match against normalised title.
_NON_CHAPTER_PATTERNS = frozenset([
    "contents",
    "table of contents",
    "preface",
    "foreword",
    "introduction",
    "acknowledgement",
    "acknowledgment",
    "about the author",
    "about the book",
    "bibliography",
    "references",
    "index",
    "appendix",
    "footnote",
    "footnotes",
    "endnote",
    "endnotes",
    "glossary",
    "license",
    "licence",
    "copyright",
    "legal",
    "disclaimer",
    "dedication",
    "epigraph",
    "colophon",
    "gutenberg",
    "project gutenberg",
    "full project gutenberg",
])


def _is_non_chapter(title: str | None, text: str) -> bool:
    """
    Return True if this spine item should be excluded from chapter processing.
    Uses title matching first, then falls back to text heuristics.
    """
    if title:
        normalised = title.lower().strip()

        # Exact match against known non-chapter titles
        if normalised in _NON_CHAPTER_PATTERNS:
            return True

        # Substring match: "preface by lionel giles" contains "preface"
        for pattern in _NON_CHAPTER_PATTERNS:
            if pattern in normalised:
                return True

    # Text-based heuristic: very short items with no real content
    # (title pages, separator pages, etc.)
    if len(text.strip()) < 150:
        return True

    return False


def ingest_epub(file_path: str) -> list[dict]:
    """
    Parse an EPUB file and return a list of chapters with their text content.
    Returns the same structure as ingest_pdf so summarize_from_chapters
    can consume both without modification.

    Return format:
    [{"chapter_number": int, "chapter_title": str | None, "raw_text": str}]

    Never raises. Falls back to whole-book single chapter on any error.
    """

    try:
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
    except ImportError as exc:
        log.warning(
            "ebooklib or beautifulsoup4 not installed: %s. "
            "Add ebooklib and beautifulsoup4 to requirements.txt. "
            "Returning whole-book fallback.",
            exc,
        )
        return [{"chapter_number": 1, "chapter_title": None, "raw_text": ""}]

    full_text = ""

    try:
        book = epub.read_epub(file_path)

        # Step 1: build a title map from the TOC
        # epub TOC is a nested structure of tuples: (Section, [children]) or Link objects
        title_map = _extract_toc_titles(book)

        # Step 2: get spine items in reading order
        # book.spine is a list of (idref, linear) tuples
        # We only want linear items (linear='yes' or linear=True)
        spine_ids = [
            idref
            for idref, linear in book.spine
            if str(linear).lower() not in ("no", "false", "0")
        ]

        if not spine_ids:
            log.warning(
                "EPUB spine is empty or all items are non-linear. "
                "Falling back to all document items. File: %s",
                file_path,
            )
            spine_ids = [
                item.get_id()
                for item in book.get_items_of_type(ITEM_DOCUMENT)
            ]

        # Step 3: extract text from each spine item
        raw_chapters: list[dict] = []

        for idref in spine_ids:
            item = book.get_item_with_id(idref)
            if item is None:
                continue

            html_content = item.get_content()
            text = _html_to_text(html_content)

            if not text.strip():
                continue

            # Try to get title from TOC map first, then from HTML
            title = (
                title_map.get(idref)
                or title_map.get(item.get_name())
                or _extract_title_from_html(html_content)
            )

            raw_chapters.append({
                "idref": idref,
                "chapter_title": title,
                "raw_text": text,
            })

        if not raw_chapters:
            log.warning(
                "EPUB produced no text content after parsing all spine items. "
                "File may be image-based or have no extractable text. "
                "File: %s",
                file_path,
            )
            return [{"chapter_number": 1, "chapter_title": None, "raw_text": ""}]

        # Step 3B: filter out non-chapter content
        content_chapters = [
            ch for ch in raw_chapters
            if not _is_non_chapter(ch["chapter_title"], ch["raw_text"])
        ]

        if not content_chapters:
            log.warning(
                "EPUB filtering removed all chapters. "
                "Falling back to unfiltered list. File: %s",
                file_path,
            )
            content_chapters = raw_chapters

        log.info(
            "EPUB filtering: %d spine items -> %d content chapters. "
            "Removed: %s. File: %s",
            len(raw_chapters),
            len(content_chapters),
            [ch["chapter_title"] for ch in raw_chapters
            if ch not in content_chapters],
            file_path,
        )

        # Step 4: merge short items (operate on filtered list)
        merged = _merge_short_chapters(content_chapters, min_chars=300)

        # Step 4: merge suspiciously short items into the previous chapter
        # Some EPUBs split a single chapter across multiple small HTML files
        # (e.g. a title page file followed by the chapter content file).
        # Merge any item under 300 characters into the previous one.
        merged = _merge_short_chapters(raw_chapters, min_chars=300)

        # Step 5: check total extractable text
        full_text = " ".join(ch["raw_text"] for ch in merged)
        if len(full_text.strip()) < 500:
            log.warning(
                "EPUB appears to be image-based or has no extractable text. "
                "Total text under 500 characters. "
                "File: %s",
                file_path,
            )
            return [{"chapter_number": 1, "chapter_title": None, "raw_text": full_text}]

        # Step 6: assign chapter numbers and return
        result = [
            {
                "chapter_number": idx + 1,
                "chapter_title": ch["chapter_title"],
                "raw_text": ch["raw_text"],
            }
            for idx, ch in enumerate(merged)
        ]

        log.info(
            "EPUB ingest complete: %d chapters found. File: %s",
            len(result),
            file_path,
        )
        return result

    except Exception as exc:
        log.warning(
            "EPUB ingest failed: %s. Returning whole-book fallback. File: %s",
            exc,
            file_path,
        )
        return [{"chapter_number": 1, "chapter_title": None, "raw_text": full_text}]


def _extract_toc_titles(book) -> dict[str, str]:
    """
    Walk the EPUB TOC and build a mapping of idref or href -> chapter title.
    Handles both NCX-style and nav-style TOCs via ebooklib's unified interface.
    Returns a flat dict so lookups are O(1).
    """
    from ebooklib import epub

    title_map: dict[str, str] = {}

    def _walk(toc_items):
        """Recursively walk nested TOC entries (Section, [children]) tuples or Link objects, populating title_map."""
        for item in toc_items:
            if isinstance(item, tuple):
                # (Section, [children]) tuple
                section, children = item
                if hasattr(section, "href") and hasattr(section, "title"):
                    # href may include a fragment: chapter01.html#section1
                    # we want just the file part for matching
                    href = section.href.split("#")[0]
                    if section.title:
                        title_map[href] = section.title.strip()
                _walk(children)
            elif hasattr(item, "href") and hasattr(item, "title"):
                href = item.href.split("#")[0]
                if item.title:
                    title_map[href] = item.title.strip()

    _walk(book.toc)
    return title_map


def _html_to_text(html_bytes: bytes) -> str:
    """
    Extract clean plain text from an HTML chapter file.
    Removes script, style, and nav tags entirely.
    Preserves paragraph breaks as double newlines.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_bytes, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "aside", "figure", "figcaption"]):
        tag.decompose()

    # Extract text with paragraph breaks
    # Use separator so block elements produce newlines
    text = soup.get_text(separator="\n")

    # Collapse excessive blank lines to at most two
    lines = text.splitlines()
    cleaned: list[str] = []
    blank_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped:
            blank_count = 0
            cleaned.append(stripped)
        else:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")

    return "\n".join(cleaned).strip()


def _extract_title_from_html(html_bytes: bytes) -> str | None:
    """
    Try to find a chapter title from the HTML content itself
    when the TOC map has no entry for this item.
    Looks for the first h1, h2, or h3 tag.
    Returns None if nothing useful is found.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag_name in ("h1", "h2", "h3"):
        tag = soup.find(tag_name)
        if tag:
            text = tag.get_text(strip=True)
            if text and len(text) < 120:
                return text
    return None


def _merge_short_chapters(
    chapters: list[dict], min_chars: int = 300
) -> list[dict]:
    """
    Merge chapters that are too short into the preceding chapter.
    This handles EPUBs that split a chapter across multiple small HTML files,
    for example a standalone title page followed by the chapter body.
    The first item is never merged away regardless of length.
    """
    if not chapters:
        return chapters

    merged: list[dict] = [chapters[0]]

    for ch in chapters[1:]:
        if len(ch["raw_text"].strip()) < min_chars:
            # Append to previous chapter's text
            merged[-1]["raw_text"] = (
                merged[-1]["raw_text"] + "\n\n" + ch["raw_text"]
            )
            # Keep previous chapter's title unless it had none
            if merged[-1]["chapter_title"] is None and ch["chapter_title"]:
                merged[-1]["chapter_title"] = ch["chapter_title"]
        else:
            merged.append(ch)

    return merged