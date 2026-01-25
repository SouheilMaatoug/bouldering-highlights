# --- Minimal helpers for Boulder parsing & smoothing (used by SceneSplitterOCR) ---
import re

_ROMAN = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}
_RE_NUM = re.compile(r"\bboulder\s*[:#\-]?\s*(\d{1,2})\b", re.IGNORECASE)
_RE_NUM_TIGHT = re.compile(r"\bboulder(\d{1,2})\b", re.IGNORECASE)
_RE_ROMAN = re.compile(r"\bboulder\s*[:#\-]?\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", re.IGNORECASE)
_RE_ROMAN_T = re.compile(r"\bboulder(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", re.IGNORECASE)


def _normalize(s):
    if not s:
        return ""
    s = s.lower()
    s = s.replace("0", "o").replace("|", "l")
    return s.strip()


def parse_boulder_number(text):
    """
    Returns (is_boulder: bool, number: int|None)
    Accepts: 'Boulder 1', 'Boulder1', 'Boulder I', and small OCR confusions.
    """
    t = _normalize(text)
    if "boulder" not in t:
        return False, None

    m = _RE_NUM.search(t) or _RE_NUM_TIGHT.search(t)
    if m:
        try:
            n = int(m.group(1))
            return True, n
        except:
            pass
    m = _RE_ROMAN.search(t) or _RE_ROMAN_T.search(t)
    if m:
        n = _ROMAN.get(m.group(1).lower())
        if n:
            return True, n

    # Single-char ambiguous tokens after 'boulder'
    m = re.search(r"\bboulder\s*[:#\-]?\s*([il|zs])\b", t) or re.search(r"\bboulder([il|zs])\b", t)
    if m:
        c = m.group(1).lower()
        if c in ("i", "l", "|"):
            return True, 1
        if c == "z":
            return True, 2
        if c == "s":
            return True, 5

    return True, None
