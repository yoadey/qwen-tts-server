"""Splitting text into the pieces the server generates in one batch."""

import math
import re

# The end of a sentence, with a closing quote or bracket kept with it
# (English and German quotes: ” ’ “ « »), before whitespace.
_END = re.compile(r"([.!?…]+[\"'”’“«»)\]]*)\s+")
_PARAGRAPH = re.compile(r"\n\s*\n")


def sentences(text: str, min_chars: int) -> list[str]:
    """Splits text into sentences of at least min_chars (a short one joins
    the next; a short last one joins the one before)."""
    parts = [s.strip() for s in _PARAGRAPH.split(_END.sub("\\1\n\n", text)) if s.strip()]
    out: list[str] = []
    for s in parts:
        if out and len(out[-1]) < min_chars:
            out[-1] += " " + s
        else:
            out.append(s)
    if len(out) > 1 and len(out[-1]) < min_chars:
        last = out.pop()
        out[-1] += " " + last
    return out


def pieces(text: str, min_chars: int, max_chars: int) -> list[str]:
    """Splits text into the pieces generated as one: whole, if it has at
    most max_chars, else its sentences joined into pieces of about equal
    length (one sentence longer than max_chars stays whole). With max_chars
    0, every sentence is a piece of its own."""
    parts = sentences(text, min_chars)
    if max_chars <= 0 or not parts:
        return parts
    total = len(" ".join(parts))
    target = total / math.ceil(total / max_chars)
    out = [parts[0]]
    for s in parts[1:]:
        if len(out[-1]) < target and len(out[-1]) + 1 + len(s) <= max_chars:
            out[-1] += " " + s
        else:
            out.append(s)
    return out
