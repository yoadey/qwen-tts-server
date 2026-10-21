"""Splitting text into the sentences the server generates in one batch."""

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
