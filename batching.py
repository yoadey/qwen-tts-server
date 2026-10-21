"""Choosing the sentences generated together in one batch."""

import math

# The 12Hz codec makes 12.5 frames per second of speech. Slow speech is
# about 10 characters per second; a sentence may take three times that
# before it is cut, so a run that does not end cannot grow without bound.
FRAMES_PER_CHAR = 12.5 / 10 * 3
MIN_FRAMES = 75


def frame_cap(text: str) -> int:
    """The most frames a sentence may take."""
    return max(MIN_FRAMES, math.ceil(len(text) * FRAMES_PER_CHAR))


def select(queue: list, max_rows: int, max_row_frames: int) -> tuple[list, list]:
    """Splits the queue (items with .text, oldest first) into a batch and
    the rest. The oldest always goes; others join, in order, while the
    batch stays within max_rows and within max_row_frames for its rows
    times its longest cap (every row of a batch runs, and holds memory, as
    long as its longest sentence)."""
    if not queue:
        return [], []
    batch, rest = [queue[0]], []
    longest = frame_cap(queue[0].text)
    for item in queue[1:]:
        cap = max(longest, frame_cap(item.text))
        if len(batch) < max_rows and (len(batch) + 1) * cap <= max_row_frames:
            batch.append(item)
            longest = cap
        else:
            rest.append(item)
    return batch, rest
