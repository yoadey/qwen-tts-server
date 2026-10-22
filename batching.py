"""Choosing the sentences generated together in one batch."""

import math

# The 12Hz codec makes 12.5 frames per second of speech. Speech is about
# 10 characters per second; a sentence may take twice that (half the pace)
# before it is cut, so a run that does not end cannot grow without bound.
# The cap also bounds a batch (select), so a lower one lets more rows in.
FRAME_RATE = 12.5
FRAMES_PER_CHAR = FRAME_RATE / 10 * 2
MIN_FRAMES = 75


def frame_cap(text: str) -> int:
    """The most frames a sentence may take."""
    return max(MIN_FRAMES, math.ceil(len(text) * FRAMES_PER_CHAR))


def select(queue: list, max_rows: int, max_row_frames: int) -> tuple[list, list]:
    """Splits the queue (items with .text, oldest first) into a batch and
    the rest. The oldest always goes (so none waits forever); the others
    closest to it in length join while the batch stays within max_rows and
    within max_row_frames for its rows times its longest cap. Every row of
    a batch runs, and holds memory, as long as its longest sentence, so
    sentences of a length together waste the least."""
    if not queue:
        return [], []
    head = frame_cap(queue[0].text)
    batch, longest = [queue[0]], head
    for item in sorted(queue[1:], key=lambda i: abs(frame_cap(i.text) - head)):
        cap = max(longest, frame_cap(item.text))
        if len(batch) < max_rows and (len(batch) + 1) * cap <= max_row_frames:
            batch.append(item)
            longest = cap
    chosen = {id(i) for i in batch}
    return batch, [i for i in queue if id(i) not in chosen]
