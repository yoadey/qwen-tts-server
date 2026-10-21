import unittest
from types import SimpleNamespace

from batching import MIN_FRAMES, frame_cap, select


def items(*lengths):
    return [SimpleNamespace(text="x" * n) for n in lengths]


class SelectTest(unittest.TestCase):
    def test_cap(self):
        self.assertEqual(frame_cap("short"), MIN_FRAMES)
        self.assertEqual(frame_cap("x" * 100), 375)

    def test_rows(self):
        batch, rest = select(items(100, 100, 100), 2, 10**6)
        self.assertEqual((len(batch), len(rest)), (2, 1))

    def test_budget_by_longest(self):
        # 2 x 375 fits 800; a third row would make 3 x 375.
        batch, rest = select(items(100, 100, 100), 8, 800)
        self.assertEqual((len(batch), len(rest)), (2, 1))
        # A long one would make every row as long: it waits, a short one joins.
        q = items(100, 400, 100)
        batch, rest = select(q, 8, 800)
        self.assertEqual(batch, [q[0], q[2]])
        self.assertEqual(rest, [q[1]])

    def test_oldest_always_goes(self):
        q = items(2000, 10)
        batch, rest = select(q, 8, 100)
        self.assertEqual((batch, rest), ([q[0]], [q[1]]))

    def test_empty(self):
        self.assertEqual(select([], 8, 100), ([], []))


if __name__ == "__main__":
    unittest.main()
