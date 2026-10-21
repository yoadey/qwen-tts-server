import unittest

from text import sentences


class SentencesTest(unittest.TestCase):
    def test_split(self):
        self.assertEqual(
            sentences("The rain had not stopped for three days. Then the bolt slid back! Who is there?", 10),
            ["The rain had not stopped for three days.", "Then the bolt slid back!", "Who is there?"])

    def test_short_joins_next(self):
        self.assertEqual(sentences("No. I will not go out there tonight, not for anyone.", 40),
                         ["No. I will not go out there tonight, not for anyone."])

    def test_short_last_joins_previous(self):
        self.assertEqual(sentences("I will not go out there tonight, not for anyone. Never.", 40),
                         ["I will not go out there tonight, not for anyone. Never."])

    def test_closing_quotes_stay(self):
        self.assertEqual(sentences('„Komm sofort her!“ Sie lief los. "Wait..." he said.', 1),
                         ["„Komm sofort her!“", "Sie lief los.", '"Wait..."', "he said."])
        self.assertEqual(sentences("(It was late.) She ran.", 1), ["(It was late.)", "She ran."])

    def test_paragraphs(self):
        self.assertEqual(sentences("First line without a stop\n\nSecond one", 1),
                         ["First line without a stop", "Second one"])

    def test_empty(self):
        self.assertEqual(sentences("  \n\n ", 40), [])


if __name__ == "__main__":
    unittest.main()
