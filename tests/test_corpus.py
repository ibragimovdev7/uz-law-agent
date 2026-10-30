"""Tests for the Labor Code data, its source note, and the two tools. Offline: no key, no network.

    python -m unittest discover -s tests -v
"""
import hashlib
import re
import tempfile
import unittest
from pathlib import Path

from agent.corpus import LABOR_CODE, Law

LAW = Law(LABOR_CODE)
NOTE = LABOR_CODE.with_suffix(".SOURCE.md").read_text(encoding="utf-8")


class TheFile(unittest.TestCase):
    """Every number the source note states, recomputed from the file."""
    def test_it_is_the_pinned_file(self):
        self.assertIn(hashlib.sha256(LABOR_CODE.read_bytes()).hexdigest(), NOTE)

    def test_the_articles_are_1_to_581_with_no_gaps(self):
        self.assertEqual(list(LAW.articles), [str(n) for n in range(1, 582)])
        self.assertEqual(len(LABOR_CODE.read_text(encoding="utf-8").splitlines()), 581)
        self.assertEqual({a["kind"] for a in LAW.articles.values()}, {"base"})

    def test_the_counts_in_the_note(self):
        a = LAW.articles.values()
        self.assertFalse([x for x in a if not x["title"].strip() or not x["text"].strip()])
        lengths = sorted(len(x["text"]) for x in a)
        self.assertEqual((lengths[0], lengths[len(lengths) // 2], lengths[-1]), (97, 840, 6045))
        self.assertEqual(sum(len((x["title"] + " " + x["text"]).split()) for x in a), 80270)
        self.assertEqual(len({x["section"] for x in a}), 7)
        self.assertEqual(len({x["chapter"] for x in a}), 34)

    def test_the_spelling_in_the_note(self):
        text = "".join(x["title"] + x["text"] for x in LAW.articles.values())
        self.assertEqual((text.count("ʻ"), text.count("ʼ")), (9555, 1210))
        self.assertFalse([c for c in "'‘’`" if c in text])

    def test_facts_the_agent_will_be_asked_about(self):
        self.assertIn("haftasiga qirq soatdan ortiq", LAW.fetch_article("182"))          # 40 hours a week
        a415 = LAW.fetch_article("415")
        self.assertIn("oʻn oltidan oʻn sakkiz yoshgacha", a415)                         # ages 16-18 ...
        self.assertIn("haftasiga oʻttiz olti soatdan", a415)                             # ... 36 hours
        self.assertIn("oʻn beshdan oʻn olti yoshgacha", a415)                            # ages 15-16 ...
        self.assertIn("haftasiga yigirma toʻrt soatdan", a415)                           # ... 24 hours
        self.assertIn("Prezidenti tomonidan", LAW.fetch_article("245"))                  # the minimum wage is set
                                                                                         # by the President ...
        self.assertFalse([k for k, a in LAW.articles.items() if re.search(r"\d\s*soʻm", a["text"])])  # no amount
        self.assertIn("milliy valyutada (soʻmda)", LAW.fetch_article("256"))            # "soʻm" only as the currency


class Search(unittest.TestCase):
    def test_the_right_article_comes_first(self):
        self.assertEqual(LAW.search("ish vaqtining normal davomiyligi")[0][0], "182")
        self.assertEqual(LAW.search("oʻn sakkiz yosh qisqartirilgan ish vaqti")[0][0], "415")
        self.assertEqual(LAW.search("mehnatga haq toʻlashning eng kam miqdori")[0][0], "245")

    def test_a_keyboard_apostrophe_finds_the_same_articles(self):
        self.assertEqual(LAW.search("o'n sakkiz yosh"), LAW.search("oʻn sakkiz yosh"))
        self.assertEqual(LAW.search("ma'lumot"), LAW.search("maʼlumot"))

    def test_the_result_shape(self):
        hits = LAW.search("ish vaqtining davomiyligi", k=3)
        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0], ("182", "Ish vaqtining normal davomiyligi", 6))       # 3 stems, all in the title
        self.assertEqual([n for *_, n in hits], sorted((n for *_, n in hits), reverse=True))

    def test_no_shared_word_gives_no_hits(self):
        self.assertEqual(LAW.search("xyzzy"), [])
        self.assertEqual(LAW.search(""), [])


class Fetch(unittest.TestCase):
    def test_what_a_model_may_write(self):
        expected = LAW.fetch_article("182")
        self.assertTrue(expected.startswith("182-modda. Ish vaqtining normal davomiyligi\nXodim uchun"))
        for ref in ["182-modda", "182- modda", " 182 ", "Art. 182", "art 182"]:
            self.assertEqual(LAW.fetch_article(ref), expected, ref)

    def test_no_such_article_is_none(self):
        for ref in ["1182", "0", "abc", "", "182, 183", "182.1"]:
            self.assertIsNone(LAW.fetch_article(ref), ref)
            self.assertIsNone(LAW.article_id(ref), ref)

    def test_the_id_of_what_a_model_may_write(self):
        self.assertEqual({LAW.article_id(r) for r in ["182", "182-modda", "Art. 182", " 182 "]}, {"182"})


class Loading(unittest.TestCase):
    def test_a_repeated_article_id_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "law.jsonl"
            record = '{"article_id": "1", "article_display": "1-modda", "title": "t", "text": "x"}'
            path.write_text(record + "\n" + record + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                Law(path)


if __name__ == "__main__":
    unittest.main()
