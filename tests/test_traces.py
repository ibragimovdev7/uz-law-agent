"""The recorded runs in traces/: each replays offline to its recorded answer, and the answers say what the law says.

    python -m unittest discover -s tests -v
"""
import json
import re
import unittest

from agent.corpus import LABOR_CODE, Law
from agent.llm import TRACES, Replay
from agent.loop import run

LAW = Law(LABOR_CODE)


def trace(name):
    return json.loads((TRACES / f"{name}.json").read_text(encoding="utf-8"))


def cited(answer):
    return set(re.findall(r"\[(\d+(?:\.\d+)?)-modda\]", answer))


def fetched(t):
    return {re.search(r"fetch_article\[(.*?)\]", s["model"])[1] for s in t["trace"] if "fetch_article[" in s["model"]}


class EveryRun(unittest.TestCase):
    def test_every_run_replays_to_its_recorded_answer(self):
        paths = sorted(TRACES.glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            t = trace(path.stem)
            got = run(t["question"], LAW, Replay(t["calls"], path.stem), max_steps=t["max_steps"], stop=t["stop"])
            self.assertEqual(got, (t["answer"], t["trace"]), path.stem)

    def test_every_cited_article_was_read_in_the_run(self):
        for path in sorted(TRACES.glob("*.json")):
            t = trace(path.stem)
            if t["answer"]:
                self.assertTrue(cited(t["answer"]), path.stem)
                self.assertLessEqual(cited(t["answer"]), fetched(t), path.stem)


class WhatTheRunsShow(unittest.TestCase):
    def test_q1_one_search_one_fetch_forty_hours(self):
        t = trace("q1_week")
        self.assertEqual([re.search(r"Action: (\w+)", s["model"])[1] for s in t["trace"]],
                         ["search", "fetch_article", "answer"])
        self.assertEqual(cited(t["answer"]), {"182"})
        self.assertIn("qirq soat", t["answer"])                          # 40 hours, as Art. 182 has it

    def test_q2_a_second_search_finds_415(self):
        t = trace("q2_under18")
        self.assertEqual(fetched(t), {"182", "415"})
        self.assertEqual(cited(t["answer"]), {"182", "415"})
        self.assertRegex(t["answer"], r"(?s)40 soat.*36 soat.*24 soat")
        searches = [s for s in t["trace"] if "Action: search[" in s["model"]]
        self.assertIn("o'n sakkiz", searches[1]["model"])                 # typed with a keyboard apostrophe ...
        self.assertTrue(searches[1]["observation"].startswith("415-modda"))   # ... and 415 still came first

    def test_q3_says_the_code_states_no_amount(self):
        t = trace("q3_min_wage")
        self.assertEqual(cited(t["answer"]), {"245"})
        self.assertIn("Prezidenti tomonidan", t["answer"])
        self.assertIn("aniq pul miqdori keltirilmagan", t["answer"])
        self.assertNotRegex(t["answer"], r"\d\s*soʻm")                    # no amount made up


if __name__ == "__main__":
    unittest.main()
