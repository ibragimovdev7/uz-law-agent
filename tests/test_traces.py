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
RUNS = sorted(p.stem for p in TRACES.glob("*.json"))


def trace(name):
    return json.loads((TRACES / f"{name}.json").read_text(encoding="utf-8"))


def calls(t):
    return [[c["name"] for c in s["calls"]] for s in t["trace"]]


def fetched(t):
    return {LAW.article_id(c["args"]["article"]) for s in t["trace"] for c in s["calls"] if c["name"] == "fetch_article"}


def cited_in_text(answer):
    return set(re.findall(r"\[(\d+(?:\.\d+)?)-modda\]", answer))


class EveryRun(unittest.TestCase):
    def test_every_run_replays_to_its_recorded_answer(self):
        self.assertTrue(RUNS)
        for name in RUNS:
            t = trace(name)
            self.assertEqual(run(t["question"], LAW, Replay(t["calls"], name), max_steps=t["max_steps"]),
                             (t["answer"], t["trace"]), name)

    def test_the_citations_were_read_and_match_the_text(self):
        for name in RUNS:
            answer = trace(name)["answer"]
            self.assertTrue(answer["citations"], name)
            self.assertLessEqual(set(answer["citations"]), fetched(trace(name)), name)
            self.assertEqual(cited_in_text(answer["answer"]), set(answer["citations"]), name)


class WhatTheRunsShow(unittest.TestCase):
    def test_q1_one_search_one_fetch_forty_hours(self):
        t = trace("q1_week")
        self.assertEqual(calls(t), [["search"], ["fetch_article"], ["answer"]])
        self.assertEqual(t["answer"]["citations"], ["182"])
        self.assertIn("qirq soat", t["answer"]["answer"])                    # 40 hours, as Art. 182 has it

    def test_q2_reads_182_and_415_in_one_turn(self):
        t = trace("q2_under18")
        self.assertEqual(calls(t), [["search"], ["fetch_article", "fetch_article"], ["answer"]])   # parallel calls
        self.assertEqual(fetched(t), {"182", "415"})
        self.assertEqual(t["answer"]["citations"], ["182", "415"])
        self.assertRegex(t["answer"]["answer"], r"(?s)40 soat.*36 soat.*24 soat")

    def test_q3_says_the_code_states_no_amount(self):
        t = trace("q3_min_wage")
        self.assertEqual(t["answer"]["citations"], ["245"])
        self.assertIn("Prezidenti tomonidan", t["answer"]["answer"])
        self.assertIn("soʻmdagi miqdori koʻrsatilmagan", t["answer"]["answer"])
        self.assertNotRegex(t["answer"]["answer"], r"\d\s*soʻm")             # no amount made up


if __name__ == "__main__":
    unittest.main()
