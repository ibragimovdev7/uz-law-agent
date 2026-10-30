"""Tests for the loop, driven by a scripted model: no key, no network.

    python -m unittest discover -s tests -v
"""
import unittest

from agent.corpus import LABOR_CODE, Law
from agent.loop import CANNOT_READ, STOP, act, run, system_prompt

LAW = Law(LABOR_CODE)


class Script:
    """A model that says what we tell it to, and remembers what it was asked."""
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, system, messages, stop=None):
        self.seen.append((system, list(messages), stop))
        return self.replies.pop(0)


class Actions(unittest.TestCase):
    def test_search_lists_articles_by_name(self):
        obs = act(LAW, "search", "ish vaqtining normal davomiyligi")
        self.assertTrue(obs.startswith("182-modda: Ish vaqtining normal davomiyligi\n"))
        self.assertEqual(len(obs.splitlines()), 5)

    def test_what_goes_wrong_is_said_not_raised(self):
        self.assertIn("no article contains these words", act(LAW, "search", "xyzzy"))
        self.assertTrue(act(LAW, "fetch_article", "182-modda").startswith("182-modda. Ish vaqtining"))
        self.assertEqual(act(LAW, "fetch_article", "1182"), "there is no article 1182")
        self.assertIn("unknown action calculator", act(LAW, "calculator", "36/5"))

    def test_the_prompt_names_the_law(self):
        prompt = system_prompt(LAW)
        self.assertIn("Oʻzbekiston Respublikasining Mehnat kodeksi (581 articles)", prompt)
        self.assertIn("[182-modda]", prompt)


class Loop(unittest.TestCase):
    def test_observations_go_back_and_answer_ends_the_loop(self):
        m = Script("Thought: look\nAction: fetch_article[182]",
                   "Thought: done\nAction: answer[40 soat [182-modda]]")
        answer, trace = run("savol", LAW, m)
        self.assertEqual(answer, "40 soat [182-modda]")
        self.assertEqual([s["step"] for s in trace], [1, 2])
        system, messages, stop = m.seen[1]
        self.assertEqual(system, system_prompt(LAW))
        self.assertEqual([x["role"] for x in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[0]["content"], "savol")
        self.assertTrue(messages[2]["content"].startswith("Observation: 182-modda. Ish vaqtining"))
        self.assertEqual(stop, STOP)                                           # cut off at "Observation:"

    def test_the_step_limit_stops_a_loop_that_never_answers(self):
        m = Script(*["Thought: again\nAction: search[ish vaqti]"] * 3)
        answer, trace = run("savol", LAW, m, max_steps=3)
        self.assertIsNone(answer)
        self.assertEqual(len(trace), 3)

    def test_an_unreadable_turn_gets_a_correction_not_a_crash(self):
        m = Script("I think it is 40 hours.", "Thought: format\nAction: answer[40 soat]")
        answer, trace = run("savol", LAW, m)
        self.assertEqual(answer, "40 soat")
        self.assertEqual(trace[0]["observation"], CANNOT_READ)

    def test_a_multi_line_answer_is_read_whole(self):
        m = Script("Thought: ok\nAction: answer[40 soat [182-modda].\n36 soat [415-modda].]")
        self.assertEqual(run("savol", LAW, m)[0], "40 soat [182-modda].\n36 soat [415-modda].")

    def test_without_stop_the_model_is_not_cut_off(self):
        m = Script("Thought: ok\nAction: answer[x]")
        run("savol", LAW, m, stop=False)
        self.assertIsNone(m.seen[0][2])


if __name__ == "__main__":
    unittest.main()
