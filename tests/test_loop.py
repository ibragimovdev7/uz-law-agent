"""Tests for the loop, driven by a scripted model: no key, no network.

    python -m unittest discover -s tests -v
"""
import unittest

from agent.corpus import LABOR_CODE, Law
from agent.loop import NO_CALL, TOOLS, act, check, run, system_prompt

LAW = Law(LABOR_CODE)


def call(name, **args):
    return {"id": f"call_{name}", "name": name, "args": args}


def turn(*calls, text=""):
    return {"text": text, "calls": list(calls)}


FETCH_182 = turn(call("fetch_article", article="182"))
ANSWER_182 = turn(call("answer", answer="40 soat [182-modda]", citations=["182"]))


class Script:
    """A model that says what we tell it to, and remembers what it was asked."""
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, system, messages, tools):
        self.seen.append((system, list(messages), tools))
        return self.replies.pop(0)


class Schemas(unittest.TestCase):
    def test_every_tool_has_a_schema_with_typed_required_arguments(self):
        self.assertEqual([t["name"] for t in TOOLS], ["search", "fetch_article", "answer"])
        for t in TOOLS:
            p = t["parameters"]
            self.assertEqual((p["type"], p["additionalProperties"]), ("object", False), t["name"])
            self.assertEqual(set(p["required"]), set(p["properties"]), t["name"])
            self.assertTrue(all(v["type"] in ("string", "array") for v in p["properties"].values()), t["name"])

    def test_check_names_what_is_wrong(self):
        schema = TOOLS[2]["parameters"]
        self.assertIsNone(check({"answer": "x", "citations": ["182"]}, schema))
        self.assertEqual(check({"answer": "x"}, schema), "missing argument citations")
        self.assertEqual(check({"answer": "x", "citations": [], "k": 1}, schema), "unknown argument k")
        self.assertEqual(check({"answer": 40, "citations": []}, schema), "answer must be a string")
        self.assertEqual(check({"answer": "x", "citations": "182"}, schema), "citations must be a list of strings")
        self.assertEqual(check({"answer": "x", "citations": [182]}, schema), "citations must be a list of strings")
        self.assertEqual(check(["x"], schema), "the arguments must be an object")


class Actions(unittest.TestCase):
    def test_search_lists_articles_by_name(self):
        result, final = act(LAW, call("search", query="ish vaqtining normal davomiyligi"), set())
        self.assertTrue(result.startswith("182-modda: Ish vaqtining normal davomiyligi\n"))
        self.assertEqual((len(result.splitlines()), final), (5, None))

    def test_a_fetch_is_remembered_as_read(self):
        read = set()
        result, _ = act(LAW, call("fetch_article", article="182-modda"), read)
        self.assertTrue(result.startswith("182-modda. Ish vaqtining"))
        self.assertEqual(read, {"182"})

    def test_what_goes_wrong_is_said_not_raised(self):
        for c, expected in [(call("search", query="xyzzy"), "no article contains these words"),
                            (call("fetch_article", article="1182"), "error: there is no article 1182"),
                            (call("fetch_article", article=182), "error: article must be a string"),
                            (call("fetch_article"), "error: missing argument article"),
                            (call("calculator", expr="36/5"), "error: there is no tool calculator")]:
            result, final = act(LAW, c, set())
            self.assertTrue(result.startswith(expected), result)
            self.assertIsNone(final)

    def test_an_answer_may_cite_only_existing_articles_it_has_read(self):
        a = call("answer", answer="x [182-modda]", citations=["182-modda", "182"])
        self.assertEqual(act(LAW, a, {"182"}), ("accepted", {"answer": "x [182-modda]", "citations": ["182"]}))
        result, final = act(LAW, call("answer", answer="x", citations=["9999"]), {"182"})
        self.assertEqual((result[:30], final), ("error: there is no article 999", None))
        result, final = act(LAW, call("answer", answer="x", citations=["182", "415"]), {"182"})
        self.assertEqual((result, final), ("error: cite only articles you have read: fetch 415 first, or leave it "
                                           "out", None))
        self.assertEqual(act(LAW, call("answer", answer="the code does not say", citations=[]), set())[1],
                         {"answer": "the code does not say", "citations": []})

    def test_the_prompt_names_the_law(self):
        self.assertIn("Oʻzbekiston Respublikasining Mehnat kodeksi (581 articles)", system_prompt(LAW))


class Loop(unittest.TestCase):
    def test_results_go_back_and_answer_ends_the_loop(self):
        m = Script(FETCH_182, ANSWER_182)
        answer, trace = run("savol", LAW, m)
        self.assertEqual(answer, {"answer": "40 soat [182-modda]", "citations": ["182"]})
        self.assertEqual([s["step"] for s in trace], [1, 2])
        system, messages, tools = m.seen[1]
        self.assertEqual((system, tools), (system_prompt(LAW), TOOLS))
        self.assertEqual([x["role"] for x in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[1], {"role": "assistant", **FETCH_182})
        [result] = messages[2]["results"]
        self.assertEqual((result["id"], result["name"]), ("call_fetch_article", "fetch_article"))
        self.assertTrue(result["content"].startswith("182-modda. Ish vaqtining"))

    def test_parallel_calls_all_run_and_go_back_together(self):
        m = Script(turn(call("fetch_article", article="182"), call("fetch_article", article="415")),
                   turn(call("answer", answer="x", citations=["182", "415"])))
        answer, trace = run("savol", LAW, m)
        self.assertEqual(answer["citations"], ["182", "415"])
        self.assertEqual([r["content"][:9] for r in m.seen[1][1][2]["results"]], ["182-modda", "415-modda"])

    def test_a_malformed_call_gets_an_error_and_the_run_goes_on(self):
        m = Script(turn(call("fetch_article", number="182")), FETCH_182, ANSWER_182)
        answer, trace = run("savol", LAW, m)
        self.assertEqual(trace[0]["results"][0]["content"], "error: missing argument article; unknown argument number")
        self.assertEqual(answer["citations"], ["182"])

    def test_an_answer_citing_an_unread_article_is_sent_back(self):
        m = Script(turn(call("answer", answer="x", citations=["182"])), FETCH_182, ANSWER_182)
        answer, trace = run("savol", LAW, m)
        self.assertTrue(trace[0]["results"][0]["content"].startswith("error: cite only articles you have read"))
        self.assertEqual(len(trace), 3)
        self.assertEqual(answer["citations"], ["182"])

    def test_a_turn_without_a_call_is_asked_for_one(self):
        m = Script(turn(text="I think it is 40 hours."), FETCH_182, ANSWER_182)
        answer, trace = run("savol", LAW, m)
        self.assertEqual(m.seen[1][1][-1], {"role": "user", "content": NO_CALL})
        self.assertEqual(answer["citations"], ["182"])

    def test_the_step_limit_stops_a_loop_that_never_answers(self):
        m = Script(*[turn(call("search", query="ish vaqti"))] * 3)
        answer, trace = run("savol", LAW, m, max_steps=3)
        self.assertIsNone(answer)
        self.assertEqual(len(trace), 3)


if __name__ == "__main__":
    unittest.main()
