"""Tests for the model layer: the Gemini request and reply, recording and replay. No key, no network.

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent.__main__ as cli
from agent.corpus import LABOR_CODE, Law
from agent.llm import Recorder, Replay, reply, request_body
from agent.loop import run
from test_loop import Script

LAW = Law(LABOR_CODE)
TURNS = ("Thought: look\nAction: fetch_article[182]", "Thought: done\nAction: answer[40 soat [182-modda]]")


class GeminiRequest(unittest.TestCase):
    def test_roles_system_and_stop(self):
        body = request_body("sys", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                                    {"role": "user", "content": "Observation: o"}], ["Observation:"])
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "sys"}]})
        self.assertEqual([(c["role"], c["parts"][0]["text"]) for c in body["contents"]],
                         [("user", "q"), ("model", "a"), ("user", "Observation: o")])
        self.assertEqual(body["generationConfig"], {"stopSequences": ["Observation:"]})
        self.assertNotIn("generationConfig", request_body("sys", [{"role": "user", "content": "q"}]))

    def test_a_model_turn_goes_back_as_gemini_wrote_it(self):
        sent = {"role": "model", "parts": [{"text": "a", "thoughtSignature": "c2ln"}]}
        body = request_body("sys", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                            replies={"a": sent})
        self.assertEqual(body["contents"][1], sent)

    def test_the_reply_text_leaves_out_thoughts(self):
        response = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                          {"text": "Thought: x\n"}, {"text": "Action: answer[y]"}]},
                                    "finishReason": "STOP"}]}
        text, content = reply(response)
        self.assertEqual(text, "Thought: x\nAction: answer[y]")
        self.assertEqual(content["role"], "model")

    def test_a_blocked_prompt_is_an_error(self):
        with self.assertRaises(RuntimeError):
            reply({"promptFeedback": {"blockReason": "SAFETY"}})


class RecordAndReplay(unittest.TestCase):
    def test_a_recorded_run_replays_to_the_same_answer(self):
        recorder = Recorder(Script(*TURNS))
        answer, trace = run("savol", LAW, recorder)
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(run("savol", LAW, Replay(recorder.calls)), (answer, trace))

    def test_a_changed_prompt_fails_loudly(self):
        recorder = Recorder(Script(*TURNS))
        run("savol", LAW, recorder)
        with self.assertRaisesRegex(RuntimeError, "call 1 differs"):
            run("boshqa savol", LAW, Replay(recorder.calls))

    def test_the_command_line_replays_a_saved_run(self):
        recorder = Recorder(Script(*TURNS))
        answer, trace = run("savol", LAW, recorder)
        with tempfile.TemporaryDirectory() as d:
            saved = {"model": "scripted", "question": "savol", "max_steps": 8, "stop": True,
                     "answer": answer, "trace": trace, "calls": recorder.calls}
            Path(d, "demo.json").write_text(json.dumps(saved), encoding="utf-8")
            out = io.StringIO()
            with mock.patch.object(cli, "TRACES", Path(d)), contextlib.redirect_stdout(out):
                cli.main(["--replay", "demo"])
        self.assertIn("ANSWER: 40 soat [182-modda]", out.getvalue())


if __name__ == "__main__":
    unittest.main()
