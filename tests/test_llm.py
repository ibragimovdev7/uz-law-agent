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
from agent.llm import Recorder, Replay, reply, request_body, turn_key
from agent.loop import TOOLS, run
from test_loop import ANSWER_182, FETCH_182, Script

LAW = Law(LABOR_CODE)
CALL = {"id": "call_1", "name": "fetch_article", "args": {"article": "182"}}
MESSAGES = [{"role": "user", "content": "q"},
            {"role": "assistant", "text": "", "calls": [CALL]},
            {"role": "tool", "results": [{"id": "call_1", "name": "fetch_article", "content": "182-modda. ..."}]}]


class GeminiRequest(unittest.TestCase):
    def test_tools_become_function_declarations_and_every_turn_is_a_call(self):
        body = request_body("sys", MESSAGES[:1], TOOLS)
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "sys"}]})
        [declarations] = body["tools"]
        self.assertEqual([d["name"] for d in declarations["functionDeclarations"]], ["search", "fetch_article", "answer"])
        self.assertEqual(declarations["functionDeclarations"][1]["parametersJsonSchema"], TOOLS[1]["parameters"])
        self.assertEqual(body["toolConfig"], {"functionCallingConfig": {"mode": "ANY"}})

    def test_roles_calls_and_results(self):
        contents = request_body("sys", MESSAGES, TOOLS)["contents"]
        self.assertEqual([c["role"] for c in contents], ["user", "model", "user"])
        self.assertEqual(contents[1]["parts"], [{"functionCall": {"name": "fetch_article", "args": {"article": "182"},
                                                                  "id": "call_1"}}])
        self.assertEqual(contents[2]["parts"], [{"functionResponse": {"name": "fetch_article",
                                                                      "response": {"result": "182-modda. ..."},
                                                                      "id": "call_1"}}])

    def test_a_model_turn_goes_back_as_gemini_wrote_it(self):
        sent = {"role": "model", "parts": [{"functionCall": {"name": "fetch_article", "args": {"article": "182"},
                                                             "id": "call_1"}, "thoughtSignature": "c2ln"}]}
        contents = request_body("sys", MESSAGES, TOOLS, sent={turn_key(MESSAGES[1]): sent})["contents"]
        self.assertEqual(contents[1], sent)

    def test_the_reply_is_text_without_thoughts_and_the_calls(self):
        response = {"candidates": [{"content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": "Let me read it."},
            {"functionCall": {"name": "fetch_article", "args": {"article": "182"}, "id": "call_1"},
             "thoughtSignature": "c2ln"}]}, "finishReason": "STOP"}]}
        answer, content = reply(response)
        self.assertEqual(answer, {"text": "Let me read it.", "calls": [CALL]})
        self.assertEqual((content["role"], len(content["parts"])), ("model", 3))

    def test_missing_parts_and_a_blocked_prompt(self):
        self.assertEqual(reply({"candidates": [{"finishReason": "STOP"}]})[0], {"text": "", "calls": []})
        with self.assertRaises(RuntimeError):
            reply({"promptFeedback": {"blockReason": "SAFETY"}})


class RecordAndReplay(unittest.TestCase):
    def test_a_recorded_run_replays_to_the_same_answer(self):
        recorder = Recorder(Script(FETCH_182, ANSWER_182))
        answer, trace = run("savol", LAW, recorder)
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(run("savol", LAW, Replay(recorder.calls)), (answer, trace))

    def test_a_changed_prompt_fails_loudly(self):
        recorder = Recorder(Script(FETCH_182, ANSWER_182))
        run("savol", LAW, recorder)
        with self.assertRaisesRegex(RuntimeError, "call 1 differs"):
            run("boshqa savol", LAW, Replay(recorder.calls))

    def test_the_command_line_replays_a_saved_run(self):
        recorder = Recorder(Script(FETCH_182, ANSWER_182))
        answer, trace = run("savol", LAW, recorder)
        with tempfile.TemporaryDirectory() as d:
            saved = {"model": "scripted", "question": "savol", "max_steps": 8, "answer": answer, "trace": trace,
                     "calls": json.loads(json.dumps(recorder.calls))}
            Path(d, "demo.json").write_text(json.dumps(saved), encoding="utf-8")
            out = io.StringIO()
            with mock.patch.object(cli, "TRACES", Path(d)), contextlib.redirect_stdout(out):
                cli.main(["--replay", "demo"])
        self.assertIn("ANSWER: 40 soat [182-modda]\nCITATIONS: 182", out.getvalue())


if __name__ == "__main__":
    unittest.main()
