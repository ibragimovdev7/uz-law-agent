"""Tests for the web server, with a scripted model: no key, no network beyond this computer.

    python -m unittest discover -s tests -v
"""
import http.client
import json
import threading
import unittest

from agent.corpus import LABOR_CODE, Law
from agent.web import make_server
from test_loop import ANSWER_182, FETCH_182, Script

LAW = Law(LABOR_CODE)


class Server(unittest.TestCase):
    model = staticmethod(lambda: Script(FETCH_182, ANSWER_182))

    def setUp(self):
        self.server = make_server(0, law=LAW, model=lambda: self.model())
        self.server.RequestHandlerClass.log_message = lambda *args: None          # keep the test output clean
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=body, headers={"Host": f"127.0.0.1:{self.port}", **(headers or {})})
        r = conn.getresponse()
        data = r.read().decode()
        conn.close()
        return r.status, r.getheader("Content-Type"), data

    def ask(self, question, content_type="application/json"):
        return self.request("POST", "/api/ask", json.dumps({"question": question}), {"Content-Type": content_type})


class Pages(Server):
    def test_the_page_and_the_law(self):
        status, kind, body = self.request("GET", "/")
        self.assertEqual((status, kind), (200, "text/html; charset=utf-8"))
        self.assertIn("<title>Mehnat kodeksi agenti</title>", body)
        status, _, body = self.request("GET", "/api/law")
        self.assertEqual(json.loads(body), {"title": "Oʻzbekiston Respublikasining Mehnat kodeksi", "articles": 581})

    def test_an_article_by_what_a_reader_may_write(self):
        status, _, body = self.request("GET", "/api/article/182-modda")
        self.assertEqual((status, json.loads(body)["article_id"]), (200, "182"))
        self.assertEqual(self.request("GET", "/api/article/9999")[0], 404)

    def test_the_recorded_runs(self):
        status, _, body = self.request("GET", "/api/runs")
        runs = {r["name"]: r["question"] for r in json.loads(body)}
        self.assertEqual(runs["q1_week"], "Normal ish haftasi necha soat?")
        status, _, body = self.request("GET", "/api/runs/q2_under18")
        self.assertEqual(json.loads(body)["answer"]["citations"], ["182", "415"])
        self.assertEqual(self.request("GET", "/api/runs/..%2F.env")[0], 404)      # only the runs in traces/


class Ask(Server):
    def test_the_run_streams_its_steps_then_the_answer(self):
        status, kind, body = self.ask("savol")
        self.assertEqual((status, kind), (200, "application/x-ndjson; charset=utf-8"))
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual([e["type"] for e in events], ["step", "step", "answer"])
        self.assertEqual(events[0]["step"]["calls"][0]["name"], "fetch_article")
        self.assertEqual(events[-1]["answer"], {"answer": "40 soat [182-modda]", "citations": ["182"]})

    def test_a_question_must_be_json_with_text(self):
        self.assertEqual(self.ask("savol", content_type="text/plain")[0], 415)     # a cross-site form can't send it
        self.assertEqual(self.ask("  ")[0], 400)
        self.assertEqual(self.request("POST", "/api/ask", "not json", {"Content-Type": "application/json"})[0], 400)

    def test_only_requests_addressed_to_this_server(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example"})[0], 403)   # DNS rebinding
        self.assertEqual(self.request("POST", "/api/ask", "{}", {"Host": "evil.example",
                                                                 "Content-Type": "application/json"})[0], 403)


class NoKey(Server):
    @staticmethod
    def model():
        raise SystemExit("no GEMINI_API_KEY")

    def test_a_missing_key_is_shown_not_a_crash(self):
        status, _, body = self.ask("savol")
        self.assertEqual(json.loads(body), {"type": "error", "error": "no GEMINI_API_KEY"})


if __name__ == "__main__":
    unittest.main()
