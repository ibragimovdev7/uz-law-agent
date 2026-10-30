"""Tests for the web server, with a scripted model: no key, no network beyond this computer.

    python -m unittest discover -s tests -v
"""
import base64
import http.client
import json
import threading
import unittest

from agent.corpus import LABOR_CODE, Law
from agent.web import MAX_QUESTION, Limits, make_server
from test_loop import ANSWER_182, FETCH_182, Script

LAW = Law(LABOR_CODE)


class Server(unittest.TestCase):
    model = staticmethod(lambda: Script(FETCH_182, ANSWER_182))
    public, password, limits = False, None, None

    def setUp(self):
        self.server = make_server(0, law=LAW, model=lambda: self.model(), public=self.public, password=self.password,
                                  limits=self.limits() if self.limits else None)
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
        runs = {r["name"]: r for r in json.loads(body)}
        self.assertEqual(runs["q1_week"]["question"], "Normal ish haftasi necha soat?")
        self.assertEqual((runs["q2_under18"]["steps"], runs["q2_under18"]["citations"]), (3, ["182", "415"]))
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


class Public(Server):
    """--public: reached by any host name (a tunnel), but only with the password."""
    public, password = True, "correct horse battery"

    def auth(self, password):
        return {"Host": "abc123.ngrok-free.app",
                "Authorization": "Basic " + base64.b64encode(f"anyone:{password}".encode()).decode()}

    def test_without_the_password_the_browser_is_asked_for_it(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/", headers={"Host": "abc123.ngrok-free.app"})
        r = conn.getresponse()
        self.assertEqual((r.status, r.getheader("WWW-Authenticate")), (401, 'Basic realm="uz-law-agent", charset="UTF-8"'))
        conn.close()
        self.assertEqual(self.request("GET", "/", headers=self.auth("wrong"))[0], 401)
        self.assertEqual(self.request("GET", "/", headers={"Authorization": "Basic %%%"})[0], 401)

    def test_with_the_password_any_host_name_works(self):
        self.assertEqual(self.request("GET", "/api/law", headers=self.auth(self.password))[0], 200)
        status, _, body = self.request("POST", "/api/ask", json.dumps({"question": "savol"}),
                                       {**self.auth(self.password), "Content-Type": "application/json"})
        self.assertEqual(json.loads(body.splitlines()[-1])["answer"]["citations"], ["182"])


class Open(Server):
    """--public --no-password: anyone may ask, within the limits."""
    public, limits = True, staticmethod(lambda: Limits(per_hour=2, per_day=3))

    def ask_as(self, visitor, question="savol"):
        return self.request("POST", "/api/ask", json.dumps({"question": question}),
                            {"Host": "abc123.ngrok-free.app", "Content-Type": "application/json",
                             "X-Forwarded-For": f"10.0.0.9, {visitor}"})          # the tunnel adds the last entry

    def test_anyone_may_ask_until_the_hourly_limit(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "abc123.ngrok-free.app"})[0], 200)
        self.assertEqual([self.ask_as("1.1.1.1")[0] for _ in range(3)], [200, 200, 429])
        self.assertIn("2 tadan ortiq", json.loads(self.ask_as("1.1.1.1")[2])["error"])
        self.assertEqual(self.ask_as("2.2.2.2")[0], 200)                       # another visitor, the 3rd today
        self.assertIn("Bugungi", json.loads(self.ask_as("3.3.3.3")[2])["error"])   # the day's limit

    def test_a_long_question_is_refused(self):
        self.assertEqual(self.ask_as("1.1.1.1", "x" * (MAX_QUESTION + 1))[0], 400)


class LimitsCount(unittest.TestCase):
    def test_an_hour_later_a_visitor_may_ask_again(self):
        limits = Limits(per_hour=1, per_day=10)
        self.assertIsNone(limits.admit("a", now=1000))
        self.assertIsNotNone(limits.admit("a", now=1000 + 3599))
        self.assertIsNone(limits.admit("a", now=1000 + 3600))

    def test_the_day_count_starts_again_at_midnight_utc(self):
        limits = Limits(per_hour=10, per_day=1)
        self.assertIsNone(limits.admit("a", now=86400 - 10))
        self.assertIsNotNone(limits.admit("b", now=86400 - 5))
        self.assertIsNone(limits.admit("b", now=86400 + 5))

    def test_only_so_many_at_once(self):
        limits = Limits(at_once=1)
        self.assertTrue(limits.slots.acquire(blocking=False))
        self.assertFalse(limits.slots.acquire(blocking=False))


class NoKey(Server):
    @staticmethod
    def model():
        raise SystemExit("no GEMINI_API_KEY")

    def test_a_missing_key_is_shown_not_a_crash(self):
        status, _, body = self.ask("savol")
        self.assertEqual(json.loads(body), {"type": "error", "error": "no GEMINI_API_KEY"})


if __name__ == "__main__":
    unittest.main()
