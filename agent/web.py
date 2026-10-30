"""A web page for the agent, served on this computer only.

    python3 -m agent.web               # then open http://127.0.0.1:8765
    python3 -m agent.web --port 8080

POST /api/ask {"question"} runs the agent and streams the run as JSON lines, one per step and then the answer, so
the page shows each step as it happens. GET /api/runs lists the recorded runs and GET /api/runs/NAME is one of them
(offline, no key); GET /api/article/182 is an article; GET /api/law names the law. Only the standard library.

The key stays on this computer: the server listens on 127.0.0.1 only, answers only requests addressed to it by that
name (against DNS rebinding), and takes questions only as application/json, which a page from another site cannot
send here without the browser asking first.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from agent.corpus import LABOR_CODE, Law
from agent.llm import TRACES, Gemini
from agent.loop import run

PAGE = Path(__file__).with_name("web.html")


def make_server(port=8765, law=None, model=Gemini, traces=TRACES, max_steps=8):
    """The server on 127.0.0.1 (port 0 picks a free one). `model()` makes the model for each question."""
    law = law or Law(LABOR_CODE)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def addressed_to_us(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.addressed_to_us():
                return self.send_json({"error": "forbidden"}, 403)
            path = unquote(urlparse(self.path).path)
            runs = {p.stem: p for p in sorted(traces.glob("*.json"))}
            if path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/law":
                self.send_json({"title": law.title, "articles": len(law.articles)})
            elif path == "/api/runs":
                self.send_json([{"name": n, "question": json.loads(p.read_text(encoding="utf-8"))["question"]}
                                for n, p in runs.items()])
            elif path.startswith("/api/runs/") and path[len("/api/runs/"):] in runs:
                t = json.loads(runs[path[len("/api/runs/"):]].read_text(encoding="utf-8"))
                self.send_json({k: t[k] for k in ("model", "question", "answer", "trace")})
            elif path.startswith("/api/article/") and law.article_id(path[len("/api/article/"):]):
                a = law.articles[law.article_id(path[len("/api/article/"):])]
                self.send_json({k: a[k] for k in ("article_id", "article_display", "title", "chapter", "text")})
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            if not self.addressed_to_us():
                return self.send_json({"error": "forbidden"}, 403)
            if urlparse(self.path).path != "/api/ask":
                return self.send_json({"error": "not found"}, 404)
            if self.headers.get_content_type() != "application/json":
                return self.send_json({"error": "send the question as application/json"}, 415)
            try:
                question = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))).get("question")
            except (ValueError, AttributeError):
                question = None
            if not isinstance(question, str) or not question.strip():
                return self.send_json({"error": 'send {"question": "..."}'}, 400)

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(kind, **data):
                self.wfile.write((json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

            try:
                answer, _ = run(question.strip(), law, model(), max_steps=max_steps,
                                on_step=lambda step: send("step", step=step))
                send("answer", answer=answer)
            except (RuntimeError, SystemExit) as e:        # an API error, or no key
                send("error", error=str(e))
            except (BrokenPipeError, ConnectionResetError):  # the page was closed: stop asking the model
                pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m agent.web")
    ap.add_argument("--port", type=int, default=8765)
    server = make_server(ap.parse_args(argv).port)
    print(f"http://127.0.0.1:{server.server_address[1]}   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
