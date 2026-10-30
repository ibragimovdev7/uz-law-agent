"""The model behind one method, complete(system, messages, tools) -> {"text": str, "calls": [{"id", "name", "args"}]}.

    Gemini()           the live model, Google's Gemini API; needs GEMINI_API_KEY in the environment or in .env
    Recorder(model)    a live model that keeps every reply, so the run can be saved to traces/
    Replay(calls)      plays saved replies back in order, offline; fails if a prompt is not the one recorded

The messages are provider-neutral: {"role": "user", "content"}, {"role": "assistant", "text", "calls"} and
{"role": "tool", "results": [{"id", "name", "content"}]}. A tool is {"name", "description", "parameters"}, the
parameters a JSON Schema. Only the standard library: the request is plain JSON over HTTPS, so what is sent to the
model is visible here.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRACES = ROOT / "traces"
MODEL = "gemini-3.5-flash-lite"      # free tier; it keeps to the format, gemini-3.8-flash did not in text mode
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY = (429, 500, 503)              # the free tier allows only a few requests a minute


def load_env(path=ROOT / ".env"):
    """KEY=value lines from .env into the environment; a variable already set wins."""
    for line in path.read_text().splitlines() if path.exists() else []:
        k, sep, v = line.partition("=")
        if sep and not k.strip().startswith("#") and v.strip():
            os.environ.setdefault(k.strip(), v.strip())


def turn_key(m):
    """What identifies a model turn, so the content Gemini wrote for it can be found again."""
    return json.dumps({"text": m["text"], "calls": m["calls"]}, ensure_ascii=False, sort_keys=True)


def request_body(system, messages, tools, sent=None):
    """The JSON generateContent expects. Our "assistant" is Gemini's "model", and tool results go back as "user"
    functionResponse parts. A model turn goes back exactly as Gemini wrote it (`sent`: turn_key -> its content):
    Gemini 3 refuses a functionCall part without its thoughtSignature (HTTP 400)."""
    sent = sent or {}

    def content(m):
        if m["role"] == "assistant":
            return sent.get(turn_key(m)) or {"role": "model", "parts": ([{"text": m["text"]}] if m["text"] else []) + [
                {"functionCall": {"name": c["name"], "args": c["args"]} | ({"id": c["id"]} if c["id"] else {})}
                for c in m["calls"]]}
        if m["role"] == "tool":
            return {"role": "user", "parts": [
                {"functionResponse": {"name": r["name"], "response": {"result": r["content"]}} |
                                     ({"id": r["id"]} if r["id"] else {})} for r in m["results"]]}
        return {"role": "user", "parts": [{"text": m["content"]}]}

    return {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [content(m) for m in messages],
            "tools": [{"functionDeclarations": [{"name": t["name"], "description": t["description"],
                                                 "parametersJsonSchema": t["parameters"]} for t in tools]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}}      # every turn is a tool call


def reply(response):
    """(our reply: the text without thought parts and the function calls; Gemini's content, to send back as is)."""
    if not response.get("candidates"):
        raise RuntimeError(f"Gemini gave no answer: {response.get('promptFeedback', response)}")
    content = {"role": "model", "parts": [], **response["candidates"][0].get("content", {})}   # parts may be missing
    parts = content["parts"]
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    calls = [{"id": p["functionCall"].get("id"), "name": p["functionCall"]["name"],
              "args": p["functionCall"].get("args", {})} for p in parts if "functionCall" in p]
    return {"text": text, "calls": calls}, content


class Gemini:
    def __init__(self, model=None):
        load_env()
        self.model = model or os.environ.get("GEMINI_MODEL") or MODEL
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise SystemExit("no GEMINI_API_KEY: get a free key at https://aistudio.google.com/apikey and put it in "
                             ".env (see .env.example), or replay a recorded run: python -m agent --replay NAME")
        self.sent = {}

    def complete(self, system, messages, tools):
        data = json.dumps(request_body(system, messages, tools, self.sent)).encode()
        headers = {"x-goog-api-key": self.key, "Content-Type": "application/json"}
        for attempt in range(5):
            try:
                req = urllib.request.Request(URL.format(model=self.model), data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as r:
                    answer, content = reply(json.load(r))
                break
            except urllib.error.HTTPError as e:
                if e.code not in RETRY or attempt == 4:
                    raise RuntimeError(f"Gemini {e.code}: {e.read().decode(errors='replace')[:800]}") from None
                time.sleep(5 * 2 ** attempt)                    # 5, 10, 20, 40 seconds
        self.sent[turn_key(answer)] = content
        return answer


def fingerprint(system, messages, tools):
    return hashlib.sha256(json.dumps([system, messages, tools], ensure_ascii=False).encode()).hexdigest()[:16]


class Recorder:
    def __init__(self, model):
        self.model, self.calls = model, []

    def complete(self, system, messages, tools):
        answer = self.model.complete(system, messages, tools)
        self.calls.append({"prompt": fingerprint(system, messages, tools), "reply": answer})
        return answer


class Replay:
    def __init__(self, calls, name="replay"):
        self.calls, self.name, self.n = calls, name, 0

    def complete(self, system, messages, tools):
        if self.n >= len(self.calls) or self.calls[self.n]["prompt"] != fingerprint(system, messages, tools):
            raise RuntimeError(f"{self.name}: call {self.n + 1} differs from the recording; record it again")
        self.n += 1
        return self.calls[self.n - 1]["reply"]
