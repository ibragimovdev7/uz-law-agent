"""The model behind one method, complete(system, messages, stop) -> the text it writes.

    Gemini()           the live model, Google's Gemini API; needs GEMINI_API_KEY in the environment or in .env
    Recorder(model)    a live model that keeps every reply, so the run can be saved to traces/
    Replay(calls)      plays saved replies back in order, offline; fails if a prompt is not the one recorded

Only the standard library: the request is plain JSON over HTTPS, so what is sent to the model is visible here.
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
MODEL = "gemini-3.5-flash-lite"      # free tier; it keeps to Thought / Action, gemini-3.8-flash often drops the Thought
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY = (429, 500, 503)              # the free tier allows only a few requests a minute


def load_env(path=ROOT / ".env"):
    """KEY=value lines from .env into the environment; a variable already set wins."""
    for line in path.read_text().splitlines() if path.exists() else []:
        k, sep, v = line.partition("=")
        if sep and not k.strip().startswith("#") and v.strip():
            os.environ.setdefault(k.strip(), v.strip())


def request_body(system, messages, stop=None, replies=None):
    """The JSON generateContent expects. Our "assistant" is Gemini's "model". A model turn goes back as Gemini wrote
    it (`replies`: its text -> its content), because the parts carry thought signatures Gemini 3 wants returned."""
    replies = replies or {}

    def content(m):
        if m["role"] == "assistant":
            return replies.get(m["content"]) or {"role": "model", "parts": [{"text": m["content"]}]}
        return {"role": "user", "parts": [{"text": m["content"]}]}

    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [content(m) for m in messages]}
    if stop:
        body["generationConfig"] = {"stopSequences": stop}
    return body


def reply(response):
    """(the text of the first candidate without thought parts, its content as sent); an error if none came back."""
    if not response.get("candidates"):
        raise RuntimeError(f"Gemini gave no answer: {response.get('promptFeedback', response)}")
    content = {"role": "model", "parts": [], **response["candidates"][0].get("content", {})}   # parts may be missing
    return "".join(p.get("text", "") for p in content.get("parts", []) if not p.get("thought")), content


class Gemini:
    def __init__(self, model=None):
        load_env()
        self.model = model or os.environ.get("GEMINI_MODEL") or MODEL
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise SystemExit("no GEMINI_API_KEY: get a free key at https://aistudio.google.com/apikey and put it in "
                             ".env (see .env.example), or replay a recorded run: python -m agent --replay NAME")
        self.replies = {}

    def complete(self, system, messages, stop=None):
        data = json.dumps(request_body(system, messages, stop, self.replies)).encode()
        headers = {"x-goog-api-key": self.key, "Content-Type": "application/json"}
        for attempt in range(5):
            try:
                req = urllib.request.Request(URL.format(model=self.model), data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as r:
                    text, content = reply(json.load(r))
                break
            except urllib.error.HTTPError as e:
                if e.code not in RETRY or attempt == 4:
                    raise RuntimeError(f"Gemini {e.code}: {e.read().decode(errors='replace')[:800]}") from None
                time.sleep(5 * 2 ** attempt)                    # 5, 10, 20, 40 seconds
        self.replies[text.strip()] = content                    # the loop sends the reply back stripped
        return text


def fingerprint(system, messages, stop):
    return hashlib.sha256(json.dumps([system, messages, stop], ensure_ascii=False).encode()).hexdigest()[:16]


class Recorder:
    def __init__(self, model):
        self.model, self.calls = model, []

    def complete(self, system, messages, stop=None):
        text = self.model.complete(system, messages, stop)
        self.calls.append({"prompt": fingerprint(system, messages, stop), "reply": text})
        return text


class Replay:
    def __init__(self, calls, name="replay"):
        self.calls, self.name, self.n = calls, name, 0

    def complete(self, system, messages, stop=None):
        if self.n >= len(self.calls) or self.calls[self.n]["prompt"] != fingerprint(system, messages, stop):
            raise RuntimeError(f"{self.name}: call {self.n + 1} differs from the recording; record it again")
        self.n += 1
        return self.calls[self.n - 1]["reply"]
