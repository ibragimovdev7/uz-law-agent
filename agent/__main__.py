"""Ask the agent a question about the Labor Code.

    python -m agent "Normal ish haftasi necha soat?"                      # live, needs GEMINI_API_KEY
    python -m agent --record q1_week "Normal ish haftasi necha soat?"     # live, and save traces/q1_week.json
    python -m agent --replay q1_week                                      # the saved run, offline, no key
"""
import argparse
import json

from agent.corpus import LABOR_CODE, Law
from agent.llm import TRACES, Gemini, Recorder, Replay
from agent.loop import run


def show(step):
    print(f"\n── step {step['step']}" + (f"\n{step['text']}" if step["text"] else ""))
    for call, result in zip(step["calls"], step["results"]):
        content = result["content"] if len(result["content"]) < 400 else result["content"][:400] + " …"
        print(f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)})\n→ {content}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m agent")
    ap.add_argument("question", nargs="?")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--record", metavar="NAME", help="run live and save the run to traces/NAME.json")
    mode.add_argument("--replay", metavar="NAME", help="replay traces/NAME.json offline")
    ap.add_argument("--max-steps", type=int, default=8, help="the step limit (default 8)")
    a = ap.parse_args(argv)

    if a.replay:
        saved = json.loads((TRACES / f"{a.replay}.json").read_text(encoding="utf-8"))
        question, max_steps = saved["question"], saved["max_steps"]
        model = Replay(saved["calls"], a.replay)
    elif a.question:
        question, max_steps = a.question, a.max_steps
        live = Gemini()
        model = Recorder(live) if a.record else live
    else:
        ap.error("give a question, or --replay NAME")

    print(f"Q: {question}", flush=True)
    try:
        answer, trace = run(question, Law(LABOR_CODE), model, max_steps=max_steps, on_step=show)
    except RuntimeError as e:                 # an API error, or a replay that no longer matches
        raise SystemExit(f"error: {e}") from None
    if answer:
        print(f"\nANSWER: {answer['answer']}\nCITATIONS: {', '.join(answer['citations']) or '-'}")
    else:
        print(f"\n── stopped: {max_steps} steps without an answer\nANSWER: none")
    if a.record:
        TRACES.mkdir(exist_ok=True)
        saved = {"model": live.model, "question": question, "max_steps": max_steps, "answer": answer,
                 "trace": trace, "calls": model.calls}
        (TRACES / f"{a.record}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")
        print(f"saved traces/{a.record}.json")


if __name__ == "__main__":
    main()
