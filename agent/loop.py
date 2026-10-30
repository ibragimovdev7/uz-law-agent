"""The agent: a ReAct loop over one law.

The model writes a Thought and one Action as plain text; our code reads the Action, runs the tool, and sends the result
back as an Observation. The model reads the whole conversation again and writes the next Thought. The loop ends when
the model calls answer[...] or when the step limit is reached.

The model is any object with complete(system, messages, stop) -> text, so the loop does not depend on a provider.
"""
import re

ACTION = re.compile(r"^Action:\s*(\w+)\[(.*)\]\s*$", re.M | re.S)
STOP = ["Observation:"]          # the model is cut off here, so only our code can write an observation

SYSTEM = """You answer questions about {title} ({n} articles). You do not know its text; you can only read it with \
these actions:

  search[words]          keyword search over the articles. The code is written in Uzbek (Latin script), so search \
with Uzbek words, for example: search[ish vaqtining davomiyligi]
  fetch_article[number]  the full text of one article, for example: fetch_article[182]
  answer[text]           your final answer, in the language of the question. Cite every claim as [182-modda]. If the \
code does not answer the question, say so.

Work in turns. In each turn write exactly two lines:
Thought: <one sentence: what you know so far, and why the next action>
Action: <one action>
The Thought line is required: it is shown to the reader as the trace of your reasoning. Then stop. The result comes back to you as "Observation: ...". Never write an Observation yourself."""

CANNOT_READ = "I could not read an action. Write one line: Action: search[...], fetch_article[...] or answer[...]"


def system_prompt(law):
    return SYSTEM.format(title=law.title, n=len(law.articles))


def act(law, name, arg):
    """Run one action; the returned text is what the model sees as the Observation."""
    if name == "search":
        hits = law.search(arg)
        return "\n".join(f"{law.articles[a]['article_display']}: {title}" for a, title, _ in hits) or \
            "no article contains these words; try other Uzbek words"
    if name == "fetch_article":
        return law.fetch_article(arg) or f"there is no article {arg}"
    return f"unknown action {name}; the actions are search, fetch_article, answer"


def run(question, law, model, max_steps=8, stop=True, show=lambda text: None):
    """The loop. Returns (the answer, or None if the steps ran out; the trace, one dict per step)."""
    system, messages, trace = system_prompt(law), [{"role": "user", "content": question}], []
    for step in range(1, max_steps + 1):
        text = model.complete(system, messages, stop=STOP if stop else None).strip()
        messages.append({"role": "assistant", "content": text})
        m = ACTION.search(text)
        if m and m.group(1) == "answer":
            trace.append({"step": step, "model": text, "observation": None})
            show(f"\n── step {step}\n{text}")
            return m.group(2).strip(), trace
        obs = act(law, m.group(1), m.group(2).strip()) if m else CANNOT_READ
        trace.append({"step": step, "model": text, "observation": obs})
        show(f"\n── step {step}\n{text}\nObservation: {obs if len(obs) < 400 else obs[:400] + ' …'}")
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    show(f"\n── stopped: {max_steps} steps without an answer")
    return None, trace
