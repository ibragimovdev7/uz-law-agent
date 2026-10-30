"""The agent: a loop over one law in which the model calls tools declared with JSON schemas.

Each turn the model returns structured tool calls, so there is no text to parse. Our code checks every call against
its tool's schema, runs it and sends the result back; a malformed call gets an error as its result, never a crash.
The loop ends when the model calls answer with citations that are articles it has read, or at the step limit.

The model is any object with complete(system, messages, tools) -> {"text": str, "calls": [{"id", "name", "args"}]},
so the loop does not depend on a provider.
"""


def params(**properties):
    """The schema of a tool's arguments: an object whose properties are all required, and nothing else."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TOOLS = [
    {"name": "search",
     "description": "Keyword search over the articles: up to 5 article numbers with their titles, best first. The "
                    "code is written in Uzbek (Latin script), so search with Uzbek words.",
     "parameters": params(query={"type": "string", "description": "Uzbek words, e.g. ish vaqtining davomiyligi"})},
    {"name": "fetch_article",
     "description": "The full text of one article.",
     "parameters": params(article={"type": "string", "description": "the article number, e.g. 182"})},
    {"name": "answer",
     "description": "The final answer. Call it when you have read the articles the answer rests on.",
     "parameters": params(
         answer={"type": "string", "description": "the answer in the language of the question; cite every claim "
                                                  "in the text as [182-modda]"},
         citations={"type": "array", "items": {"type": "string"},
                    "description": "the numbers of the articles the answer cites, each one read with fetch_article"})},
]
SCHEMAS = {t["name"]: t["parameters"] for t in TOOLS}

SYSTEM = """You answer questions about {title} ({n} articles). You do not know its text; you can only read it with \
the tools: search it, read the articles that bear on the question with fetch_article, then call answer. If the code \
does not answer the question, say so in the answer."""

NO_CALL = "Call one of the tools: search, fetch_article or answer."


def system_prompt(law):
    return SYSTEM.format(title=law.title, n=len(law.articles))


def check(args, schema):
    """What is wrong with the arguments under the schema, or None. It covers what the schemas above use: an object
    of strings and lists of strings, all required, nothing else."""
    if not isinstance(args, dict):
        return "the arguments must be an object"
    missing = [k for k in schema["required"] if k not in args]
    unknown = [k for k in args if k not in schema["properties"]]
    if missing or unknown:
        return "; ".join(f"{what} argument {', '.join(ks)}" for what, ks in [("missing", missing), ("unknown", unknown)]
                         if ks)
    for k, v in args.items():
        if schema["properties"][k]["type"] == "string" and not isinstance(v, str):
            return f"{k} must be a string"
        if schema["properties"][k]["type"] == "array" and not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            return f"{k} must be a list of strings"
    return None


def act(law, call, read):
    """Run one tool call: (the result the model sees, the final answer if this call ends the run).
    `read` is the set of articles fetched so far; a fetch adds to it."""
    if call["name"] not in SCHEMAS:
        return f"error: there is no tool {call['name']}; the tools are search, fetch_article, answer", None
    problem = check(call["args"], SCHEMAS[call["name"]])
    if problem:
        return f"error: {problem}", None
    args = call["args"]
    if call["name"] == "search":
        hits = law.search(args["query"])
        return "\n".join(f"{law.articles[a]['article_display']}: {title}" for a, title, _ in hits) or \
            "no article contains these words; try other Uzbek words", None
    if call["name"] == "fetch_article":
        a = law.article_id(args["article"])
        if a is None:
            return f"error: there is no article {args['article']}", None
        read.add(a)
        return law.fetch_article(a), None
    ids = [law.article_id(c) for c in args["citations"]]
    unknown = [c for c, a in zip(args["citations"], ids) if a is None]
    if unknown:
        return f"error: there is no article {', '.join(unknown)}; cite article numbers such as 182", None
    unread = [a for a in ids if a not in read]
    if unread:
        return f"error: cite only articles you have read: fetch {', '.join(unread)} first, or leave it out", None
    return "accepted", {"answer": args["answer"], "citations": list(dict.fromkeys(ids))}


def run(question, law, model, max_steps=8, on_step=lambda step: None):
    """The loop. Returns (the answer {"answer", "citations"}, or None if the steps ran out; the trace, one dict per
    step: what the model wrote, its calls, and their results). on_step gets each step as it ends, to show it."""
    system, messages, trace, read = system_prompt(law), [{"role": "user", "content": question}], [], set()
    for step in range(1, max_steps + 1):
        reply = model.complete(system, messages, TOOLS)
        messages.append({"role": "assistant", **reply})
        results, final = [], None
        for call in reply["calls"]:
            result, done = act(law, call, read)
            results.append({"id": call["id"], "name": call["name"], "content": result})
            final = final or done
        trace.append({"step": step, **reply, "results": results})
        on_step(trace[-1])
        if final:
            return final, trace
        messages.append({"role": "tool", "results": results} if results else {"role": "user", "content": NO_CALL})
    return None, trace
