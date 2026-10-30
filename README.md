# uz-law-agent

An open-source Uzbek law agent: it answers a question by searching Uzbek law and reading articles, and cites the
articles it used. It is developed following Module 7 (AI Agents, Tool Use & Retrieval) of the Advanced AI course.

The law included so far is the Labor Code of Uzbekistan (581 articles). The agent is a tool-calling loop: the tools
`search`, `fetch_article` and `answer` are declared with JSON schemas, the model returns structured calls, and our code
checks each call against its schema, runs it and sends back the result. The run ends when the model calls `answer`
with an answer and the articles it cites, each one an article it has read, or at the step limit. The model is Google's Gemini (free tier). Only Python's
standard library is used (tested with Python 3.12): nothing to install.

## Quick start

```bash
cp .env.example .env                  # then put your key in .env: https://aistudio.google.com/apikey
python3 -m agent "Normal ish haftasi necha soat?"
python3 -m agent --record q1_week "Normal ish haftasi necha soat?"   # also save the run to traces/q1_week.json
python3 -m agent --replay q1_week                                    # replay a saved run: no key, no network
```

The free tier allows only a few requests a minute (the client waits and retries), and Google may use what is sent on
it to improve its products.

## Tests

```bash
python3 -m unittest discover -s tests -v     # offline, no key
```

## Layout

| Path | What |
|---|---|
| `data/` | the laws (one file per law) and a source note next to each: where the text comes from, the date, the licence |
| `agent/corpus.py` | a law: its articles and the two tools, `search` and `fetch_article` |
| `agent/loop.py` | the agent loop, the tools' JSON schemas and the system prompt |
| `agent/llm.py` | the model: Gemini live, recording and replay |
| `agent/__main__.py` | the command line |
| `tests/` | tests |
| `traces/` | recorded runs |
| `docs/` | documentation |

Keys go in `.env` (git-ignored); `.env.example` is the template. Never commit a key; `tests/test_no_secrets.py` fails
if a tracked file contains one.

## Licence

Code: MIT, see [LICENSE](LICENSE). Each law's text carries the licence stated in its source note.
