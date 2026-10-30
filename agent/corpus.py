"""A law as the agent sees it: articles loaded from a JSONL file, and the two tools the agent may call on it.

    law = Law(LABOR_CODE)
    law.search("ish vaqtining davomiyligi")   # [("182", "Ish vaqtining normal davomiyligi", 6), ...]
    law.fetch_article("182")                  # "182-modda. Ish vaqtining normal davomiyligi\nXodim uchun ..."

`search` is deliberately naive: it counts the word stems an article shares with the query. A better retriever can
replace it later; the interface (`search`, `fetch_article`) stays the same.
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
LABOR_CODE = DATA / "mehnat_kodeksi.jsonl"

# A keyboard gives ' or ‘ ’ where the corpus has ʻ (oʻ, gʻ) or ʼ (tutuq belgisi). Both sides are mapped to one
# character before matching, or "o'n sakkiz" would never find "oʻn sakkiz".
APOSTROPHES = str.maketrans({c: "ʻ" for c in "ʼ'‘’`"})

# What a model may write for one article: "182", "182-modda", "182- modda", "Art. 182"; inserted articles are "182.1"
ARTICLE_REF = re.compile(r"(?:art\.?\s*)?(\d+(?:\.\d+)?)(?:\s*-\s*modda)?", re.I)


def words(text):
    return re.findall(r"\w+", text.lower().translate(APOSTROPHES))


def stems(text):
    """Crude stems: Uzbek adds suffixes (davomiylik, davomiyligi), so compare the first five letters."""
    return {w[:5] for w in words(text)}


class Law:
    def __init__(self, path):
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.articles = {r["article_id"]: r for r in records}
        if len(self.articles) != len(records):
            raise ValueError(f"{path}: an article_id is used twice")
        self.title = records[0].get("code_title", Path(path).stem)  # "Oʻzbekiston Respublikasining Mehnat kodeksi"
        self.stems = {k: stems(a["title"] + " " + a["text"]) for k, a in self.articles.items()}
        self.title_stems = {k: stems(a["title"]) for k, a in self.articles.items()}

    def search(self, query, k=5):
        """The k articles sharing the most stems with the query, a stem in the title counting twice, best first:
        [(article id, title, score)]. Ties keep the order of the code."""
        q = stems(query)
        scored = [(len(q & s) + len(q & self.title_stems[a]), a) for a, s in self.stems.items() if q & s]
        scored.sort(key=lambda x: -x[0])         # sort is stable, so equal scores stay in file order
        return [(a, self.articles[a]["title"], n) for n, a in scored[:k]]

    def article_id(self, ref):
        """The id of the article `ref` names ("182-modda" -> "182"), or None if there is no such article."""
        m = ARTICLE_REF.fullmatch(str(ref).strip())
        return m.group(1) if m and m.group(1) in self.articles else None

    def fetch_article(self, ref):
        """The full text of one article, or None if there is no such article."""
        a = self.articles.get(self.article_id(ref))
        return f"{a['article_display']}. {a['title']}\n{a['text']}" if a else None
