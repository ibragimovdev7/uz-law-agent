# mehnat_kodeksi.jsonl: the Labor Code of Uzbekistan

**What.** The Labor Code of the Republic of Uzbekistan (Oʻzbekiston Respublikasining Mehnat kodeksi), adopted
2022-10-28, lex.uz document `-6257288`. 581 articles, one JSON object per line, in Uzbek (Latin script). The text is
the in-force consolidated edition as of the snapshot date, 2026-07.

**Where from.** Taken unchanged from [javohirmat/uzbek-legal-corpus](https://github.com/javohirmat/uzbek-legal-corpus)
at commit `7fabb822cca04439ac6197bfa1441eca6112bc19`, file `data/articles/mehnat_kodeksi.jsonl`, retrieved 2026-10-30:

```bash
curl -fsSL -o data/mehnat_kodeksi.jsonl \
  https://raw.githubusercontent.com/javohirmat/uzbek-legal-corpus/7fabb822cca04439ac6197bfa1441eca6112bc19/data/articles/mehnat_kodeksi.jsonl
```

The corpus took its texts from the National Database of Legislation, [lex.uz](https://lex.uz), the only authoritative
source. This file is an unofficial copy, and laws change.

**Licence.** Data released under CC0 1.0 by the corpus. Legislative texts are not objects of copyright under Uzbek law
(Law "On Copyright and Related Rights", OʻRQ-42 of 20.07.2006, Art. 8).

**sha256** `cd91401cf6973f8fd9a738daca6cc986ae36d6871ec312721fc54d27d29cf887`

## Records

Every record has the same 13 fields:

| Field | Example | Meaning |
|---|---|---|
| `article_id` | `"182"` | the article number as a string; what the agent cites |
| `article_display` | `"182-modda"` | how the article is named in the text |
| `title` | `"Ish vaqtining normal davomiyligi"` | the article's heading |
| `text` | `"Xodim uchun ish vaqtining …"` | the full text; paragraphs separated by `\n` |
| `section`, `chapter` | `"13-bob. Ish vaqti"` | where the article sits in the code |
| `seq`, `article_raw`, `kind`, `code`, `code_title`, `lex_uz_doc`, `snapshot_date` | | position and provenance |

As found on 2026-10-30:

- 581 records, `article_id` `"1"` to `"581"` in order, no gaps, no repeats, all `kind: "base"` (no inserted articles
  such as 182¹);
- no empty `title` or `text`; texts are 97 to 6,045 characters (median 840), 80,270 words with the titles;
- 7 sections, 34 chapters.

## Spelling

The corpus writes oʻ and gʻ with U+02BB (ʻ, 9,555 times) and the tutuq belgisi with U+02BC (ʼ, 1,210 times). It has
no ASCII `'`, no ‘ ’ and no backtick. A user types `o'n`, so search must map these to one character on both sides,
or `o'n sakkiz` never finds `oʻn sakkiz`. Do not "fix" the file.
