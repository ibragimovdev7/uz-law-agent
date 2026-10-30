"""No API key may be committed: scans every tracked file, and checks that .env is ignored."""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Google API key (AQ.)": re.compile(r"\bAQ\.[0-9A-Za-z_\-]{30,}"),
    "GEMINI_API_KEY with a value": re.compile(r"GEMINI_API_KEY[ \t]*=[ \t]*['\"]?[0-9A-Za-z_.\-]{20,}"),
    "WEB_PASSWORD with a value": re.compile(r"WEB_PASSWORD[ \t]*=[ \t]*['\"]?\S{12,}"),
}


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [ROOT / p for p in out.decode().split("\0") if p]


def findings(name, text):
    return [f"{name}: {label}" for label, rx in PATTERNS.items() if rx.search(text)]


class NoSecrets(unittest.TestCase):
    def test_no_tracked_file_contains_a_key(self):
        found = []
        for path in tracked_files():
            if path.name != Path(__file__).name and path.is_file():     # this file spells the patterns out
                found += findings(path.relative_to(ROOT), path.read_text(encoding="utf-8", errors="ignore"))
        self.assertEqual(found, [])

    def test_the_scanner_catches_a_key_and_accepts_the_example(self):
        self.assertTrue(findings("x", "GEMINI_API_KEY=" + "a1" * 15))
        self.assertTrue(findings("x", "key = 'AIza" + "B" * 35 + "'"))
        self.assertTrue(findings("x", "key = 'AQ." + "Ab-9" * 10 + "'"))
        self.assertTrue(findings("x", "GEMINI_API_KEY=AQ." + "Ab-9" * 10))
        self.assertTrue(findings("x", "WEB_PASSWORD=" + "x7_K" * 4))
        self.assertEqual(findings(".env.example", (ROOT / ".env.example").read_text()), [])

    def test_env_is_ignored(self):
        ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode == 0
        self.assertTrue(ignored)
        self.assertFalse([p for p in tracked_files() if p.name.startswith(".env") and p.name != ".env.example"])


if __name__ == "__main__":
    unittest.main()
