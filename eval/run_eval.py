"""Run the golden set against the running app and report pass rate and quote checks.

Usage (from the repo root):  python eval/run_eval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def main() -> int:
    client = TestClient(create_app(Settings(_env_file=None)))
    sample_texts: dict[str, str] = {}
    for path in sorted((ROOT / "samples").glob("*.txt")):
        content = path.read_bytes()
        client.post("/documents", files={"file": (path.name, content, "text/plain")}).raise_for_status()
        sample_texts[path.name] = content.decode("utf-8")

    rows = [
        json.loads(line)
        for line in (ROOT / "eval" / "golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    for row in rows:
        body = client.post("/query", json={"question": row["question"]}).json()
        ok = body["outcome"] == row["expect_outcome"]
        if row.get("expect_document") and body.get("quotes"):
            ok = ok and any(q["document_name"] == row["expect_document"] for q in body["quotes"])
        for quote in body.get("quotes", []):
            ok = ok and quote["text"] in sample_texts.get(quote["document_name"], "")
        passed += ok
        print(("PASS" if ok else "FAIL"), body["outcome"], "|", row["question"])

    print(f"{passed}/{len(rows)} passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
