import json
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

question = " ".join(sys.argv[1:]).strip()
if not question:
    raise SystemExit('Usage: python ask.py "ваш вопрос"')

payload = json.dumps(
    {
        "question": question,
        "diagnostics": True,
    },
    ensure_ascii=False,
).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8000/ask",
    data=payload,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print(json.dumps(data, ensure_ascii=False, indent=2))
