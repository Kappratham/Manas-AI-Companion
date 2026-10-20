# audit_output.py
import json
import re
from pathlib import Path

CONTAMINATION_SIGNALS = [
    # Self-commentary
    "this last paragraph",
    "this sequence flows",
    "the ai creates",
    "by framing it as",
    "hence this sequence",
    "thus patience reigns",
    "soaring solutions",
    "gradualism honors",
    # Session management
    "this concludes session",
    "let us check back here",
    "see you soon if you decide",
    "goodnight. this",
    "until then. goodnight",
    "here we go again tomorrow",
    # Medication dumps
    "zoloft", "lexapro", "prozac", "xanax",
    "sertraline", "fluoxetine", "quetiapine",
    # Acronym dumps
    "emdr", "gdpr compliant", "live-stream style",
    "participant input dynamics",
    "sliding scale payment",
    # UserSimulator loop signal
    "i don't know how to explain it, i just feel really stuck.",
]

output_dir = Path("output")
total_contaminated = 0
total_conversations = 0

for jsonl_file in sorted(output_dir.glob("*.jsonl")):
    contaminated = 0
    clean = 0

    for line in jsonl_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total_conversations += 1
        try:
            conv = json.loads(line)
            is_contaminated = False
            for turn in conv.get("conversations", []):
                if turn.get("role") != "assistant":
                    continue
                content = turn.get("content", "").lower()
                for signal in CONTAMINATION_SIGNALS:
                    if signal.lower() in content:
                        is_contaminated = True
                        break
                if is_contaminated:
                    break
            if is_contaminated:
                contaminated += 1
                total_contaminated += 1
            else:
                clean += 1
        except Exception as e:
            contaminated += 1

    status = "🔴 CONTAMINATED" if contaminated > 0 else "✅ clean"
    print(
        f"{jsonl_file.stem:30s} : {status} "
        f"({contaminated} bad / {clean} clean)"
    )

print(f"\nTotal: {total_contaminated}/{total_conversations} contaminated")