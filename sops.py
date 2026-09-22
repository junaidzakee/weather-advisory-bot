"""
Loads Standard Operating Procedures (SOPs) from sops.yaml.

Deliberately NOT cached at import time. We re-read the file on every call so
that editing sops.yaml (adding, changing, or removing a rule) takes effect on
the very next chat message -- no server restart, no code touched. The file is
tiny, so re-reading it every request costs nothing worth optimizing for.
"""
import yaml
from pathlib import Path

SOPS_PATH = Path(__file__).parent / "sops.yaml"


def load_sops() -> list[dict]:
    with open(SOPS_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data["sops"]


def format_sops_for_prompt(sops: list[dict]) -> str:
    """Render the SOP list as plain text for the LLM prompt."""
    lines = []
    for s in sops:
        lines.append(
            f"- id: {s['id']} | category: {s['category']} | severity: {s['severity']}\n"
            f"  condition: {s['condition'].strip()}\n"
            f"  advice: {s['advice'].strip()}"
        )
    return "\n".join(lines)