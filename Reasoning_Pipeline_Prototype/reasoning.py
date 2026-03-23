import json
import subprocess
from pathlib import Path

JSON_PATH = Path("eeg_findings.json")
# MODEL_NAME = "deepseek-r1:1.5b"
# MODEL_NAME = "deepseek-r1:8b"
MODEL_NAME = "qwen3:8b"


def load_findings(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(findings: dict) -> str:
    return f"""
You are assisting with EEG interpretation in a clinical research setting.

Rules:
- Do NOT make a diagnosis
- Do NOT overclaim
- Be cautious and evidence-based
- Use phrases like "may suggest", "could be associated with"
- Mention uncertainty and limitations

Structured EEG findings:
{json.dumps(findings, indent=2)}

Write the output in the following format:

1. Summary of findings
2. Possible interpretation
3. Limitations and uncertainty

Keep it concise, clear, and professional.
""".strip()

def run_ollama(prompt: str, model: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ollama error:\n{result.stderr}")

    return result.stdout.strip()

def main():
    findings = load_findings(JSON_PATH)
    prompt = build_prompt(findings)

    response = run_ollama(prompt, MODEL_NAME)

    print("\n=== Clinician-style Explanation ===\n")
    print(response)

if __name__ == "__main__":
    main()