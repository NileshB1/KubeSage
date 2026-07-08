"""Test script: verify JSON repair handles all 7 LLM failure modes."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.rag_pipeline import ReportParser

PASSED = 0
FAILED = 0


def check(label: str, text: str, expect_keys: list[str]) -> None:
    global PASSED, FAILED
    result = ReportParser.extract_json(text)
    if "error" in result:
        print(f"  FAIL [{label}]: {result['error']}")
        FAILED += 1
        return
    missing = [k for k in expect_keys if k not in result]
    if missing:
        print(f"  FAIL [{label}]: missing keys {missing}")
        FAILED += 1
    else:
        print(f"  PASS [{label}]: got {len(result)} keys")
        PASSED += 1


# --- Case 1: Trailing commas ------------------------------------------------
check(
    "Trailing commas",
    '{"a": 1, "b": [2, 3,],}',
    ["a", "b"],
)

# --- Case 2: Single quotes instead of double quotes -------------------------
check(
    "Single quotes",
    "{'incident_id': 'INC-1234', 'severity': 'High', 'root_cause': 'oom', "
    "'evidence': ['log line 1', 'log line 2'], 'affected_services': ['svc-a'], "
    "'confidence_score': 85, 'recommended_fixes': ['restart'], "
    "'generated_summary': 'summary text'}",
    ["incident_id", "severity", "root_cause", "evidence", "affected_services",
     "confidence_score", "recommended_fixes", "generated_summary"],
)

# --- Case 3: Missing closing braces -----------------------------------------
check(
    "Missing closing braces",
    '{"incident_id": "INC-999", "severity": "Low", "root_cause": "test", '
    '"evidence": ["e1"], "affected_services": ["svc"], "confidence_score": 50, '
    '"recommended_fixes": ["fix"], "generated_summary": "ok"',
    ["incident_id", "severity"],
)

# --- Case 4: Extra text after closing brace ----------------------------------
check(
    "Extra text after JSON",
    '{"incident_id": "INC-1", "severity": "High", "root_cause": "r", '
    '"evidence": ["e"], "affected_services": ["a"], "confidence_score": 90, '
    '"recommended_fixes": ["f"], "generated_summary": "s"}'
    '\n\nHere is some extra explanation about the incident...',
    ["incident_id", "generated_summary"],
)

# --- Case 5: Unquoted keys --------------------------------------------------
check(
    "Unquoted keys",
    '{incident_id: "INC-42", severity: "Critical", root_cause: "db pool", '
    'evidence: ["e"], affected_services: ["s"], confidence_score: 100, '
    'recommended_fixes: ["f"], generated_summary: "ok"}',
    ["incident_id", "severity", "confidence_score"],
)

# --- Case 6: Double-quoted keys, single-quoted values -----------------------
check(
    "Mixed quotes (keys OK, values single-quoted)",
    '{"incident_id": \'INC-X\', "severity": \'High\', "root_cause": \'cpu\', '
    '"evidence": [\'ev1\', \'ev2\'], "affected_services": [\'svc\'], '
    '"confidence_score": 75, "recommended_fixes": [\'fix\'], '
    '"generated_summary": \'summary\'}',
    ["incident_id", "evidence", "generated_summary"],
)

# --- Case 7: Nested markdown backticks --------------------------------------
check(
    "Markdown code fences",
    '```json\n'
    '{"incident_id": "INC-MD", "severity": "High", "root_cause": "md", '
    '"evidence": ["e"], "affected_services": ["a"], "confidence_score": 60, '
    '"recommended_fixes": ["f"], "generated_summary": "s"}\n'
    '```',
    ["incident_id", "generated_summary"],
)

# --- Case 7b: Unwrapped backticks with explanatory text ---------------------
check(
    "Backticks mixed with explanation",
    '```Here is the report:\n'
    '{"incident_id": "INC-BT", "severity": "Medium", "root_cause": "dns", '
    '"evidence": ["e"], "affected_services": ["a"], "confidence_score": 60, '
    '"recommended_fixes": ["f"], "generated_summary": "s"}'
    '\n```\nI hope this helps!',
    ["incident_id", "severity"],
)

# --- Edge case: valid JSON already (should pass through unchanged) ----------
check(
    "Already-valid JSON (pass-through)",
    json.dumps({
        "incident_id": "INC-OK",
        "severity": "Low",
        "root_cause": "ok",
        "evidence": ["e"],
        "affected_services": ["a"],
        "confidence_score": 10,
        "recommended_fixes": ["f"],
        "generated_summary": "s",
    }),
    ["incident_id", "recommended_fixes"],
)

# --- Edge case: single-quoted values in arrays (after [ delimiter) ----------
check(
    "Array-starting single quotes",
    '{"incident_id": "INC-ARR", "evidence": [\'ev1\', \'ev2\', \'ev3\'], '
    '"generated_summary": "ok"}',
    ["incident_id", "evidence"],
)

# --- Extreme: combination of issues -----------------------------------------
check(
    "Combination: markdown + single quotes + trailing comma + extra text",
    '```json\n'
    "{'incident_id': 'INC-CBO', 'severity': 'Critical', 'root_cause': 'mixed', "
    "'evidence': ['e1', 'e2',], 'affected_services': ['svc',], "
    "'confidence_score': 95, 'recommended_fixes': ['fix',], "
    "'generated_summary': 'combo test',}\n"
    '```\n\nAdditional explanation here.',
    ["incident_id", "severity", "root_cause", "confidence_score", "generated_summary"],
)


print(f"\n{'='*50}")
print(f"Results: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED}")
if FAILED > 0:
    sys.exit(1)
