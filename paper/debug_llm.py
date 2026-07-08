"""Debug script v2: test real LLM generation with improved prompting for JSON output."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from backend.config import settings

model_name = settings.LLM_MODEL_NAME

# Load model and tokenizer
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True,
    torch_dtype=torch.float32, low_cpu_mem_usage=True,
).cpu().eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

# Better system prompt - very explicit about JSON
system = (
    "You are a Kubernetes SRE. Your ONLY task is to output a single JSON object. "
    "Do not write any text before or after the JSON. "
    "The JSON must have these exact keys: "
    "incident_id, severity, root_cause, evidence, affected_services, "
    "confidence_score, recommended_fixes, generated_summary. "
    "Output nothing but the JSON object."
)

# User prompt with explicit example
user = (
    "Incident: Pod payment-service-7d4f production CrashLoopBackOff. "
    "Logs: connection refused to PostgreSQL, max_connections reached.\n\n"
    "Retrieved: INC-1010 ConnectionPoolExhaustion (DB pool starvation, fixed: increased pool). "
    "INC-1020 CrashLoopBackOff (missing DB secret, fixed: added secret).\n\n"
    "Output JSON (replace values with your analysis):\n"
    '{"incident_id":"INC-XXXX","severity":"Critical","root_cause":"...",'
    '"evidence":["..."],"affected_services":["..."],"confidence_score":95,'
    '"recommended_fixes":["..."],"generated_summary":"..."}'
)

# Build prompt using Qwen chat template
prompt = (
    f"<|im_start|>system\n{system}<|im_end|>\n"
    f"<|im_start|>user\n{user}<|im_end|>\n"
    f"<|im_start|>assistant\n"
    "{"  # Pre-fill with opening brace to force JSON
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
elapsed = time.time() - start

# Decode
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
response = "{" + tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Generated {len(generated_ids)} tokens in {elapsed:.1f}s ({len(response)} chars)")

# Save raw output
output_dir = Path(__file__).resolve().parent.parent / "results"
output_dir.mkdir(exist_ok=True)
with open(output_dir / "raw_llm_v2.txt", "w", encoding="utf-8") as f:
    f.write(response)

# Parse JSON
from models.rag_pipeline import ReportParser
report = ReportParser.extract_json(response)
if "error" in report:
    print(f"JSON parse FAILED: {report['error']}")
    print(f"Response preview: {response[:600]}")
else:
    validated = ReportParser.validate_report(report)
    print(f"SUCCESS! ID={validated.get('incident_id')}, "
          f"Severity={validated.get('severity')}, "
          f"Confidence={validated.get('confidence_score')}")
    with open(output_dir / "real_llm_report.json", "w", encoding="utf-8") as f:
        json.dump({"report": validated, "raw": response}, f, indent=2, default=str)
    print(f"Report saved: {output_dir / 'real_llm_report.json'}")
