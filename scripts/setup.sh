#!/usr/bin/env bash
# =============================================================================
# KubeSage — one-shot setup (bash / Linux / macOS / Git-Bash-on-Windows)
# =============================================================================
# Runs Steps 1–6 of the manual install path described in README.md:
#
#   1. Create Python venv (.venv)
#   2. Install dependencies (requirements.txt) + NLTK tokenizer data
#   3. Generate synthetic K8s incident data  → data/synthetic_incidents.json
#   4. Compute SBERT embeddings              → embeddings/incident_embeddings.npy
#   5. Build persistent ChromaDB index       → vector_db/chroma_store/
#   6. Run retrieval evaluation              → results/retrieval_eval_results.json
#
# Optional step (slow, ~28 min on CPU):
#   --with-real-eval    → also runs paper/run_real_eval.py
#
# ─── Flags ────────────────────────────────────────────────────────────────
#   --reset              wipe .venv + chroma_store + .npy + results before setup
#   --skip-data          reuse existing data/synthetic_incidents.json
#   --skip-embed         reuse existing embeddings/incident_embeddings.npy
#   --skip-index         reuse existing vector_db/chroma_store (verify it has docs)
#   --skip-eval          skip both eval steps
#   --with-real-eval     also run paper/run_real_eval.py (28 min CPU)
#   -h | --help          show this help
#
# ─── Usage ────────────────────────────────────────────────────────────────
#   chmod +x scripts/setup.sh && ./scripts/setup.sh              # default
#   bash scripts/setup.sh --reset                                # wipe + rebuild
#   bash scripts/setup.sh --with-real-eval                       # adds real-eval (slow)
#   bash scripts/setup.sh --help                                 # full flag list
#
# Requires Python 3.10+ on PATH (matches README prerequisites).
#
# Logs:    logs/setup_<timestamp>.log
# Exit:    non-zero on first failure (set -e + pipefail)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/setup_${RUN_TS}.log"

banner() { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
ok()     { printf "\033[1;32m✓\033[0m %s\n" "$*"; }
warn()   { printf "\033[1;33m!\033[0m %s\n" "$*"; }
fail()   { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# ---------- arg parsing -----------------------------------------------------
RESET=0
SKIP_DATA=0
SKIP_EMBED=0
SKIP_INDEX=0
SKIP_EVAL=0
WITH_REAL_EVAL=0
print_help() {
    sed -n '2,40p' "$0"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset)          RESET=1; shift ;;
        --skip-data)      SKIP_DATA=1; shift ;;
        --skip-embed)     SKIP_EMBED=1; shift ;;
        --skip-index)     SKIP_INDEX=1; shift ;;
        --skip-eval)      SKIP_EVAL=1; shift ;;
        --with-real-eval) WITH_REAL_EVAL=1; shift ;;
        -h|--help)        print_help; exit 0 ;;
        *)                fail "Unknown flag: $1.  Re-run with --help." ;;
    esac
done

# ---------- Python sanity ---------------------------------------------------
command -v python3 >/dev/null 2>&1 && PY="python3" || PY="python"
command -v "${PY}" >/dev/null 2>&1 || fail "python3 / python not on PATH. Install Python 3.10+ from https://python.org."
PY_MAJOR="$(${PY} -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$(${PY} -c 'import sys; print(sys.version_info.minor)')"
[[ "${PY_MAJOR}" -ge 3 && "${PY_MINOR}" -ge 10 ]] || fail "Python >=3.10 required (found ${PY_MAJOR}.${PY_MINOR} via '${PY}').  Install 3.10/3.11 from https://python.org."

# ---------- --reset ---------------------------------------------------------
if [[ $RESET -eq 1 ]]; then
    banner "Reset: wiping .venv, vector_db/chroma_store, results, embeddings .npy, synthetic data"
    rm -rf .venv
    rm -rf vector_db/chroma_store
    rm -f  results/*.json 2>/dev/null || true
    rm -f  embeddings/incident_embeddings.npy 2>/dev/null || true
    rm -f  data/synthetic_incidents.json 2>/dev/null || true
    ok "Reset complete"
fi

# ---------- Step 1: venv ----------------------------------------------------
banner "Step 1/6 — Python virtual environment (.venv)"
if [[ ! -d ".venv" ]]; then
    python -m venv .venv
    ok "Created .venv"
else
    warn ".venv already exists — reusing. Use --reset to recreate."
fi
# shellcheck disable=SC1091
source .venv/bin/activate
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
[[ -x "${VENV_PY}" ]] || fail "venv activation succeeded but ${VENV_PY} not executable."
"${VENV_PY}" -m pip install --quiet --upgrade pip wheel setuptools 2>&1 | tail -3 || true
ok "venv active (Python ${PY_MAJOR}.${PY_MINOR}) — using ${VENV_PY}"

# ---------- Step 2: dependencies -------------------------------------------
banner "Step 2/6 — Install Python dependencies (requirements.txt) + NLTK data"
"${VENV_PY}" -m pip install -r requirements.txt 2>&1 | tail -10
"${VENV_PY}" -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" \
    2>&1 | tail -3 || warn "NLTK data fetch warning — runtime BLEU/ROUGE may fail until you re-run."
ok "Dependencies installed"

# ---------- Step 3: synthetic data -----------------------------------------
banner "Step 3/6 — Generate synthetic K8s incident data (500 incidents)"
if [[ $SKIP_DATA -eq 0 ]]; then
    if [[ -f data/synthetic_incidents.json && $RESET -eq 0 ]]; then
        warn "data/synthetic_incidents.json already exists — skipping. Use --reset to regenerate."
    else
        "${VENV_PY}" scripts/generate_data.py 2>&1 | tee -a "$LOG_FILE" | tail -10
        ok "Synthetic data ready at data/synthetic_incidents.json"
    fi
else
    warn "--skip-data: keeping existing data/synthetic_incidents.json"
fi

# ---------- Step 4: SBERT embeddings ---------------------------------------
banner "Step 4/6 — Compute SBERT embeddings (all-MiniLM-L6-v2, 384-dim)"
if [[ $SKIP_EMBED -eq 0 ]]; then
    mkdir -p embeddings
    if [[ -f embeddings/incident_embeddings.npy && $RESET -eq 0 ]]; then
        warn "embeddings/incident_embeddings.npy already exists — skipping. Use --reset to recompute."
    else
        "${VENV_PY}" scripts/build_embeddings.py 2>&1 | tee -a "$LOG_FILE" | tail -10
        ok "Embeddings saved"
    fi
else
    warn "--skip-embed: reusing existing embeddings/incident_embeddings.npy"
fi

# ---------- Step 5: ChromaDB index ------------------------------------------
banner "Step 5/6 — Build persistent ChromaDB index"
if [[ $SKIP_INDEX -eq 0 ]]; then
    NEED_BUILD=1
    if [[ -d vector_db/chroma_store && $RESET -eq 0 ]]; then
        N_DOCS="$(python -c "from vector_db.build_index import VectorDatabase; print(VectorDatabase().count())" 2>/dev/null | tr -d '[:space:]' || echo 0)"
        if [[ "${N_DOCS}" =~ ^[0-9]+$ && "${N_DOCS}" -gt 0 ]]; then
            warn "vector_db/chroma_store already has ${N_DOCS} docs — skipping. Use --reset to rebuild."
            NEED_BUILD=0
        fi
    fi
    if [[ $NEED_BUILD -eq 1 ]]; then
        "${VENV_PY}" scripts/index_vectors.py 2>&1 | tee -a "$LOG_FILE" | tail -10
        ok "ChromaDB index built at vector_db/chroma_store/"
    fi
else
    warn "--skip-index: reusing existing vector_db/chroma_store/"
fi

# ---------- Step 6: Evaluation ---------------------------------------------
banner "Step 6/6 — Run evaluation (writes results/*.json)"
if [[ $SKIP_EVAL -eq 0 ]]; then
    mkdir -p results
    if [[ ! -f results/retrieval_eval_results.json || $RESET -eq 1 ]]; then
        "${VENV_PY}" paper/run_retrieval_eval.py 2>&1 | tee -a "$LOG_FILE" | tail -10
        ok "Retrieval evaluation written → results/retrieval_eval_results.json"
    else
        warn "results/retrieval_eval_results.json already exists — skipping. Use --reset to rerun."
    fi

    if [[ $WITH_REAL_EVAL -eq 1 ]]; then
        echo
        echo "⚠  Real eval with SmolLM2-1.7B on CPU takes ~28 minutes."
        "${VENV_PY}" paper/run_real_eval.py 2>&1 | tee -a "$LOG_FILE" | tail -20
        ok "Real evaluation written → results/real_eval_results.json"
    else
        echo
        echo "ℹ  Skipped paper/run_real_eval.py (28 min CPU). Pass --with-real-eval to run it:"
        echo "       bash scripts/setup.sh --skip-data --skip-embed --skip-index --with-real-eval"
    fi
else
    warn "--skip-eval: skipping all evaluation steps"
fi

# ---------- Done ------------------------------------------------------------
banner "Setup complete ✓"
cat <<EOF

Next steps:
  • Venv is already active in this shell. Run the dashboard:
        python -m streamlit run frontend/app.py --server.port 8501
       → open http://localhost:8501

  • Or run the FastAPI backend:
        python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
       → open http://localhost:8000/docs

  • To run the slow real-eval later (~28 min CPU):
        source .venv/bin/activate
        python paper/run_real_eval.py

Logs: $LOG_FILE
EOF
