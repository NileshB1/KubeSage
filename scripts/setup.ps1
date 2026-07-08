<#
.SYNOPSIS
    KubeSage — one-shot setup (Windows PowerShell 5+ / PowerShell Core 7+).

.DESCRIPTION
    Mirrors scripts/setup.sh but written for Windows hosts. Runs the same
    Steps 1-6 documented in README.md:

      1. Create a Python venv (.venv)
      2. Install dependencies (requirements.txt) + NLTK tokenizer data
      3. Generate synthetic K8s incident data
      4. Compute SBERT embeddings
      5. Build the persistent ChromaDB vector index
      6. Run retrieval evaluation

    Optional:
      -WithRealEval   Also run paper/run_real_eval.py (SmolLM2-1.7B, ~28 min CPU)

.PARAMETER Reset
    Wipe .venv, vector_db/chroma_store, embeddings .npy, results prior to setup.
.PARAMETER SkipData
    Keep existing data/synthetic_incidents.json (do not regenerate).
.PARAMETER SkipEmbed
    Keep existing embeddings/incident_embeddings.npy (do not recompute).
.PARAMETER SkipIndex
    Keep existing vector_db/chroma_store (do not rebuild).
.PARAMETER SkipEval
    Skip all evaluation steps.
.PARAMETER WithRealEval
    Also run paper/run_real_eval.py (slow; SmolLM2-1.7B on CPU ≈ 28 min).

.EXAMPLE
    PS> .\scripts\setup.ps1                  # default
    PS> .\scripts\setup.ps1 -Reset           # wipe + rebuild
    PS> .\scripts\setup.ps1 -WithRealEval    # adds slow real-eval

.NOTES
    Author: KubeSage team (NCI MSc)
    Logs:   logs\setup_<timestamp>.log
#>

# --- param() MUST be the FIRST executable block (before any function decls) -
param(
    [switch]$Reset       = $false,
    [switch]$SkipData    = $false,
    [switch]$SkipEmbed   = $false,
    [switch]$SkipIndex   = $false,
    [switch]$SkipEval    = $false,
    [switch]$WithRealEval= $false,
    [switch]$h           = $false
)

# --- Hard fail on first error ----------------------------------------------
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

# --- Resolve project root from this script's directory ----------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir '..')
Set-Location $ProjectRoot

# --- Logging ----------------------------------------------------------------
$LogDir  = Join-Path $ProjectRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunTs   = (Get-Date -Format 'yyyyMMdd_HHmmss')
$LogFile = Join-Path $LogDir ("setup_{0}.log" -f $RunTs)

# --- color helpers ----------------------------------------------------------
function Write-Banner($msg) { Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-OK($msg)     { Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Warn($msg)   { Write-Host "! $msg" -ForegroundColor Yellow }
function Write-Fail($msg)   { Write-Host "✗ $msg" -ForegroundColor Red; exit 1 }

# --- Help -------------------------------------------------------------------
if ($h) {
@'
KubeSage one-shot setup (Windows PowerShell)

Usage:
  .\scripts\setup.ps1                  # default
  .\scripts\setup.ps1 -Reset           # wipe .venv / chroma_store / results / .npy
  .\scripts\setup.ps1 -SkipData        # keep existing synthetic data
  .\scripts\setup.ps1 -SkipEmbed       # keep existing embeddings .npy
  .\scripts\setup.ps1 -SkipIndex       # keep existing chroma_store
  .\scripts\setup.ps1 -SkipEval        # skip evaluation
  .\scripts\setup.ps1 -WithRealEval    # also run slow paper\run_real_eval.py

Requires Python 3.10+ on PATH (matches README prerequisites).

Logs:    logs\setup_<timestamp>.log
'@
    exit 0
}

# --- Python sanity ----------------------------------------------------------
try {
    $pyVerRaw   = (python --version 2>&1) | Out-String
    $pyVersion  = ($pyVerRaw -replace 'Python ','').Trim()
} catch {
    Write-Fail "python not on PATH."
}
$pyMajor = 0; $pyMinor = 0
if ($pyVersion -match '^(\d+)\.(\d+)') {
    $pyMajor = [int]$Matches[1]; $pyMinor = [int]$Matches[2]
}
if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 10)) {
    Write-Fail "Python >=3.10 required (found $pyVersion). Install Python 3.10 or 3.11 from https://python.org."
}

# --- -Reset -----------------------------------------------------------------
if ($Reset) {
    Write-Banner "Reset: wiping .venv, vector_db\chroma_store, results, embeddings .npy, synthetic data"
    foreach ($p in @('.venv',
                     (Join-Path $ProjectRoot 'vector_db\chroma_store'),
                     (Join-Path $ProjectRoot 'embeddings\incident_embeddings.npy'),
                     (Join-Path $ProjectRoot 'data\synthetic_incidents.json'))) {
        if (Test-Path $p) { Remove-Item -Recurse -Force $p }
    }
    Get-ChildItem (Join-Path $ProjectRoot 'results') -Filter '*.json' -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Write-OK "Reset complete"
}

# --- Step 1: venv -----------------------------------------------------------
Write-Banner "Step 1/6 — Python virtual environment (.venv)"
if (-not (Test-Path '.venv')) {
    python -m venv .venv
    Write-OK "Created .venv"
} else {
    Write-Warn ".venv exists — reusing. Use -Reset to recreate."
}
& .\.venv\Scripts\Activate.ps1
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
& $VenvPython -m pip install --quiet --upgrade pip wheel setuptools 2>&1 | Out-Null
Write-OK "venv active (Python $pyVersion) — using $VenvPython"

# --- Step 2: dependencies + NLTK -------------------------------------------
Write-Banner "Step 2/6 — Install Python dependencies (requirements.txt) + NLTK data"
& $VenvPython -m pip install -r requirements.txt 2>&1 | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
try {
    & $VenvPython -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" 2>&1 | Out-Null
} catch {
    Write-Warn "NLTK data fetch warning — runtime BLEU/ROUGE may fail until re-run."
}
Write-OK "Dependencies installed"

# --- Step 3: synthetic data -------------------------------------------------
Write-Banner "Step 3/6 — Generate synthetic K8s incident data (500 incidents)"
$dataFile = Join-Path $ProjectRoot 'data\synthetic_incidents.json'
if ($SkipData) {
    Write-Warn "-SkipData: keeping existing $dataFile"
} elseif ((Test-Path $dataFile) -and (-not $Reset)) {
    Write-Warn "$dataFile already exists — skipping. Use -Reset to regenerate."
} else {
    & $VenvPython scripts/generate_data.py 2>&1 | Tee-Object -Append -FilePath $LogFile | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
    Write-OK "Synthetic data ready at $dataFile"
}

# --- Step 4: embeddings ----------------------------------------------------
Write-Banner "Step 4/6 — Compute SBERT embeddings (all-MiniLM-L6-v2, 384-dim)"
$embFile = Join-Path $ProjectRoot 'embeddings\incident_embeddings.npy'
if ($SkipEmbed) {
    Write-Warn "-SkipEmbed: reusing existing $embFile"
} elseif ((Test-Path $embFile) -and (-not $Reset)) {
    Write-Warn "$embFile already exists — skipping. Use -Reset to recompute."
} else {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot 'embeddings') | Out-Null
    & $VenvPython scripts/build_embeddings.py 2>&1 | Tee-Object -Append -FilePath $LogFile | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
    Write-OK "Embeddings saved"
}

# --- Step 5: ChromaDB index -----------------------------------------------
Write-Banner "Step 5/6 — Build persistent ChromaDB index"
$chromaDir = Join-Path $ProjectRoot 'vector_db\chroma_store'
$needBuild = $true
if ($SkipIndex) {
    Write-Warn "-SkipIndex: reusing existing $chromaDir"
    $needBuild = $false
} elseif ((Test-Path $chromaDir) -and (-not $Reset)) {
    $nDocs = '0'
    try {
        $nDocs = (& $VenvPython -c "from vector_db.build_index import VectorDatabase; print(VectorDatabase().count())" 2>$null).Trim()
    } catch { $nDocs = '0' }
    if (-not ($nDocs -match '^\d+$')) { $nDocs = '0' }
    if ([int]$nDocs -gt 0) {
        Write-Warn "$chromaDir already has $nDocs docs — skipping. Use -Reset to rebuild."
        $needBuild = $false
    }
}
if ($needBuild) {
    & $VenvPython scripts/index_vectors.py 2>&1 | Tee-Object -Append -FilePath $LogFile | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
    Write-OK "ChromaDB index built at $chromaDir"
}

# --- Step 6: Evaluation ----------------------------------------------------
Write-Banner "Step 6/6 — Run evaluation (writes results\*.json)"
$resultsDir = Join-Path $ProjectRoot 'results'
$retJson    = Join-Path $resultsDir    'retrieval_eval_results.json'
$realJson   = Join-Path $resultsDir    'real_eval_results.json'
if ($SkipEval) {
    Write-Warn "-SkipEval: skipping all evaluation steps"
} else {
    if ((-not (Test-Path $retJson)) -or $Reset) {
        & $VenvPython paper/run_retrieval_eval.py 2>&1 | Tee-Object -Append -FilePath $LogFile | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
        Write-OK "Retrieval evaluation → $retJson"
    } else {
        Write-Warn "$retJson already exists — skipping. Use -Reset to rerun."
    }
    if ($WithRealEval) {
        Write-Host "`n⚠  Real eval with SmolLM2-1.7B on CPU takes ~28 minutes." -ForegroundColor Yellow
        & $VenvPython paper/run_real_eval.py 2>&1 | Tee-Object -Append -FilePath $LogFile | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
        Write-OK "Real evaluation → $realJson"
    } else {
        Write-Host "`nℹ  Skipped paper\run_real_eval.py (28 min CPU). Pass -WithRealEval to run it:" -ForegroundColor Cyan
        Write-Host "    .\scripts\setup.ps1 -SkipData -SkipEmbed -SkipIndex -WithRealEval" -ForegroundColor Cyan
    }
}

# --- Done ------------------------------------------------------------------
Write-Banner "Setup complete ✓"
@"
Next steps (Windows):

  • Activate the venv and start the Dashboard:
       .\.venv\Scripts\Activate.ps1
       python -m streamlit run frontend\app.py --server.port 8501
     → open http://localhost:8501

  • Or start the FastAPI backend:
       .\.venv\Scripts\Activate.ps1
       python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
     → open http://localhost:8000/docs

  • To run the slow real-eval later (~28 min CPU):
       .\.venv\Scripts\Activate.ps1
       python paper\run_real_eval.py

Logs: $LogFile
"@
