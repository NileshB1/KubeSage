#!/usr/bin/env python
"""
Capture the KubeSage Streamlit dashboard into PNG screenshots for README.md.

Usage:
    # 1) Install Playwright + Chromium (one-time)
    pip install 'playwright>=1.40'
    playwright install chromium

    # 2) Start Streamlit on 8501 in another terminal
    python -m streamlit run frontend/app.py \
        --server.port 8501 --server.address 0.0.0.0 --server.headless true

    # 3) Wait ~30 s for first compile, then run:
    python scripts/capture_screenshots.py

Output (PNG, full-page, written to docs/screenshots/):
    01-overview.png             KPI cards + distribution charts
    02-investigation-form.png   Incident Description textarea (empty state)
    03-investigation-report.png Real RAG pipeline result (mock mode)
    04-search-results.png       ChromaDB semantic search results
    05-reports-dashboard.png    Reports KPIs + recent incidents table
    06-evaluation-retrieval.png Tab 1 — Precision@5, Recall@5, MRR, NDCG@5
    07-evaluation-generation.png Tab 2 — BLEU, ROUGE-L, Completeness
    08-evaluation-hallucination.png Tab 3 — Faithfulness + rate
    09-evaluation-per-sample.png Tab 4 — 5-row table (INC-01000 … INC-01006)

Notes:
- All captures are 1600 × 1000 viewport, full-page PNGs.
- Windows cp1252 consoles: emoji-free logs; ASCII-only status lines.
"""
from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PWTimeout, sync_playwright

BASE_URL = "http://localhost:8501"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit compile + RAG pipeline / embedding latencies
COLD_COMPILE_TIMEOUT_MS = 90_000
PAGE_TRANSITION_DELAY_S = 4
PIPELINE_RUN_DELAY_S = 12
# First-render selector waits. Bumped from defaults (45s / 15s) because
# Streamlit 1.54 cold-compile plus the first interactive widget hover can
# push the page-goto + first render past 50 s on a virgin worker.
FIRST_RENDER_WAIT_MS = 90_000
LATER_RENDER_WAIT_MS = 30_000
# Streamlit 1.54 (as of writing) keeps a websocket open after every tab
# click, so `networkidle` routinely times out. We treat the timeout as a
# successful settle and rely on the subsequent step's wait_for_selector
# to gate.
TAB_SETTLE_MS = 8_000
# Sidebar nav timeouts. The sidebar mounts quickly once Streamlit is warm,
# but the visibility check + click + post-click networkidle can all race
# the websocket.
SIDEBAR_VISIBLE_MS = 10_000
SIDEBAR_CLICK_MS = 10_000
SIDEBAR_SETTLE_MS = 10_000


def log(msg: str) -> None:
    """ASCII-only log line so Windows cp1252 consoles don't blow up on emoji."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def wait_for_streamlit() -> None:
    """Fail-fast: bail out cleanly if nothing is listening on BASE_URL."""
    try:
        with urllib.request.urlopen(BASE_URL, timeout=5) as r:
            r.read(1)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        log("=" * 70)
        log("ERROR: Streamlit is not running at " + BASE_URL)
        log("Underlying: " + str(e))
        log("")
        log("Start it first with:")
        log("    python -m streamlit run frontend/app.py \\")
        log("        --server.port 8501 --server.address 0.0.0.0 --server.headless true")
        log("Then wait ~30 s for the first cold compile and re-run.")
        log("=" * 70)
        sys.exit(2)


def wait_for_plots_settled(page: Page) -> None:
    """Wait until every Plotly chart on the page has finished rendering.

    Plotly's first render can take a couple of seconds after the surrounding
    div mounts. screenshotting mid-paint yields empty chart skeletons, so
    block until either there are no charts (no plots expected) OR every chart's
    JavaScript has produced SVG children.
    """
    page.wait_for_function(
        """() => {
            const charts = document.querySelectorAll('.stPlotlyChart, .js-plotly-plot');
            if (!charts || charts.length === 0) return true;
            return Array.from(charts).every(c => {
                const svg = c.querySelector('svg');
                return svg && svg.children && svg.children.length > 0;
            });
        }""",
        timeout=20_000,
    )


def click_sidebar_radio(page: Page, label_substring: str) -> None:
    """Click a sidebar nav item by **emoji-prefixed exact match**.

    Anchoring on the emoji prefix guarantees we match the real sidebar <label>
    — not a coincidental substring elsewhere on the page (e.g., the Streamlit
    page title rendered in <h1>, an admin badge, a status pill). Without this
    disambiguation, ``.first.click()`` could silently hit a static heading
    instead of the clickable nav radio and the script would then appear to
    hang at the next wait_for_selector.

    Historical note: the original hypothesis motivating this defense was that
    the Streamlit page title ``'KubeSage — AI Incident Investigation'``
    (rendered in <h1>) contained the substring ``'Investigation'`` and would
    be picked by ``.first`` over the sidebar nav label ``'🔬 Investigation'``,
    causing a silent no-op click. That specific collision no longer exists —
    the page title was renamed to ``'KubeSage — Incident Analysis'`` in
    ``frontend/app.py`` (page_title kwarg) so the substring ``Investigation``
    no longer appears in the rendered title on any page (as of writing).
    The emoji-prefix
    defense is kept as **defensive-coding** rather than bug-fix because the
    same shape of collision could resurface via a future title change, an
    admin overlay, or a status pill — cheap insurance against any future
    caller of Playwright's substring semantics.
    """
    EMOJI = {
        "Overview": "🏠",
        "Investigation": "🔬",
        "Search": "🔎",
        "Reports": "📊",
        "Evaluation": "📈",
    }
    target = (EMOJI.get(label_substring, "") + " " + label_substring).strip()
    sidebar = page.locator('section[data-testid="stSidebar"]')
    sidebar.wait_for(state="visible", timeout=SIDEBAR_VISIBLE_MS)
    sidebar.get_by_text(target, exact=True).first.click(timeout=SIDEBAR_CLICK_MS)
    # Settle websocket-driven rerender after a sidebar nav click.
    try:
        page.wait_for_load_state("networkidle", timeout=SIDEBAR_SETTLE_MS)
    except PWTimeout:
        pass


def open_tab(page: Page, label_substring: str) -> None:
    """Click a ``st.tabs`` tab by its visible text.

    Belt-and-braces protection against the historical substring-collision bug:

    1. The selector ``button[role="tab"]`` anchors on the ARIA-stable button
       role, not a generic ``[role="tab"]`` that any descendant could match.
       Even if Streamlit's renderer changes, the WAI-ARIA tab role on the
       ``<button>`` element is the stable contract.
    2. The text filter uses ``filter(has_text=re.compile(rf"^(?:[\W\s]+)?{re.escape(label_substring)}$"))``
       — anchored full-string match with an OPTIONAL leading run of
       non-word / whitespace characters. This tolerates Streamlit's
       emoji-prefixed tab labels (e.g. ``'\U0001F4DD Generation Quality'``)
       without matching arbitrary heading chrome: a page that surfaces the
       same substring on a header, KPI label, or status pill cannot
       collide, because the anchored end (``$``) still pins the full
       string and the body is *exactly* the label.
    3. ``re.escape`` keeps future label additions (with regex metacharacters
       like ``.`` or ``()``) safe.

    Trade-off: a label typo will now surface as a Playwright TimeoutError
    rather than silently clicking the wrong tab. That is the correct
    fail-loud behaviour.
    """
    page.locator(
        'button[role="tab"]'
    ).filter(
        has_text=re.compile(rf"^(?:[\W\s]+)?{re.escape(label_substring)}$")
    ).first.click(timeout=LATER_RENDER_WAIT_MS)
    try:
        page.wait_for_load_state("networkidle", timeout=TAB_SETTLE_MS)
    except PWTimeout:
        pass


def click_form_search(page: Page) -> None:
    """Click the Search-page form-submit button.

    Streamlit's `st.form_submit_button("🔎 Search", ...)` (frontend/app.py L624)
    always renders inside a `<div data-testid="stFormSubmitButton">` wrapper,
    regardless of whether the surrounding `<form>` element itself survived any
    rerun race. We anchor on that wrapper + a regex-anchored text match so:
    - the sidebar radio (whose accessible name contains "Search" but is
      `role=radio`, not a button) is unambiguously excluded.
    - any U+00A0 / U+2009 / U+FE0F whitespace or variation-selector
      inserted by Streamlit's renderer between the emoji and the word
      "Search" is tolerated (the regex `\s*` eats one optional character).
    """
    btn = page.locator(
        'div[data-testid="stFormSubmitButton"] button'
    ).filter(has_text=re.compile(r"^🔎\s*Search\s*$")).first
    btn.wait_for(state="visible", timeout=LATER_RENDER_WAIT_MS)
    btn.click(timeout=LATER_RENDER_WAIT_MS)


def main() -> int:
    log("[1/2] Health-checking " + BASE_URL + " ...")
    wait_for_streamlit()
    log("NOTE: this script assumes the 'mock' LLM mode (~1 s per case).")
    log("      If you started Streamlit with LLM_MODE=local, the capture will")
    log("      take ~10x longer (SmolLM2-1.7B CPU inference).\n")
    log("[2/2] Output directory: " + str(OUT_DIR))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()

        # ─── 01 Overview (default landing) ──────────────────────────────
        log("-> 01 Overview ...")
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=COLD_COMPILE_TIMEOUT_MS)
        page.wait_for_timeout(8000)
        page.wait_for_selector("text=/^Total Incidents\\s*$/", timeout=FIRST_RENDER_WAIT_MS)
        page.wait_for_timeout(1500)
        wait_for_plots_settled(page)
        page.screenshot(path=str(OUT_DIR / "01-overview.png"), full_page=True)

        # ─── 02 Investigation form (before submit) ──────────────────────
        log("-> 02 Investigation form ...")
        click_sidebar_radio(page, "Investigation")
        page.wait_for_timeout(PAGE_TRANSITION_DELAY_S * 1000)
        page.wait_for_selector("text=/^Incident Description\\s*$/", timeout=LATER_RENDER_WAIT_MS)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT_DIR / "02-investigation-form.png"), full_page=True)

        # ─── 03 Run RAG pipeline (mock mode, ~1 s) ──────────────────────
        log("-> 03 Investigation report (real RAG) ...")
        page.get_by_role("button", name=re.compile(r"^Investigate Incident\s*$")).click()
        page.wait_for_timeout(PIPELINE_RUN_DELAY_S * 1000)
        page.wait_for_selector("text=/^Investigation complete\\s*$/", timeout=LATER_RENDER_WAIT_MS)
        # Best-effort: spinner overlay may vary between Streamlit versions.
        try:
            page.wait_for_function(
                "() => !document.querySelector('.stSpinner, [data-testid=\"stSpinner\"]')",
                timeout=15_000,
            )
        except PWTimeout:
            log("   (warn) spinner overlay selector not found in this Streamlit build; continuing")
        page.wait_for_timeout(2500)
        wait_for_plots_settled(page)
        page.screenshot(path=str(OUT_DIR / "03-investigation-report.png"), full_page=True)

        # ─── 04 Semantic search ─────────────────────────────────────────
        log("-> 04 Search results (ChromaDB) ...")
        click_sidebar_radio(page, "Search")
        page.wait_for_timeout(PAGE_TRANSITION_DELAY_S * 1000)
        page.get_by_placeholder(re.compile(r"^e\.g\., 'database connection timeout'\s*$")).fill(
            "database connection refused to PostgreSQL"
        )
        page.wait_for_timeout(800)
        click_form_search(page)
        page.wait_for_timeout(PIPELINE_RUN_DELAY_S * 1000)
        # The success message is "[OK] Found N matches in X ms" — wait for the
        # strong phrase first (catches both the render and the timing).
        page.wait_for_selector("text=/^Found\\s+\\d+\\s+matches\\s+in/i", timeout=LATER_RENDER_WAIT_MS)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT_DIR / "04-search-results.png"), full_page=True)

        # ─── 05 Reports dashboard ───────────────────────────────────────
        log("-> 05 Reports dashboard ...")
        click_sidebar_radio(page, "Reports")
        page.wait_for_timeout(PAGE_TRANSITION_DELAY_S * 1000)
        page.wait_for_selector("text=/^Recent Incidents\\s*$/", timeout=LATER_RENDER_WAIT_MS)
        page.wait_for_timeout(2000)
        wait_for_plots_settled(page)
        page.screenshot(path=str(OUT_DIR / "05-reports-dashboard.png"), full_page=True)

        # ─── 06 Evaluation: Retrieval tab ────────────────────────────────
        log("-> 06 Evaluation / Retrieval ...")
        click_sidebar_radio(page, "Evaluation")
        page.wait_for_timeout(PAGE_TRANSITION_DELAY_S * 1000)
        page.wait_for_selector("text=/^Precision@5\\s*$/", timeout=LATER_RENDER_WAIT_MS)
        page.wait_for_timeout(1500)
        wait_for_plots_settled(page)
        page.screenshot(path=str(OUT_DIR / "06-evaluation-retrieval.png"), full_page=True)

        # ─── 07 Generation tab ─────────────────────────────────────────
        log("-> 07 Evaluation / Generation ...")
        open_tab(page, "Generation Quality")
        page.wait_for_timeout(2500)
        wait_for_plots_settled(page)
        page.screenshot(path=str(OUT_DIR / "07-evaluation-generation.png"), full_page=True)

        # ─── 08 Hallucination tab ──────────────────────────────────────
        log("-> 08 Evaluation / Hallucination ...")
        open_tab(page, "Hallucination")
        page.wait_for_timeout(2500)
        wait_for_plots_settled(page)
        page.screenshot(path=str(OUT_DIR / "08-evaluation-hallucination.png"), full_page=True)

        # ─── 09 Per-Sample tab ─────────────────────────────────────────
        log("-> 09 Evaluation / Per-Sample ...")
        open_tab(page, "Per-Sample Detail")
        page.wait_for_timeout(2500)
        wait_for_plots_settled(page)
        page.screenshot(path=str(OUT_DIR / "09-evaluation-per-sample.png"), full_page=True)

        browser.close()

    files = sorted(OUT_DIR.glob("0?-*.png"))
    log("")
    log("OK: captured " + str(len(files)) + " screenshots")
    for f in files:
        size_kb = f.stat().st_size // 1024
        log("   " + f.name.ljust(40) + " " + str(size_kb).rjust(5) + " KB")
    if len(files) != 9:
        log("WARN: expected 9 PNGs (01..09); got " + str(len(files)))
        log("      Some sections were skipped; re-run or fix and re-run.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
