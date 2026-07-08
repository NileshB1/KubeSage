"""
Regression Test — Sidebar Nav Substrings
========================================
Asserts that sidebar nav labels do not collide with elements in the main body content,
preventing brittle locators during automated UI capture and validation.
"""
from __future__ import annotations

import os
import re
import urllib.error
import urllib.request

import pytest

STREAMLIT_URL = os.environ.get("STREAMLIT_URL", "http://localhost:8501").rstrip("/")

# Sidebar-nav labels that scripts/capture_screenshots.py expects to click.
# Mirrors the keys in click_sidebar_radio()'s EMOJI dict AND the entries in
# frontend/app.py's render_sidebar() radio. If frontend/app.py adds a new
# sidebar entry, add it here AND to capture_screenshots.py together.
SIDEBAR_NAV_LABELS = [
    "Overview",
    "Investigation",
    "Search",
    "Reports",
    "Evaluation",
]


# ─── module-scoped fixtures ─────────────────────────────────────────────

def _streamlit_is_up(url: str, timeout_s: int = 3) -> bool:
    """Quick urllib probe so we skip cleanly when no Streamlit is around."""
    try:
        with urllib.request.urlopen(f"{url}/_stcore/health", timeout=timeout_s) as r:
            return r.status == 200 and r.read().decode().strip() == "ok"
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


@pytest.fixture(scope="module")
def streamlit_page():
    """A Playwright page attached to a freshly-booted headless Chromium.

    Skips the entire module's tests when Streamlit is unreachable or the
    Playwright Python package is missing. Either condition is the normal
    state on a developer machine that isn't currently running Streamlit --
    CI is expected to boot Streamlit first via the screenshot-jobs workflow.
    """
    if not _streamlit_is_up(STREAMLIT_URL):
        pytest.skip(
            f"Streamlit not reachable at {STREAMLIT_URL} -- "
            f"start it with `python -m streamlit run frontend/app.py "
            f"--server.port 8501 --server.headless true` before running this test."
        )

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        pytest.skip("Playwright is not installed; pip install playwright && "
                    "playwright install chromium")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
            page = ctx.new_page()
            page.goto(f"{STREAMLIT_URL}/", wait_until="domcontentloaded",
                      timeout=60_000)
            # Streamlit cold-compile + first render can take 30 - 60 s; wait
            # for the sidebar shell to mount AND for the radio nav control
            # inside it to populate. The shell mounts early during bootstrap
            # but the radio labels (and hence the labels we assert on) only
            # appear after the React tree has rendered inside the sidebar.
            # Without this second wait, _sidebar_text() returns empty and
            # the sanity test falsely fails on a freshly-loaded sidebar.
            page.wait_for_selector(
                'section[data-testid="stSidebar"]',
                timeout=60_000,
            )
            page.wait_for_selector(
                'section[data-testid="stSidebar"] [data-testid="stRadio"]',
                timeout=60_000,
            )
            # Settle any websocket-driven RSC traffic so body text is stable.
            # We mirror scripts/capture_screenshots.py here: networkidle on
            # a Streamlit WS can keep ticking past the budget; treat a
            # timeout as "good enough" rather than failing the whole test.
            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except PWTimeout:
                pass
            yield page
        finally:
            browser.close()


# ─── the actual assertions ──────────────────────────────────────────────

def _sidebar_text(page) -> str:
    """The visible text inside the sidebar nav region."""
    return page.locator(
        'section[data-testid="stSidebar"]'
    ).evaluate("el => el.innerText")


def _main_content_titles(page) -> dict:
    """Heading + browser-tab text in body, sidebar subtree excluded.

    The original brittle-locator bug shape was a substring collision with a
    *static* page-title element -- either Streamlit's ``page_title``
    (which sets ``document.title`` via ``st.set_page_config``) or an
    ``st.title`` / ``st.header`` / ``st.subheader`` rendering as ``<h1>`` /
    ``<h2>`` / ``<h3>`` in body. Walking ``document.body.innerText``
    indiscriminately produced false-positives on any paragraph that
    *mentions* the substring (e.g., a pipeline diagram "Preprocessing ->
    Embeddings -> Search -> RAG -> LLM -> Report" would have flagged the
    sidebar nav label "Search" without ever affecting the capture script).
    The user's spec is "page-title match", so we restrict the assertion
    surface to ``document.title`` plus every ``h1``-``h6`` /
    ``[role="heading"]`` element outside the sidebar subtree. That
    population is exactly the one the original brittle-locator bug walked.
    """
    return page.evaluate(
        """
        () => {
            const title = document.title || '';
            const headings = [];
            for (const e of document.body.querySelectorAll(
                'h1, h2, h3, h4, h5, h6, [role="heading"]'
            )) {
                if (e.closest('section[data-testid="stSidebar"]')) continue;
                headings.push(e.innerText || '');
            }
            return { title: title, headings: headings };
        }
        """
    )


def _label_in(label: str, text: str) -> bool:
    """Substring match with Unicode-aware word boundaries.

    ``label in text`` works for ASCII labels like 'Search'. For Emoji-prefixed
    labels like '🔬 Investigation', Streamlit may render with U+2009 (thin
    space) or a variation selector between the emoji and the word; we use
    a regex with the embedded flag for robust word-boundary match.
    """
    return re.search(rf"(?u)\b{re.escape(label)}\b", text) is not None


# ─── the actual assertions ──────────────────────────────────────────────

def test_sidebar_labels_present_in_sidebar(streamlit_page):
    """Sanity: each expected sidebar-nav label is actually rendered.
    Catches a divergent sidebar text BEFORE the substring-collision check
    could yield a misleading all-clear because all labels vanished.
    """
    page = streamlit_page
    sidebar_text = _sidebar_text(page)
    # The sidebar in Streamlit 1.54 wraps labels in nested spans / inline
    # emoji glyphs; word-boundary regex catches these even when the textual
    # glue is a U+200B / U+2009 / variation selector.
    missing = [
        label for label in SIDEBAR_NAV_LABELS
        if not _label_in(label, sidebar_text)
    ]
    assert not missing, (
        f"Expected sidebar to contain every nav label, but missing: {missing}. "
        f"If you changed frontend/app.py's render_sidebar() entries, "
        f"update SIDEBAR_NAV_LABELS in this test AND the EMOJI dict in "
        f"scripts/capture_screenshots.py at the same time. "
        f"Sidebar rendered text (first 500 chars): {sidebar_text[:500]!r}"
    )


def test_sidebar_labels_do_not_collide_with_main_content(streamlit_page):
    """Each sidebar-nav label substring must NOT appear as a heading or
    inside ``document.title`` in the body region outside the sidebar.

    This is the exact bug shape that produced the original brittle-locator
    regression: the Streamlit ``page_title`` (set via ``st.set_page_config``
    and rendered into ``<title>``) and any ``st.title`` / ``st.header`` /
    ``st.subheader`` are static elements; Playwright's substring filter on
    the sidebar click could silently match a static title node before the
    clickable sidebar <label>, and the capture script would appear to hang
    at the next ``wait_for_selector``. Restricting the assertion to
    ``document.title`` + heading elements outside the sidebar matches the
    structural element population that defect lived in. A body *paragraph*
    containing the substring (e.g., a "Search" stage in a processing-flow
    diagram) is intentionally NOT considered a collision here -- the
    capture script does not walk body paragraphs for nav decisions.
    """
    page = streamlit_page
    titles = _main_content_titles(page)
    title_text = titles["title"]
    headings_text = "\n\n".join(titles["headings"])

    collisions = []
    for source_name, source_text in (
        ("title", title_text),
        ("headings", headings_text),
    ):
        for label in SIDEBAR_NAV_LABELS:
            match = re.search(rf"(?u)\b{re.escape(label)}\b", source_text)
            if match is None:
                continue
            idx = match.start()
            ctx_start = max(0, idx - 40)
            ctx_end = min(len(source_text), idx + len(label) + 40)
            context = source_text[ctx_start:ctx_end]
            collisions.append((label, source_name, context))

    assert not collisions, (
        "A sidebar-nav label substring appears as a heading or in "
        "document.title outside the sidebar -- the same shape as the "
        "historical brittle-locator bug. Either:\n"
        "  - rename the colliding heading / page_title (e.g. change "
        "    'Incident Overview Dashboard' to 'Incident Dashboard'), or\n"
        "  - tighten scripts/capture_screenshots.py for that label.\n"
        f"Collisions found: {collisions}"
    )


