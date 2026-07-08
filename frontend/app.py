"""
A Streamlit based web dashboard

"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

#TODO need to remove?
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.rag_pipeline import RAGPipeline, ReportParser

#page config should be first streamlit command
st.set_page_config(
    page_title="KubeSage - Incident Analysis",
    page_icon="🔍",layout="wide", initial_sidebar_state="expanded",)


# Session State Initialization


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    defaults: dict[str, Any] = {
        "incidents": [], "reports": [],
        "embedding_model": "all-MiniLM-L6-v2",  "rag_enabled": True,
        "top_k": 5, "dark_mode": True,  "_theme_sig": None, 
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()



# Cached Data + CSS  

@st.cache_data
def get_static_chart_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic distribution data: incident types + severities."""
    type_data = pd.DataFrame({
        "Type": ["OOMKilled", "CrashLoopBackOff", "ImagePullBackOff",
                 "ConnectionPoolExhaustion", "DNSFailure", "CPUThrottling", "NetworkFailure"],
        "Count": [71, 71, 68, 81, 59, 67, 83],
    })
    sev_data = pd.DataFrame({
        "Severity": ["Critical", "High", "Medium", "Low"],
        "Count": [196, 172, 98, 34],
    })
    return type_data, sev_data


@st.cache_data(ttl=60)
def get_random_chart_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Random data simulating live stream — regenerated every 60s for freshness
    """
    # Seed by epoch-minute so the data changes every refresh window but is
    # stable within a window 
    seed = int(time.time() // 60)
    np.random.seed(seed)

    report_types = ["OOMKilled", "CrashLoopBackOff", "ConnectionPoolExhaustion",
                    "ImagePullBackOff", "DNSFailure", "CPUThrottling", "NetworkFailure"]
    report_data = pd.DataFrame({
        "Incident ID": [f"INC-{1000+i}" for i in range(10)],
        "Type": np.random.choice(report_types, 10),  "Severity": np.random.choice(["Critical", "High", "Medium"], 10),
        "Confidence": [round(85 + np.random.uniform(0, 14), 1) for _ in range(10)],
        "RAG": [True] * 8 + [False] * 2, "Date": pd.date_range("2026-06-01", periods=10, freq="D"),
    })
    n_points = 100
    points = pd.DataFrame({
        "x": np.random.randn(n_points) * 2,  "y": np.random.randn(n_points) * 2,
        "type": np.random.choice(
            ["OOMKilled", "CrashLoopBackOff", "NetworkFailure", "ConnectionPoolExhaustion", "DNSFailure"],
            n_points,
        ),
    })
    query_point = pd.DataFrame({"x": [0.5], "y": [0.3], "type": ["Query"]})
    models_list = ["all-MiniLM-L6-v2", "bge-base-en-v1.5", "e5-base"]
    metrics_data = pd.DataFrame({
        "Model": models_list * 4,  "Metric": ["Precision@5"] * 3 + ["Recall@5"] * 3 + ["MRR"] * 3 + ["NDCG@5"] * 3,
        "Score": [0.89, 0.91, 0.87, 0.92, 0.94, 0.89, 0.87, 0.90, 0.85, 0.91, 0.93, 0.88],
    })
    comparison = pd.DataFrame({
        "Method": ["Keyword Search", "SBERT Embeddings", "LLM Only", "RAG Pipeline"],
        "Accuracy": [0.62, 0.89, 0.71, 0.94],  "Hallucination Rate": [0.15, 0.08, 0.32, 0.07],
    })
    return report_data, points, query_point, metrics_data, comparison, report_types


@st.cache_data
def get_dark_css(is_dark: bool):
    """Build dark/light CSS string"""
    if is_dark:
        bg = "#0E1117"
        secondary_bg = "#1A1D24"
        text = "#E0E0E0"
        accent = "#4ECDC4"
    else:
        bg = "#FFFFFF"
        secondary_bg = "#F5F5F5"
        text = "#333333"
        accent = "#2563EB"

    return f"""
    <style>
    .stApp {{
        background-color: {bg};
    }}
    .main .block-container {{
        padding-top: 1rem;
    }}

    .stMetric {{
        background-color: {secondary_bg};  padding: 1rem; border-radius: 8px;
        border: 1px solid {accent}33;
    }}

    .report-box {{
        background-color: {secondary_bg};
        border: 1px solid {accent}66;
        border-radius: 12px; padding: 1.5rem;
        margin: 1rem 0;  font-family: 'Courier New', monospace; color: {text};
    }}
    .sidebar .sidebar-content {{
        background-color: {secondary_bg};
    }}
    h1, h2, h3, p {{
        color: {text} !important;
    }}
    .stButton>button {{
        background-color: {accent};color: white; border-radius: 8px; padding: 0.5rem 1.5rem;
        font-weight: 600;   border: none;  transition: all 0.2s;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);    box-shadow: 0 4px 12px {accent}66;
    }}
    .kpi-card {{
        background: linear-gradient(135deg, {accent}22, {secondary_bg});
        border: 1px solid {accent}44;  border-radius: 16px;  padding: 1.5rem; text-align: center;
        transition: all 0.3s;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);   box-shadow: 0 8px 24px {accent}33;
    }}
    </style>
    """


def apply_dark_mode() -> None:
    """Inject dark/light CSS via cached strin
    """
    dark_mode = st.session_state.get("dark_mode", True)
    sig = ("dark",) if dark_mode else ("light",)
    if st.session_state.get("_theme_sig") != sig:
        st.markdown(get_dark_css(dark_mode), unsafe_allow_html=True)
        st.session_state["_theme_sig"] = sig


apply_dark_mode()


#cached pipeline resource


@st.cache_resource
def get_pipeline(llm_mode: str = "mock") -> RAGPipeline:
    """
    Load the RAGPipeline once and reuse across Streamlit reruns
    """
    return RAGPipeline(llm_mode=llm_mode)


# Real-Data Cached Resources (ChromaDB and SBERT and JSON metrics)


@st.cache_resource
def get_embedding_generator() -> Any:
    """Load SentenceTransformer-backed EmbeddingGenerator. None if unavailable"""
    try:
        from embeddings.generate_embeddings import EmbeddingGenerator
        model_name = st.session_state.get("embedding_model", "all-MiniLM-L6-v2")
        return EmbeddingGenerator(model_name=model_name)
    except Exception:
        return None


@st.cache_resource
def get_vector_db() -> Any:
    """Load ChromaDB PersistentClient + collection. 
    None if not available."""
    try:
        from vector_db.build_index import VectorDatabase
        return VectorDatabase(enable_lazy_init=True)
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_db_snapshot(limit: int = 50) -> list[dict[str, Any]]:
    """Fetch up to limit incident metadatas from ChromaDB (cached 5 min)."""
    vdb = get_vector_db()
    if vdb is None or vdb.count() == 0:
        return []
    snap = vdb.collection.get(limit=limit, include=["metadatas"])
    return snap.get("metadatas") or []


@st.cache_data(ttl=120)
def get_vector_stats() -> dict[str, Any]:
    """Aggregate counts from ChromaDB by type and severity (cached 2 min)."""
    vdb = get_vector_db()
    if vdb is None or vdb.count() == 0:
        return {"total": 0, "by_type": {}, "by_severity": {}}
    snap = vdb.collection.get(limit=vdb.count(), include=["metadatas"])
    metas = snap.get("metadatas") or []
    by_type: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for m in metas:
        by_type[m.get("incident_type", "Unknown")] = by_type.get(m.get("incident_type", "Unknown"), 0) + 1
        by_sev[m.get("severity", "Unknown")] = by_sev.get(m.get("severity", "Unknown"), 0) + 1
    return {"total": len(metas), "by_type": by_type, "by_severity": by_sev}


@st.cache_data(ttl=120)
def load_eval_metrics() -> tuple[dict | None, dict | None]:
    """Load retrieval and generation evaluation JSONs from results"""
    repo_root = Path(__file__).resolve().parent.parent
    out: dict[str, dict | None] = {"retrieval": None, "generation": None}
    for key, fname in (("retrieval", "retrieval_eval_results.json"),
                       ("generation", "real_eval_results.json")):
        p = repo_root / "results" / fname
        try:
            with open(p, "r", encoding="utf-8") as f:
                out[key] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            out[key] = None
    return out["retrieval"], out["generation"]



# Sidebar


def render_sidebar() -> None:
    """Render the sidebar with controls."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/?size=100&id=33049&format=png",
            width=60,
        )
        st.title("KubeSage 🔍")

        st.markdown("---")

        #navigation
        st.markdown("### 📋 Navigation")
        page = st.radio(
            "Select Section",
            ["🏠 Overview", "🔬 Investigation", "📊 Reports", "📈 Evaluation"],
            label_visibility="hidden",
        )

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")

        st.session_state.rag_enabled = st.toggle("RAG Enabled", value=True)
        st.session_state.top_k = st.slider(
            "Top-K Retrieval",  min_value=1,  max_value=20,  value=5,
            help="Number of similar incidents to retrieve",
        )

        embedding_model = st.selectbox(
            "Embedding Model",   ["all-MiniLM-L6-v2", "bge-base-en-v1.5", "e5-base"],
            index=0,
        )
        st.session_state.embedding_model = embedding_model

        st.markdown("---")

        st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=True)

        st.markdown("---")
        st.caption("KubeSage v1.0.0 | NCI MSc Research")
        st.caption("Deep Learning & Generative AI")


        return page



# Pages

def render_overview() -> None:
    """Render the overview dashboard page"""
    st.header("🏠 Incident Overview Dashboard")

    # KPI Cards (static HTML, no caching needed)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="kpi-card">
            <h1 style="color: #4ECDC4; margin: 0;">500</h1>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Total Incidents</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="kpi-card">
            <h1 style="color: #FF6B6B; margin: 0;">7</h1>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Incident Types</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="kpi-card">
            <h1 style="color: #45B7D1; margin: 0;">384</h1>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Embedding Dim</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="kpi-card">
            <h1 style="color: #FFEAA7; margin: 0;">93%</h1>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Avg Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    #Charts
    type_data, sev_data = get_static_chart_data()
    template = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Incident Distribution by Type")
        fig = px.bar(
            type_data,  x="Type", y="Count", color="Type",
            color_discrete_sequence=px.colors.qualitative.Bold,  template=template,
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Severity Distribution")
        fig = px.pie(sev_data,
            values="Count", names="Severity",
            color_discrete_sequence=["#FF6B6B", "#E67E22", "#F1C40F", "#2ECC71"],  hole=0.4, template=template, )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # System architecture
    #TODO Not showing architectute for now
    ##st.markdown("---")
    ##st.subheader("System Architecture")
    ##st.markdown("""
    ##```
    ##Data Sources → Preprocessing → Embeddings → ChromaDB → Search → RAG → LLM → Report
    ##```
    ##""")


def render_investigation() -> None:
    """Render the Investigation page"""
    st.header("Incident Investigation")

    # Input panel
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Incident Description")
        incident_text = st.text_area(
            "Describe the incident",
            value=(
                "Pod payment-service-7d4f in namespace production is in CrashLoopBackOff. "

                "Logs show: 'Error: connection refused' to PostgreSQL at 10.0.2.15:5432. "
                "The database pod is running but max_connections has been reached. "

                "This started after a deployment 30 minutes ago."
            ),
            height=200,
            placeholder="Paste incident description, logs, and context here....",
            label_visibility="hidden",
        )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            use_rag = st.toggle("Use RAG", value=st.session_state.rag_enabled, key="rag_toggle_inv",)
        with col_b:
            top_k = st.select_slider(
                "Top-K", options=[1, 3, 5, 10, 20], value=st.session_state.get("top_k", 5),  key="top_k_inv",
            )
        with col_c:
            llm_mode = st.selectbox(
                "LLM", options=["mock", "local"],
                index=0,  key="llm_mode_inv",   help="mock = instant deterministic; local = SmolLM2-1.7B (~230s/case, real LLM)",
            )

        investigate_btn = st.button(
            "🔍 Investigate Incident", type="primary",
            use_container_width=True,disabled=(len(incident_text.strip()) < 10),
        )

    #Run pipeline on click
    if investigate_btn:
        try:
            pipeline = get_pipeline(llm_mode=llm_mode)
            with st.spinner(
                "Running RAG pipeline (embedding -> retrieval -> LLM)...."
            ):
                results = pipeline.investigate(
                    incident_text, top_k=top_k, rag_enabled=use_rag,
                )
            st.session_state["last_investigation"] = results
            st.success(
                f"✅ Investigation complete in {results['processing_time_ms']:.0f} ms "
                f"({results['retrieval_count']} incidents retrieved)"
            )
        except Exception:
            st.exception(f"Investigation failed. See traceback below....")
            return

    #retrieved incidents sidebar
    results = st.session_state.get("last_investigation")
    with col2:
        st.subheader("Retrieved Incidents")
        if results and results.get("retrieved_incidents"):
            for i, item in enumerate(results["retrieved_incidents"][:5], 1):
                meta = item.get("metadata", {})
                score = item.get("similarity_score", 0.0)
                with st.expander(
                    f"{meta.get('incident_id', item.get('incident_id', f'#{i}'))} "
                    f"- {score:.2f}"
                ):
                    st.markdown(f"**Type:** {meta.get('incident_type', 'N/A')}")
                    st.markdown(f"**Severity:** {meta.get('severity', 'N/A')}")
                    st.markdown(f"**Root Cause:** {meta.get('root_cause', 'N/A')}")
                    st.markdown(f"**Resolution:** {meta.get('resolution', 'N/A')}")
                    services = meta.get("affected_services", "")
                    if services:
                        st.markdown(f"**Affected:** {services}")
        elif results:
            st.info("No incidents retrieved (RAG disabled or empty vector store).")

    #report display + downloads
    if results and "error" not in results.get("report", {}):
        st.markdown("---")
        report = results["report"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieved", results.get("retrieval_count", 0))
        col2.metric("Confidence", f"{results.get('confidence_score', 0):.0f}%")
        col3.metric(
            "RAG",
            "YES" if results.get("rag_enabled") else "NO"
        )
        col4.metric("Time", f"{results.get('processing_time_ms', 0):.0f} ms")

        st.subheader("📄 Generated Investigation Report")
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.text(ReportParser.format_for_display(report))
        st.markdown('</div>', unsafe_allow_html=True)

        #downloads
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("📥 Download JSON",
                data=json.dumps(report, indent=2, default=str),   file_name=f"{report.get('incident_id', 'report')}.json",
                mime="application/json", use_container_width=True,)
        with col2:
            txt = ReportParser.format_for_display(report)
            st.download_button(
                "📄 Download Report (TXT)",
                data=txt,  file_name=f"{report.get('incident_id', 'report')}.txt",
                mime="text/plain", use_container_width=True,
                help="Plain-text export. (True PDF requires optional reportlab.)",
            )
        with col3:
            if st.button("📤 Copy JSON to clipboard", use_container_width=True):
                st.code(json.dumps(report, indent=2, default=str), language="json")
    elif results and "error" in results.get("report", {}):
        st.markdown("---")
        st.error(f"Report generation failed: {results['report']['error']}")


def render_search() -> None:
    """Render the Semantic Search page backed by the real ChromaDB vector store."""
    st.header("🔎 Semantic Search")

    gen = get_embedding_generator()



    vdb = get_vector_db()

    if gen is None:
        st.error(
            "⚠️ Text embedding model failed to load. Ensure "
            "`sentence-transformers` is installed and the model is reachable."
        )
        return
    
    
    
    if vdb is None or vdb.count() == 0:
        st.warning(
            "ChromaDB is empty or uninitialized. Run "
            "`vector_db/build_index.py` first."
        )
        return

    #use a form so the embedding only fires on submit, not on every keystroke
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search incident database", placeholder="e.g., 'database connection timeout' or "
                "'memory leak causing pod crash'",
            )
        with col2:
            severity_filter = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"])
        top_k = st.select_slider("Top K",  options=[3, 5, 10, 20],
            value=st.session_state.get("top_k", 5),
        )
        submitted = st.form_submit_button("🔎 Search", type="primary", use_container_width=True)

    if submitted and search_query.strip():
        try:
            with st.spinner("Embedding query + vector search..."):
                emb = gen.generate_single_embedding(search_query)
                where = (
                    {"severity": severity_filter} if severity_filter != "All" else None
                )
                search_result = vdb.search(emb, top_k=top_k, where=where)
            hits = search_result["results"]
            
            
            st.success(f"✅ Found {search_result['count']} matches in "
                f"{search_result['query_time_ms']:.0f} ms "
                f"(model: `{st.session_state.embedding_model}`)")

            st.markdown(f"### Results for: *{search_query}*")
            for i, hit in enumerate(hits, 1):
                meta = hit.get("metadata", {})
                score = hit.get("similarity_score", 0.0)
                inc_id = hit.get("incident_id", "?")
                
                with st.container():
                    cols = st.columns([6, 1])
                    with cols[0]:
                        st.markdown(
                            f"**#{i} {inc_id}** — "
                            f"{meta.get('incident_type', 'N/A')}"
                        )
                        st.markdown(
                            f"📊 Similarity: **{score:.3f}** · "

                            f"Severity: `{meta.get('severity', 'N/A')}` · "
                            f"Source: `{meta.get('source', 'n/a')}`"
                        )
                        rc = meta.get("root_cause", "")
                        if rc:
                            st.caption(
                                f"Root cause: {rc[:200]}{'…' if len(rc) > 200 else ''}"
                            )
                    with cols[1]:
                        st.metric("Score", f"{score:.0%}")
        except Exception:
            st.exception(f"Search failed. See traceback below....")
    elif not search_query.strip():
        st.info("Enter a query above and click **Search** to find similar incidents "
            "from the ChromaDB vector store." )


def render_reports() -> None:
    """Render the Reports page backed by the real ChromaDB incident snapshot"""
    st.header("📊 Generated Reports")

    vdb = get_vector_db()
    if vdb is None or vdb.count() == 0:
        st.warning(
            "ChromaDB is empty or uninitialized. Run `vector_db/build_index.py` first."    )
        return

    stats = get_vector_stats()
    snapshot = get_db_snapshot(limit=50)
    _, generation_j = load_eval_metrics()

    #KPI cards
    total = stats["total"]
    n_types = len(stats["by_type"])
    if generation_j and generation_j.get("per_sample"):
        per_sample = generation_j["per_sample"]
        avg_conf = sum(s.get("report_confidence", 0) for s in per_sample) / len(per_sample)

        avg_gen_time = sum(s.get("gen_time_s", 0) for s in per_sample) / len(per_sample)
        completeness = sum(s.get("completeness", 0) for s in per_sample) / len(per_sample)
    else:
        avg_conf = avg_gen_time = completeness = 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Incidents", f"{total:,}")
    with col2:
        st.metric("Incident Types", f"{n_types}")
    with col3:
        st.metric("Avg Confidence (LLM-reported, n=5)",
            f"{avg_conf:.0f}%", help="Mean `report_confidence` from `results/real_eval_results.json`.", )
    with col4:
        st.metric(
            "Avg Gen Time (SmolLM2-1.7B, n=5)",
            f"{avg_gen_time:.0f}s", help=f"Completeness {completeness:.2f} across n=5 incidents.",
        )

    #type filter
    types_sorted = sorted(stats["by_type"].keys())
    selected_type = st.selectbox("Filter by Incident Type", ["All"] + types_sorted)

    st.markdown("---")
    st.subheader(f"Recent Incidents (snapshot of {len(snapshot)} of {total})")
    if snapshot:
        rows = []
        for m in snapshot:
            if selected_type != "All" and m.get("incident_type") != selected_type:
                continue
            rows.append({
                "Incident ID": m.get("incident_id", "?"),  "Type": m.get("incident_type", "N/A"),
                "Severity": m.get("severity", "N/A"), "Affected Services": m.get("affected_services", ""),
                "Root Cause": (m.get("root_cause") or "")[:120],
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info(f"No incidents of type `{selected_type}` in the current snapshot. "
                "Try a different type or clear the filter.")
    else:
        st.info("No incidents available in the snapshot.")

    st.markdown("---")
    template = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

    #Yype count chart + severity pie
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reports per Incident Type")
        type_df = pd.DataFrame(
            [
                {"Type": t, "Count": c}
                for t, c in sorted(
                    stats["by_type"].items(), key=lambda kv: -kv[1]
                )
            ]
        )
        if not type_df.empty:
            fig = px.bar(type_df, x="Type", y="Count", color="Type", template=template)
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Severity distribution")
        sev_df = pd.DataFrame(
            [
                {"Severity": s, "Count": c}
                for s, c in stats["by_severity"].items()
            ]
        )
        if not sev_df.empty:
            fig = px.pie(
                sev_df,
                values="Count", names="Severity",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4, template=template,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


def render_evaluation() -> None:
    """Render the Model Evaluation page loaded from real JSON result file."""
    st.header("📈 Model Evaluation")

    retrieval_j, generation_j = load_eval_metrics()

    if retrieval_j is None and generation_j is None:
        st.warning(
            "No evaluation results found in `results/`. "
            "Run `paper/run_retrieval_eval.py` and `paper/run_real_eval.py` first."
        )
        return

    experiment = st.selectbox(
        "Select Experiment",
        [
            "Experiment 1: Retrieval @ K (Precision / Recall / MRR / NDCG)",   "Experiment 2: Generation Quality (BLEU, ROUGE-L)",
            "Experiment 3: Hallucination (Faithfulness)","Experiment 4: Per-Sample Real Eval (n=5, SmolLM2-1.7B)",
        ],
    )

    template = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Retrieval Metrics", "📝 Generation Quality", "🎯 Hallucination",
        "📋 Per-Sample Detail",])

    #tab 1: Retrieval metrics
    with tab1:
        st.subheader("Retrieval Performance - real eval")
        if retrieval_j and "metrics" in retrieval_j:
            m = retrieval_j["metrics"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision@5", f"{m.get('precision', {}).get('@5', 0):.4f}")
            with col2:
                st.metric("Recall@5", f"{m.get('recall', {}).get('@5', 0):.4f}")
            with col3:
                st.metric("MRR", f"{m.get('mrr', {}).get('mean', 0):.4f}")
            with col4:
                st.metric("NDCG@5", f"{m.get('ndcg', {}).get('@5', 0):.4f}")

            rows: list[dict[str, Any]] = []
            for k in retrieval_j.get("k_values", [1, 3, 5, 10]):
                rows.append({"K": f"@{k}", "Metric": "Precision",
                             "Score": m["precision"].get(f"@{k}", 0)})
                
                ##TODO Need to test this row, not wrking           
                rows.append({"K": f"@{k}", "Metric": "Recall",
                             "Score": m["recall"].get(f"@{k}", 0)})

                rows.append({"K": f"@{k}", "Metric": "NDCG",
                             "Score": m["ndcg"].get(f"@{k}", 0)})
            chart_df = pd.DataFrame(rows)
            fig = px.bar(
                chart_df, x="K", y="Score", color="Metric",
                barmode="group", template=template,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Source: `results/retrieval_eval_results.json` - "
                f"{retrieval_j.get('num_queries', '?')} queries, "
                f"{retrieval_j.get('elapsed_seconds', '?'):.2f}s elapsed."
            )
        else:
            st.info("retrieval metrics not available....")

    #Tab 2: generation quality
    with tab2:
        st.subheader("Report Generation Quality - real eval (n=5)")
        if generation_j:
            gm = generation_j.get("generation_metrics", {})
            rq = generation_j.get("report_quality", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("BLEU", f"{gm.get('bleu', 0):.4f}")
            with col2:
                st.metric("ROUGE-L", f"{gm.get('rouge_l', 0):.4f}")
            with col3:
                st.metric(
                    "Completeness",
                    f"{rq.get('avg_completeness', 0):.2f}",
                    help="Fraction of required fields populated in the report.",
                )
            st.caption(
                f"Model: `{generation_j.get('model', '?')}` on "
                f"{generation_j.get('device', '?')} . "

                f"{generation_j.get('num_samples', '?')} samples . "
                f"{generation_j.get('total_time_s', 0):.0f}s total."
            )
        else:
            st.info("Generation metrics not available........")

    #Tab 3: Hallucination
    with tab3:
        st.subheader("Hallucination Analysis — real eval")
        if generation_j and "hallucination" in generation_j:
            hh = generation_j["hallucination"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Faithfulness", f"{hh.get('avg_faithfulness', 0):.4f}")
            with col2:
                st.metric("Hallucination Rate",
                          f"{hh.get('hallucination_rate', 0):.4f}")
            pct = float(hh.get("avg_faithfulness", 0)) * 100
            st.progress(pct / 100, text=f"{pct:.1f}% of claims attributed to retrieved sources")
            st.info(
                f"Of {generation_j.get('num_samples', '?')} evaluated incidents, "
                f"{(1 - hh.get('hallucination_rate', 0)) * 100:.1f}% of generated "
                f"claims could be attributed to retrieved source documents."
            )

            st.caption(
                "Method: SBERT cosine-similarity > 0.5 between generated claim "
                "and any retrieved chunk (sentence_transformers `all-MiniLM-L6-v2`)."
            )
        else:
            st.info("Hallucination metrics not available.")

    #Tab 4: Per-sample
    with tab4:
        st.subheader("Per-Sample Eval Detail - SmolLM2-1.7B-Instruct on CPU")
        if generation_j and "per_sample" in generation_j:
            ps = generation_j["per_sample"]
            df = pd.DataFrame([ {
                    "Incident ID": s.get("incident_id", "?"), "Type": s.get("incident_type", "?"),
                    "Severity (DB)": s.get("severity", "?"),
                    "Severity (Report)": s.get("report_severity", "?"), "Retrieved": s.get("retrieved_count", 0),
                    "Gen Time (s)": round(s.get("gen_time_s", 0), 1),"Completeness": s.get("completeness", 0),
                    "Conf (%)": s.get("report_confidence", 0),   "Root Cause": (s.get("report_root_cause") or "")[:80],
                }
                for s in ps
            ])
            try:
                styled_df = df.style.background_gradient(subset=["Conf (%)"], cmap="RdYlGn")
            except Exception:
                styled_df = df
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("#### No per-sample data available.")



# Main

def main() -> None:
    """main entry point for the Streamlit dashboard"""
    page = render_sidebar()

    if page == "🏠 Overview":
        render_overview()
    elif page == "🔬 Investigation":
        render_investigation()
    elif page == "📊 Reports":
        render_reports()
    elif page == "📈 Evaluation":
        render_evaluation()


if __name__ == "__main__":
    main()
