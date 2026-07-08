"""
KubeSage Architecture & Pipeline Diagram Generator
===================================================
Generates publication-quality architecture diagrams for the IEEE paper.
Uses matplotlib and networkx for programmatic, reproducible figures.

CRISP-DM Phase: Business Understanding (Architecture Design)
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set professional style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def draw_system_architecture(output_path: Path) -> None:
    """
    Draw the complete KubeSage system architecture diagram.
    Shows data sources → preprocessing → embeddings → vector DB →
    semantic search → RAG → LLM → report output.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_title("KubeSage System Architecture", fontsize=16, fontweight="bold", pad=20)

    # Color scheme
    colors = {
        "data": "#4ECDC4",       # Teal
        "preprocess": "#FF6B6B",  # Coral
        "embedding": "#45B7D1",   # Sky blue
        "vector_db": "#96CEB4",   # Sage
        "search": "#FFEAA7",      # Yellow
        "prompt": "#DDA0DD",      # Plum
        "llm": "#98D8C8",         # Mint
        "output": "#F7DC6F",      # Gold
    }

    boxes = [
        # (x, y, w, h, label, color, level)
        (2.5, 15.5, 5, 0.8, "Data Sources\nPrometheus • Grafana • K8s Events • Logs", colors["data"], 0),
        (2.5, 13.2, 5, 0.8, "Preprocessing Pipeline\nLog Parsing → Normalization → Feature Engineering", colors["preprocess"], 1),
        (2.5, 10.8, 5, 0.8, "Sentence Transformer Embeddings\nall-MiniLM-L6-v2 / bge-base-en-v1.5", colors["embedding"], 2),
        (2.5, 8.4, 5, 0.8, "Vector Database (ChromaDB)\nCosine Similarity • HNSW Index", colors["vector_db"], 3),
        (2.5, 6.0, 5, 0.8, "Semantic Search\nQuery Embedding → Top-K Similar Incidents", colors["search"], 4),
        (2.5, 3.6, 5, 0.8, "Prompt Construction (RAG)\nSystem Prompt + Incident + Retrieved + KB", colors["prompt"], 5),
        (2.5, 1.2, 5, 0.8, "LLM Generation\nLlama 3 / Mistral / Gemma", colors["llm"], 6),
        (2.5, -1.2, 5, 0.8, "Structured Incident Report\nID • Severity • Root Cause • Evidence • Timeline", colors["output"], 7),
    ]

    # Draw boxes and arrows
    for i, (x, y, w, h, label, color, level) in enumerate(boxes):
        # Draw rounded box
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(box)

        # Add label text
        ax.text(
            x + w / 2, y + h / 2,
            label,
            ha="center", va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=4,
        )

        # Draw downward arrow (except last box)
        if i < len(boxes) - 1:
            next_y = boxes[i + 1][1]
            arrow = FancyArrowPatch(
                (x + w / 2, y),
                (x + w / 2, next_y + h),
                arrowstyle="->,head_length=0.3,head_width=0.2",
                color="black",
                linewidth=2,
                zorder=2,
            )
            ax.add_patch(arrow)

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors["data"], label="Data Ingestion"),
        mpatches.Patch(color=colors["preprocess"], label="Preprocessing"),
        mpatches.Patch(color=colors["embedding"], label="Deep Learning (Embeddings)"),
        mpatches.Patch(color=colors["vector_db"], label="Vector Database"),
        mpatches.Patch(color=colors["search"], label="Semantic Retrieval"),
        mpatches.Patch(color=colors["prompt"], label="RAG Prompt Engineering"),
        mpatches.Patch(color=colors["llm"], label="Generative AI (LLM)"),
        mpatches.Patch(color=colors["output"], label="Output Report"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Architecture diagram saved: {output_path}")


def draw_crisp_dm_workflow(output_path: Path) -> None:
    """
    Draw the CRISP-DM methodology workflow diagram showing
    the 6 phases and their KubeSage-specific mappings.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("CRISP-DM Methodology — KubeSage Mapping", fontsize=16, fontweight="bold", pad=20)

    phases = [
        ("Business\nUnderstanding", "#E74C3C", "Automate K8s incident\ninvestigation & reporting"),
        ("Data\nUnderstanding", "#E67E22", "Explore K8s incident\ndatasets, logs, metrics"),
        ("Data\nPreparation", "#F1C40F", "Preprocess logs,\nnormalize, feature eng."),
        ("Modeling", "#2ECC71", "Embeddings → ChromaDB\n→ RAG → LLM pipeline"),
        ("Evaluation", "#3498DB", "Precision@K, BLEU, ROUGE\nFaithfulness, Human eval"),
        ("Deployment", "#9B59B6", "FastAPI + Streamlit\n+ Docker + PostgreSQL"),
    ]

    # Draw circular workflow
    r = 2.8
    center_x, center_y = 6, 4
    n = len(phases)

    for i, (name, color, mapping) in enumerate(phases):
        angle = 2 * np.pi * i / n - np.pi / 2
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)

        # Phase box
        box = FancyBboxPatch(
            (x - 0.9, y - 0.5), 1.8, 1.0,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(box)

        # Phase name
        ax.text(x, y + 0.15, name, ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)
        # Mapping text
        ax.text(x, y - 0.35, mapping, ha="center", va="top", fontsize=5.5, color="white", style="italic", zorder=4)

    # Connecting arrows (curved)
    for i in range(n):
        angle1 = 2 * np.pi * i / n - np.pi / 2
        angle2 = 2 * np.pi * (i + 1) / n - np.pi / 2
        # Slightly offset from box edges
        r_inner = r - 1.0
        r_outer = r + 1.0
        x1 = center_x + r_inner * np.cos(angle1)
        y1 = center_y + r_inner * np.sin(angle1)
        x2 = center_x + r_outer * np.cos(angle2)
        y2 = center_y + r_outer * np.sin(angle2)

        # Draw arrow between phases
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                linewidth=2,
                connectionstyle="arc3,rad=0.4",
                alpha=0.7,
            ),
            zorder=1,
        )

    # Center: CRISP-DM
    ax.text(center_x, center_y, "CRISP-DM", ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50")
    ax.text(center_x, center_y - 0.4, "Iterative", ha="center", va="center", fontsize=9, color="#7F8C8D", style="italic")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] CRISP-DM workflow saved: {output_path}")


def draw_rag_pipeline(output_path: Path) -> None:
    """
    Draw the detailed RAG pipeline: Retrieve → Augment → Generate.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Retrieval-Augmented Generation (RAG) Pipeline", fontsize=16, fontweight="bold", pad=20)

    # Three main phases
    phases_data = [
        ("1. RETRIEVE", 1.5, "#3498DB",
         "Query Embedding\n↓\nCosine Similarity Search\n↓\nTop-K Incidents from ChromaDB"),
        ("2. AUGMENT", 6.5, "#E67E22",
         "System Prompt:\n'You are an SRE...'\n↓\nCurrent Incident Data\n↓\nRetrieved Similar Incidents\n↓\nKnowledge Base Context"),
        ("3. GENERATE", 11.5, "#2ECC71",
         "LLM (Llama 3/Mistral)\n↓\nStructured Report\n↓\nPost-processing\n↓\nConfidence Scoring"),
    ]

    for label, x, color, content in phases_data:
        # Background box
        box = FancyBboxPatch(
            (x - 1.8, 0.5), 3.6, 5.0,
            boxstyle="round,pad=0.2",
            facecolor=color,
            edgecolor="black",
            linewidth=2,
            alpha=0.2,
            zorder=1,
        )
        ax.add_patch(box)

        # Title
        ax.text(x, 5.2, label, ha="center", va="center", fontsize=12, fontweight="bold", color=color, zorder=3)
        # Content
        ax.text(x, 2.8, content, ha="center", va="center", fontsize=9, color="black", zorder=3)

        # Arrow to next phase
        if x < 10:
            ax.annotate(
                "", xy=(x + 2.2, 3.0), xytext=(x + 1.4, 3.0),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=3),
                zorder=4,
            )

    # Vector DB icon
    ax.text(4.0, 0.3, "🔍 ChromaDB Vector Store", ha="center", fontsize=8, color="#7F8C8D", style="italic")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] RAG pipeline diagram saved: {output_path}")


def main():
    """Generate all architecture diagrams."""
    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    draw_system_architecture(figures_dir / "system_architecture.png")
    draw_crisp_dm_workflow(figures_dir / "crisp_dm_workflow.png")
    draw_rag_pipeline(figures_dir / "rag_pipeline.png")

    print(f"\n[OK] All diagrams generated in: {figures_dir}")


if __name__ == "__main__":
    main()
