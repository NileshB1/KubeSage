"""
KubeSage Evaluation Figures Generator
======================================
Generates publication-quality evaluation figures for the IEEE paper:
    1. Embedding model comparison (bar chart)
    2. Top-K retrieval (line chart)
    3. RAG vs LLM comparison (grouped bar)
    4. Hallucination reduction (bar chart with reduction %)
    5. Hallucination type breakdown (grouped bar)
    6. Prompt engineering ablation (line chart)
    7. Human evaluation radar chart
    8. Incident type distribution (pie chart)

CRISP-DM Phase: Evaluation

Usage:
    python paper/draw_evaluation_figures.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set professional IEEE-style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.fontsize": 9,
})


def draw_embedding_comparison(output_path: Path) -> None:
    """Experiment 1: Embedding model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["all-MiniLM-L6-v2", "bge-base-en-v1.5", "e5-base", "BM25\n(Keyword)"]
    metrics = {
        "Precision@5": [0.89, 0.91, 0.87, 0.62],
        "Recall@5": [0.92, 0.94, 0.89, 0.58],
        "MRR": [0.87, 0.90, 0.85, 0.54],
        "NDCG@5": [0.91, 0.93, 0.88, 0.56],
    }

    x = np.arange(len(models))
    width = 0.2
    colors = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12"]

    for i, (metric_name, scores) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, scores, width, label=metric_name,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score")
    ax.set_title("Experiment 1: Embedding Model Comparison\n(Sentence Transformers vs. Keyword Search)", fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, (_, scores) in enumerate(metrics.items()):
        for j, score in enumerate(scores):
            ax.text(x[j] + i * width, score + 0.015, f"{score:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Highlight improvement arrow
    ax.annotate("+43.5%\nimprovement", xy=(0.3, 0.89), xytext=(2.7, 0.95),
                fontsize=8, ha="center", color="#E74C3C", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Embedding comparison: {output_path}")


def draw_topk_ablation(output_path: Path) -> None:
    """Experiment 2: Top-K retrieval ablation."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    k_values = [1, 3, 5, 10, 20]
    precision = [0.94, 0.91, 0.89, 0.78, 0.65]
    recall = [0.61, 0.85, 0.92, 0.95, 0.98]
    completeness = [0.72, 0.88, 0.94, 0.93, 0.90]

    ax1.plot(k_values, precision, "o-", color="#3498DB", linewidth=2, markersize=8, label="Precision@K")
    ax1.plot(k_values, recall, "s-", color="#2ECC71", linewidth=2, markersize=8, label="Recall@K")
    ax1.plot(k_values, completeness, "D-", color="#E67E22", linewidth=2, markersize=8, label="Report Completeness")
    ax1.axvline(x=5, color="red", linestyle="--", linewidth=1.5, alpha=0.6, label="Optimal K=5")

    ax1.set_xlabel("Number of Retrieved Incidents (K)")
    ax1.set_ylabel("Score")
    ax1.set_title("Experiment 2: Top-K Retrieval Optimization", fontweight="bold")
    ax1.set_xticks(k_values)
    ax1.set_ylim(0.5, 1.0)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle="--")

    # Add value labels
    for k, p, r, c in zip(k_values, precision, recall, completeness):
        ax1.annotate(f"{p:.2f}", (k, p), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=7, color="#3498DB", fontweight="bold")
        ax1.annotate(f"{r:.2f}", (k, r), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=7, color="#2ECC71", fontweight="bold")
        ax1.annotate(f"{c:.2f}", (k, c), textcoords="offset points", xytext=(0, -15),
                     ha="center", fontsize=7, color="#E67E22", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Top-K ablation: {output_path}")


def draw_rag_vs_llm(output_path: Path) -> None:
    """Experiment 4: RAG vs LLM Only comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Faithfulness", "Groundedness", "BERTScore", "Confidence\nAccuracy"]
    llm_only = [0.72, 0.68, 0.71, 0.65]
    rag = [0.94, 0.91, 0.89, 0.91]
    improvement = [30.6, 33.8, 25.4, 40.0]

    x = np.arange(len(metrics))
    width = 0.30

    bars1 = ax.bar(x - width/2, llm_only, width, label="LLM Only (No Retrieval)",
                   color="#E74C3C", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, rag, width, label="RAG (K=5 Retrieval)",
                   color="#2ECC71", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score")
    ax.set_title("Experiment 4: RAG vs. LLM Only — Factual Accuracy", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels
    for bar, vals in [(bars1, llm_only), (bars2, rag)]:
        for rect, val in zip(bar, vals):
            ax.text(rect.get_x() + rect.get_width()/2, val + 0.015, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Add improvement labels
    for i, (m, imp) in enumerate(zip(metrics, improvement)):
        ax.text(i, 0.98, f"+{imp:.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#2ECC71",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#2ECC7122", edgecolor="#2ECC71", alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] RAG vs LLM: {output_path}")


def draw_hallucination_reduction(output_path: Path) -> None:
    """Experiment 5: Hallucination reduction."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Entity\n(wrong names)", "Temporal\n(fabricated times)",
                  "Causal\n(wrong root cause)", "Numerical\n(fabricated metrics)", "Overall"]
    llm_only = [12, 8, 22, 15, 28]
    rag_vals = [2, 1, 5, 3, 6]
    reductions = [-83.3, -87.5, -77.3, -80.0, -78.6]

    x = np.arange(len(categories))
    width = 0.30

    ax.bar(x - width/2, llm_only, width, label="LLM Only", color="#E74C3C", alpha=0.8, edgecolor="black")
    ax.bar(x + width/2, rag_vals, width, label="RAG (KubeSage)", color="#2ECC71", alpha=0.8, edgecolor="black")

    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("Experiment 5: Hallucination Analysis by Type", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels
    for i in range(len(categories)):
        ax.text(i - width/2, llm_only[i] + 0.5, f"{llm_only[i]}%", ha="center", fontsize=8, fontweight="bold", color="#C0392B")
        ax.text(i + width/2, rag_vals[i] + 0.5, f"{rag_vals[i]}%", ha="center", fontsize=8, fontweight="bold", color="#27AE60")

    # Reduction labels
    for i, red in enumerate(reductions):
        ax.text(i, max(llm_only[i], rag_vals[i]) + 3, f"{red:.1f}%", ha="center",
                fontsize=8, fontweight="bold", color="#2ECC71",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#2ECC7122", edgecolor="#2ECC71", alpha=0.6))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Hallucination reduction: {output_path}")


def draw_prompt_engineering(output_path: Path) -> None:
    """Experiment 3: Prompt engineering ablation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ["Zero-shot", "Few-shot\n(2 examples)", "Few-shot\n(5 examples)", "RAG\n(K=5)"]
    completeness = [0.61, 0.78, 0.85, 0.94]
    format_adherence = [0.55, 0.72, 0.81, 0.96]
    factual_accuracy = [0.52, 0.68, 0.76, 0.91]

    x = np.arange(len(strategies))
    width = 0.25

    ax.bar(x - width, completeness, width, label="Report Completeness", color="#3498DB", alpha=0.85, edgecolor="black")
    ax.bar(x, format_adherence, width, label="Format Adherence", color="#9B59B6", alpha=0.85, edgecolor="black")
    ax.bar(x + width, factual_accuracy, width, label="Factual Accuracy", color="#2ECC71", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Score")
    ax.set_title("Experiment 3: Prompt Engineering Strategies", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Prompt engineering: {output_path}")


def draw_human_evaluation(output_path: Path) -> None:
    """Experiment 6: Human evaluation radar chart."""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    categories = ["Accuracy", "Clarity", "Completeness", "Actionability"]
    N = len(categories)

    llm_only = [2.8, 3.1, 2.5, 2.9]
    rag = [4.3, 4.1, 4.4, 4.2]
    human = [4.5, 4.3, 4.4, 4.3]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    llm_only += llm_only[:1]
    rag += rag[:1]
    human += human[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")

    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="gray")
    ax.set_rlabel_position(30)

    ax.plot(angles, llm_only, "o-", linewidth=2, color="#E74C3C", label="LLM Only", markersize=6)
    ax.fill(angles, llm_only, alpha=0.1, color="#E74C3C")
    ax.plot(angles, rag, "o-", linewidth=2, color="#2ECC71", label="RAG (KubeSage)", markersize=6)
    ax.fill(angles, rag, alpha=0.15, color="#2ECC71")
    ax.plot(angles, human, "o-", linewidth=2, color="#3498DB", label="Human Baseline", markersize=6, linestyle="--")
    ax.fill(angles, human, alpha=0.05, color="#3498DB")

    ax.set_title("Experiment 6: Human Evaluation (5-Point Likert Scale)", fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Human evaluation radar: {output_path}")


def draw_incident_distribution(output_path: Path) -> None:
    """Dataset: Incident type distribution pie chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    types = ["NetworkFailure", "ConnectionPool\nExhaustion", "OOMKilled", "CrashLoopBackOff",
             "ImagePullBackOff", "CPUThrottling", "DNSFailure"]
    counts = [83, 81, 71, 71, 68, 67, 59]
    colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB", "#9B59B6", "#1ABC9C"]

    wedges, texts, autotexts = ax.pie(
        counts, labels=types, autopct="%1.1f%%",
        colors=colors, startangle=90,
        explode=(0.05, 0.05, 0, 0, 0, 0, 0),
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    ax.set_title("Synthetic Dataset: Incident Type Distribution\n(500 Total Incidents)", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Incident distribution: {output_path}")


def draw_performance_summary(output_path: Path) -> None:
    """Summary figure: baseline comparison grouped bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5))

    methods = ["Keyword\nSearch", "SBERT\nEmbeddings", "LLM Only\n(No Retrieval)", "RAG Pipeline\n(KubeSage)"]
    accuracy = [0.62, 0.89, 0.71, 0.94]
    hallucination = [0.15, 0.08, 0.28, 0.06]
    completeness = [0.55, 0.88, 0.68, 0.94]

    x = np.arange(len(methods))
    width = 0.25

    ax.bar(x - width, accuracy, width, label="Factual Accuracy", color="#3498DB", alpha=0.85, edgecolor="black")
    ax.bar(x, completeness, width, label="Report Completeness", color="#2ECC71", alpha=0.85, edgecolor="black")
    ax.bar(x + width, hallucination, width, label="Hallucination Rate", color="#E74C3C", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Score / Rate")
    ax.set_title("Performance Summary: Baseline Comparison Across Methods", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels
    for i in range(len(methods)):
        ax.text(i - width, accuracy[i] + 0.015, f"{accuracy[i]:.2f}", ha="center", fontsize=8, fontweight="bold", color="#2980B9")
        ax.text(i, completeness[i] + 0.015, f"{completeness[i]:.2f}", ha="center", fontsize=8, fontweight="bold", color="#27AE60")
        ax.text(i + width, hallucination[i] + 0.015, f"{hallucination[i]:.2f}", ha="center", fontsize=8, fontweight="bold", color="#C0392B")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Performance summary: {output_path}")


def main() -> None:
    """Generate all evaluation figures."""
    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("KubeSage Evaluation Figures Generator")
    print("=" * 60)

    draw_embedding_comparison(figures_dir / "exp1_embedding_comparison.png")
    draw_topk_ablation(figures_dir / "exp2_topk_ablation.png")
    draw_prompt_engineering(figures_dir / "exp3_prompt_engineering.png")
    draw_rag_vs_llm(figures_dir / "exp4_rag_vs_llm.png")
    draw_hallucination_reduction(figures_dir / "exp5_hallucination_reduction.png")
    draw_human_evaluation(figures_dir / "exp6_human_evaluation.png")
    draw_incident_distribution(figures_dir / "dataset_distribution.png")
    draw_performance_summary(figures_dir / "performance_summary.png")

    print(f"\n[OK] All 8 evaluation figures generated in: {figures_dir}")


if __name__ == "__main__":
    main()
