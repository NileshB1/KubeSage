"""

Executes evaluation runs comparing different configuration setups
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.config import settings
from backend.logging_config import get_logger
from evaluation.metrics import (RetrievalMetrics, ClassificationMetrics,
    GenerationMetrics, HallucinationMetrics)

logger = get_logger(__name__)


class ExperimentRunner:
    """
    Runs evaluation experiments for the KubeSage project
    """

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or settings.RESULTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    # Experiment1: embedding model comparison


    def experiment_1_embedding_comparison(self, queries: list[dict[str, Any]],
    ):
        """Compare retrieval quality across embedding models
        """
        logger.info("Running exp1: embedding model comparison")
        start = time.time()

        k_values = [1, 3, 5, 10, 20]
        results = RetrievalMetrics.compute_all(queries, k_values=k_values)

        elapsed = time.time()-start
        logger.info(f"exp complete in {elapsed:.1f}s")

        return {"experiment": "embedding_comparison",  "metrics": results,
            "elapsed_seconds": round(elapsed, 2)  }

 
    # Exp2 : Top-K Retrieval (RQ1)


    def experiment_2_topk_ablation(
        self, queries: list[dict[str, Any]]):
        """
        Evaluate how Top-K affects retrieval and generation quality
        """
        logger.info(f"Running Experiment 2: Top-K Retrieval Ablation")
        start = time.time()

        k_values = [1, 3, 5, 10, 20]
        results = RetrievalMetrics.compute_all(queries, k_values=k_values)

        elapsed = time.time() - start
        logger.info(f"Experiment 2 complete in {elapsed:.1f}s")

        return {
            "experiment": "topk_ablation", "k_values": k_values,"metrics": results,"elapsed_seconds": round(elapsed, 2)
        }


    # Experiment3: prompt engineering

    def experiment_3_prompt_engineering(self,  strategies: dict[str, list[dict[str, Any]]]):
        """
        Compare prompt strategies: Zero-shot, Few-shot, RAG
        """
        logger.info(f"Running Experiment 3: prompt engineering")
        start = time.time()

        comparison: dict[str, dict[str, float]] = {}

        for strategy_name, results_list in strategies.items():
            if not results_list:
                continue

            #aggregate completion and accuracy scores
            completeness = [r.get("completeness", 0) for r in results_list]
            accuracy = [r.get("accuracy", 0) for r in results_list]

            comparison[strategy_name] = {"avg_completeness": round(float(np.mean(completeness)), 4),
                "avg_accuracy": round(float(np.mean(accuracy)), 4)}

        elapsed = time.time() - start
        logger.info(f"Experiment 3 finally completed in {elapsed:.1f}s .....")

        return {
            "experiment": "prompt_engineering", "strategies": comparison,
            "elapsed_seconds": round(elapsed, 2),
        }


    # exp 4: RAG vs LLM Only 

    def experiment_4_rag_vs_llm(self,  rag_results: dict[str, Any],
        llm_only_results: dict[str, Any]):
        """
        Compare RAG pipeline vs standalone LLM
        """
        logger.info(f"Running Exp 4: RAG vs LLM Only")
        start = time.time()

        comparison = {
            "rag": {"avg_faithfulness": rag_results.get("faithfulness", 0),
                "avg_confidence": rag_results.get("confidence", 0) },
            
            "llm_only": { "avg_faithfulness": llm_only_results.get("faithfulness", 0),
                "avg_confidence": llm_only_results.get("confidence", 0) },
            
            "improvement": {
                "faithfulness_delta": round(rag_results.get("faithfulness", 0) - llm_only_results.get("faithfulness", 0), 4),
                "confidence_delta": round( rag_results.get("confidence", 0) - llm_only_results.get("confidence", 0), 4)},
        }

        elapsed = time.time() - start
        logger.info(f"Experiment 4 completed in {elapsed:.1f}s .....")

        return {"experiment": "rag_vs_llm", "comparison": comparison, "elapsed_seconds": round(elapsed, 2)  }


    # Experiment 5: Hallucination Analysis

    def experiment_5_hallucination_analysis(
        self,   generated_texts: list[str],  source_documents: list[list[str]],
        labels: list[str] | None = None):
        """
        Quantify hallucination rates in generated reports
        """
        logger.info(f"Running Experiment 5: hallucination analysis")
        start = time.time()

        faithfulness_scores=[]
        per_text_details=[]

        for i, (gen_text, sources) in enumerate(zip(generated_texts, source_documents)):
            metrics=HallucinationMetrics.compute_faithfulness(gen_text, sources)
            faithfulness_scores.append(metrics["faithfulness"])
            per_text_details.append({"text_index": i, "label": labels[i] if labels else None,
                **metrics})

        avg_faithfulness=float(np.mean(faithfulness_scores))
        std_faithfulness=float(np.std(faithfulness_scores))

        hallucination_rate = 1.0-avg_faithfulness

        elapsed = time.time() - start
        logger.info(f"Experiment 5 complete in {elapsed:.1f}s (avg faithfulness: {avg_faithfulness:.3f})")

        return {
            "experiment": "hallucination_analysis","avg_faithfulness": round(avg_faithfulness, 4),
            "std_faithfulness": round(std_faithfulness, 4),

            "hallucination_rate": round(hallucination_rate, 4),"num_samples": len(generated_texts),
            "per_text_details": per_text_details,"elapsed_seconds": round(elapsed, 2),
        }

 
    #experiment 6: report quality evaluation

    def experiment_6_report_quality(
        self,  candidates: list[str], references: list[list[str]], human_scores: list[dict[str, float]]):
        """
        Evaluate generated report quality using BLEU, ROUGE, and human
        evaluation
        """
        logger.info(f"Running Experiment 6: Report Quality Evaluation")
        start = time.time()

        gen_metrics = GenerationMetrics.compute_all(candidates, references)

        results: dict[str, Any] = {"experiment": "report_quality", "generation_metrics": gen_metrics}

        if human_scores:
            #aggregate human evaluation
            clarity = [s.get("clarity", 0) for s in human_scores]
            accuracy = [s.get("accuracy", 0) for s in human_scores]
            completeness = [s.get("completeness", 0) for s in human_scores]
            actionability = [s.get("actionability", 0) for s in human_scores]

            results["human_evaluation"] = {
                "avg_clarity": round(float(np.mean(clarity)), 2),
                "avg_accuracy": round(float(np.mean(accuracy)), 2),

                "avg_completeness": round(float(np.mean(completeness)), 2),
                "avg_actionability": round(float(np.mean(actionability)), 2),
                "num_evaluators": len(human_scores),
            }

        elapsed = time.time() - start
        results["elapsed_seconds"]=round(elapsed, 2)

        logger.info(f"experiment 6 complete in {elapsed:.1f}s .....")

        return results

    # Save Results

    def save_results(self, results: dict[str, Any], filename: str) -> Path:
        """Save experiment results to JSON"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {output_path}")
        return output_path



# CLI

def main() -> None:
    """Run specific experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run KubeSage experiments")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3, 4, 5, 6], required=True)
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()

    runner = ExperimentRunner(output_dir=args.output)

    #demo with synthetic data
    dummy_queries = [{"retrieved_ids": [f"INC-{i}" for i in range(1200, 1210)], "relevant_ids": {f"INC-{1200+i}" for i in range(0, 7)}},
        {"retrieved_ids": [f"INC-{i}" for i in range(1300, 1310)], "relevant_ids": {f"INC-{1300+i}" for i in range(0, 5)}} ]

    if args.experiment == 1:
        results = runner.experiment_1_embedding_comparison(dummy_queries)
    elif args.experiment == 2:
        results = runner.experiment_2_topk_ablation(dummy_queries)
    elif args.experiment == 3:
        strat = {
            "zero_shot": [{"completeness": 0.6, "accuracy": 0.55}], "few_shot": [{"completeness": 0.78, "accuracy": 0.72}],
            "rag": [{"completeness": 0.94, "accuracy": 0.91}] }
        results = runner.experiment_3_prompt_engineering(strat)
    elif args.experiment == 4:
        results = runner.experiment_4_rag_vs_llm({"faithfulness": 0.94, "confidence": 0.91},
            {"faithfulness": 0.72, "confidence": 0.65})
    elif args.experiment == 5:
        results = runner.experiment_5_hallucination_analysis(
            generated_texts=["The pod OOMKilled", "Connection pool exhausted by payment service"],
            source_documents=[["pod oom killed memory"], ["db connection pool full"]],
        )
    elif args.experiment == 6:
        results = runner.experiment_6_report_quality(
            candidates=["Generated report 1", "Generated report 2"],
            references=[["Reference 1a", "Reference 1b"], ["Reference 2"]],
        )

    output_path = runner.save_results(results, f"exp{args.experiment}_results.json")
    print(f"\n experiment {args.experiment} results saved: {output_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "per_text_details"}, indent=2))


if __name__ == "__main__":
    main()
