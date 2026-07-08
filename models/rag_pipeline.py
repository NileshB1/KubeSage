"""
KubeSage RAG Pipeline
=====================
Orchestrates the complete Retrieval-Augmented Generation flow.
Queries the vector database for matching incidents, constructs prompt 
templates, runs inference via the LLM wrapper, and structures the final 
JSON post-mortem report.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on the path for cross-module imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from backend.config import settings
from backend.logging_config import get_logger
from embeddings.generate_embeddings import EmbeddingGenerator
from vector_db.build_index import VectorDatabase


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced Kubernetes Site Reliability Engineer (SRE) with deep expertise in debugging production incidents.

Your task is to investigate the current incident and generate a structured investigation report.

CRITICAL RULES:
1. ONLY use information from the provided Retrieved Similar Incidents and Knowledge Base.
2. DO NOT invent facts, metrics, or timelines that are not in the provided evidence.
3. If evidence is insufficient to determine the root cause, state: "Insufficient evidence — additional investigation required."
4. Provide confidence scores that reflect the strength of the evidence.
5. Structure your response as a machine-readable JSON report."""

USER_PROMPT_TEMPLATE = """## Current Incident

{incident_description}

## Retrieved Similar Incidents

{retrieved_incidents}

## Knowledge Base

{knowledge_base}

## Instructions

Analyze the current incident by comparing it with the retrieved similar incidents. 
Output ONLY a JSON object with these keys:
incident_id (string), severity (Critical/High/Medium/Low), root_cause (string),
evidence (list of strings), affected_services (list of strings),
confidence_score (number 0-100), recommended_fixes (list of strings),
generated_summary (2-3 sentence string).
Do not include any text outside the JSON."""


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Constructs RAG prompts from incident data and retrieved context.

    Combines:
        - System prompt (role definition)
        - Current incident description
        - Retrieved similar incidents
        - Domain knowledge base
    """

    @staticmethod
    def format_retrieved_incidents(
        search_results: dict[str, Any],
        max_incidents: int = 5,
    ) -> str:
        """
        Format retrieved incidents into a readable string for the prompt.

        Args:
            search_results: Search results from VectorDatabase.
            max_incidents: Maximum number of incidents to include.

        Returns:
            Formatted string with incident details.
        """
        results = search_results.get("results", [])[:max_incidents]

        if not results:
            return "No similar incidents found in the knowledge base."

        formatted: list[str] = []
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            formatted.append(
                f"--- Incident {i} (Similarity: {result.get('similarity_score', 0):.2f}) ---\n"
                f"ID: {result.get('incident_id', 'N/A')}\n"
                f"Type: {meta.get('incident_type', 'N/A')}\n"
                f"Severity: {meta.get('severity', 'N/A')}\n"
                f"Root Cause: {meta.get('root_cause', 'N/A')}\n"
                f"Resolution: {meta.get('resolution', 'N/A')}\n"
                f"Affected Services: {meta.get('affected_services', 'N/A')}\n"
            )

        return "\n".join(formatted)

    @staticmethod
    def get_knowledge_base() -> str:
        """
        Return domain-specific knowledge base about K8s incidents.

        Returns:
            Knowledge base string.
        """
        return """
Kubernetes Incident Categories:
- OOMKilled: Pod terminated due to memory limit exceeded. Check memory requests/limits, memory leaks.
- CrashLoopBackOff: Container crashes repeatedly on startup. Check entrypoint, environment variables, dependencies.
- ImagePullBackOff: Cannot pull container image. Check registry credentials, image tag, registry availability.
- ConnectionPoolExhaustion: Database connection pool saturated. Check connection leaks, pool size, query performance.
- DNSFailure: DNS resolution issues. Check CoreDNS, NetworkPolicy, service discovery.
- CPUThrottling: Container CPU usage exceeds limit. Check CPU requests/limits, code optimization, HPA.
- NetworkFailure: Network connectivity issues. Check NetworkPolicy, service mesh, firewall rules.

Common Kubernetes Troubleshooting Commands:
- kubectl describe pod <pod> - Get detailed pod information
- kubectl logs <pod> --previous - Get logs from previous container instance
- kubectl get events --sort-by=.metadata.creationTimestamp - Get cluster events
- kubectl top pod <pod> - Get resource usage
"""

    def build_prompt(
        self,
        incident_description: str,
        retrieved_incidents: dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, str]:
        """
        Build the full RAG prompt.

        Args:
            incident_description: Current incident description.
            retrieved_incidents: Search results from VectorDatabase.
            top_k: Number of incidents to include.

        Returns:
            Dictionary with 'system' and 'user' prompt strings.
        """
        formatted_incidents = self.format_retrieved_incidents(
            retrieved_incidents,
            max_incidents=top_k,
        )
        knowledge_base = self.get_knowledge_base()

        user_prompt = USER_PROMPT_TEMPLATE.format(
            incident_description=incident_description,
            retrieved_incidents=formatted_incidents,
            knowledge_base=knowledge_base,
        )

        return {
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
        }


# ---------------------------------------------------------------------------
# Report Parser
# ---------------------------------------------------------------------------

class ReportParser:
    """
    Parses LLM output into a structured incident report.
    Handles JSON extraction, validation, and post-processing.
    """

    REQUIRED_FIELDS = [
        "incident_id", "severity", "root_cause", "evidence",
        "affected_services", "confidence_score", "recommended_fixes",
        "generated_summary",
    ]

    @staticmethod
    def _repair_json(json_str: str) -> dict[str, Any]:
        """
        Progressively repair common JSON failure modes in small LLM outputs.

        Non-destructive: each strategy starts from the original json_str to
        avoid cascading regex errors.

        Targeted Failures:
            1. Trailing commas before } or ]
            2. Single quotes instead of double quotes
            3. Missing closing braces/brackets
            4. Extra text after the closing }
            5. Unquoted keys (e.g., {incident_id: "foo"})
            6. Double-quoted keys but single-quoted values
            7. Nested markdown backticks or explanatory text
        """
        # --- Repair step functions -------------------------------------------------
        def _step_strip_extra_text(text: str) -> str:
            """Remove markdown fences and truncate after final closing brace."""
            # Case 7: remove backtick fences
            text = re.sub(r'```(?:json)?\s*', '', text)
            text = text.replace('```', '')
            # Case 4: drop everything after the last }
            last_brace = text.rfind('}')
            if last_brace != -1:
                text = text[:last_brace + 1]
            return text.strip()

        def _step_remove_trailing_commas(text: str) -> str:
            """Case 1: strip trailing commas before } or ]."""
            return re.sub(r',\s*([}\]])', r'\1', text)

        def _step_quote_unquoted_keys(text: str) -> str:
            """Case 5: quote bare identifiers used as JSON keys."""
            return re.sub(
                r'([\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
                r'\1 "\2":',
                text,
            )

        def _step_fix_single_quotes(text: str) -> str:
            """Cases 2 & 6: replace single-quoted strings with double-quoted.

            Captures surrounding JSON delimiters to avoid variable-width
            look-behind (unsupported by Python's re).  Only matches quotes
            in JSON structural positions (near : { , [ } ]).

            Uses look-ahead for trailing delimiter so consecutive values
            in arrays (['a', 'b', 'c']) are all fixed instead of just the first.
            """
            # Single-quoted values after : { , [  — trailing delimiter via look-ahead
            text = re.sub(
                r"([{:,[])\s*'([^']*)'(?=\s*[,}\]\n])",
                r'\1"\2"',
                text,
            )
            # Single-quoted keys before :
            text = re.sub(
                r"([{,])\s*'([^']*)'\s*:",
                r'\1"\2":',
                text,
            )
            return text

        def _step_close_braces(text: str) -> str:
            """Case 3: append missing closing braces/brackets."""
            text = text.strip()
            if text.endswith(','):
                text = text[:-1]
            open_braces = text.count('{') - text.count('}')
            open_brackets = text.count('[') - text.count(']')
            if open_brackets > 0:
                text += ']' * open_brackets
            if open_braces > 0:
                text += '}' * open_braces
            return text

        # --- Progressive strategies (least → most aggressive) --------------------
        strategies: list[tuple[str, list[Any]]] = [
            (
                "basic_cleanup",
                [_step_strip_extra_text, _step_remove_trailing_commas],
            ),
            (
                "unquoted_keys",
                [
                    _step_strip_extra_text,
                    _step_remove_trailing_commas,
                    _step_quote_unquoted_keys,
                ],
            ),
            (
                "fix_quotes",
                [
                    _step_strip_extra_text,
                    _step_remove_trailing_commas,
                    _step_quote_unquoted_keys,
                    _step_fix_single_quotes,
                ],
            ),
            (
                "force_close_and_fix",
                [
                    _step_strip_extra_text,
                    _step_fix_single_quotes,
                    _step_quote_unquoted_keys,
                    _step_remove_trailing_commas,
                    _step_close_braces,
                ],
            ),
        ]

        for strategy_name, steps in strategies:
            current = json_str
            for step in steps:
                current = step(current)
            try:
                repaired = json.loads(current)
                logger.info(
                    "JSON repair succeeded via strategy '%s'", strategy_name,
                )
                return repaired
            except json.JSONDecodeError:
                continue

        logger.error("All JSON repair strategies failed; returning error dict")
        return {
            "error": "Failed to parse JSON from LLM output (all repair strategies exhausted)",
            "raw_text": json_str[:1000],
        }

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """
        Extract JSON from LLM output (may be wrapped in markdown).

        Falls back to progressive repair when the initial parse fails,
        handling trailing commas, single quotes, missing braces, and
        other common LLM formatting errors.

        Args:
            text: Raw LLM response.

        Returns:
            Parsed JSON dictionary.
        """
        # Step 1: try to find JSON inside markdown code fences
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if fence_match:
            inner = fence_match.group(1).strip()
        else:
            inner = text

        # Step 2: attempt to isolate a JSON object from the (possibly noisy) text
        json_match = re.search(r'\{[\s\S]*\}', inner)
        if json_match:
            json_str = json_match.group(0).strip()
        else:
            json_str = inner.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(
                "Initial JSON parse failed; attempting progressive repair..."
            )
            return ReportParser._repair_json(json_str)

    @staticmethod
    def validate_report(report: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize the generated report.

        Args:
            report: Parsed report dictionary.

        Returns:
            Validated and normalized report.
        """
        # Ensure all required fields exist
        validated: dict[str, Any] = {}
        for field in ReportParser.REQUIRED_FIELDS:
            validated[field] = report.get(field, "N/A")

        # Normalize confidence score to float
        try:
            validated["confidence_score"] = float(validated["confidence_score"])
            validated["confidence_score"] = max(0.0, min(100.0, validated["confidence_score"]))
        except (ValueError, TypeError):
            validated["confidence_score"] = 0.0

        # Ensure lists
        for list_field in ["evidence", "affected_services", "alternative_causes",
                            "recommended_fixes", "preventive_recommendations"]:
            val = report.get(list_field, [])
            if not isinstance(val, list):
                val = [str(val)] if val else []
            validated[list_field] = val

        # Add any additional fields from the LLM output
        for key, value in report.items():
            if key not in validated and key != "error":
                validated[key] = value

        return validated

    @staticmethod
    def format_for_display(report: dict[str, Any]) -> str:
        """
        Format the report for human-readable display.

        Args:
            report: Validated report dictionary.

        Returns:
            Formatted multi-line string.
        """
        lines = [
            "=" * 50,
            "AI INCIDENT INVESTIGATION REPORT",
            "=" * 50,
            "",
            f"Incident ID:    {report.get('incident_id', 'N/A')}",
            f"Severity:       {report.get('severity', 'N/A')}",
            f"Confidence:     {report.get('confidence_score', 'N/A')}%",
            "",
            "-" * 50,
            "ROOT CAUSE",
            "-" * 50,
            report.get('root_cause', 'N/A'),
            "",
            "-" * 50,
            "EVIDENCE",
            "-" * 50,
        ]

        for ev in report.get('evidence', []):
            lines.append(f"  • {ev}")

        lines += [
            "",
            "-" * 50,
            "AFFECTED SERVICES",
            "-" * 50,
        ]
        for svc in report.get('affected_services', []):
            lines.append(f"  • {svc}")

        lines += [
            "",
            "-" * 50,
            "BUSINESS IMPACT",
            "-" * 50,
            report.get('business_impact', 'N/A'),
            "",
            "-" * 50,
            "RECOMMENDED FIXES",
            "-" * 50,
        ]
        for fix in report.get('recommended_fixes', []):
            lines.append(f"  • {fix}")

        lines += [
            "",
            "-" * 50,
            "ALTERNATIVE CAUSES",
            "-" * 50,
        ]
        for alt in report.get('alternative_causes', []):
            lines.append(f"  • {alt}")

        lines += [
            "",
            "-" * 50,
            "PREVENTIVE RECOMMENDATIONS",
            "-" * 50,
        ]
        for rec in report.get('preventive_recommendations', []):
            lines.append(f"  • {rec}")

        lines += [
            "",
            "-" * 50,
            "RELATED INCIDENTS",
            "-" * 50,
        ]
        for rel in report.get('retrieved_similar_incidents', []):
            lines.append(f"  • {rel}")

        lines += [
            "",
            "-" * 50,
            "SUMMARY",
            "-" * 50,
            report.get('generated_summary', 'N/A'),
            "",
            "=" * 50,
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Full RAG pipeline for incident investigation.

    Orchestrates:
        1. Embedding generation
        2. Vector search
        3. Prompt construction
        4. LLM generation
        5. Report parsing
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        llm_mode: str = "mock",  # "mock" or "local"
    ) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model: Name of the embedding model.
            llm_mode: "mock" for simulated LLM, "local" for real LLM.
        """
        logger.info(f"Initializing RAG Pipeline (llm_mode={llm_mode})")

        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model or settings.EMBEDDING_MODEL_NAME,
        )
        self.vector_db = VectorDatabase()
        self.prompt_builder = PromptBuilder()
        self.llm_mode = llm_mode
        self._llm_generator = None  # Lazy-loaded real LLM

        logger.info(
            f"RAG Pipeline ready | Embeddings: {self.embedding_generator.model_name} | "
            f"Vector DB: {self.vector_db.count()} docs"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        severity_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Step 1-2: Generate embedding + Retrieve similar incidents.

        Args:
            query: Incident description text.
            top_k: Number of results to retrieve.
            severity_filter: Optional severity filter.

        Returns:
            Search results from vector database.
        """
        # Generate query embedding
        query_emb = self.embedding_generator.generate_single_embedding(query)

        # Build metadata filter
        where = None
        if severity_filter:
            where = {"severity": severity_filter}

        # Search
        results = self.vector_db.search(query_emb, top_k=top_k, where=where)
        logger.info(f"Retrieved {results['count']} similar incidents in {results['query_time_ms']}ms")

        return results

    def generate_report(
        self,
        incident_description: str,
        retrieved_results: dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Step 3-5: Build prompt, generate, parse report.

        Args:
            incident_description: Current incident description.
            retrieved_results: Results from retrieve().
            top_k: Number of incidents to use.

        Returns:
            Parsed and validated report dictionary.
        """
        # Build prompt
        prompts = self.prompt_builder.build_prompt(
            incident_description=incident_description,
            retrieved_incidents=retrieved_results,
            top_k=top_k,
        )

        # Generate (mock or real LLM)
        if self.llm_mode == "mock":
            raw_output = self._mock_llm_generate(
                incident_description=incident_description,
                retrieved_results=retrieved_results,
            )
        else:
            raw_output = self._local_llm_generate(prompts)

        # Parse and validate
        report = ReportParser.extract_json(raw_output)

        if "error" in report:
            logger.warning(f"Report parsing issue: {report['error']}")
            return report

        validated = ReportParser.validate_report(report)
        logger.info(f"Generated report: {validated.get('incident_id')} (confidence: {validated.get('confidence_score')}%)")

        return validated

    def investigate(
        self,
        incident_description: str,
        top_k: int = 5,
        rag_enabled: bool = True,
    ) -> dict[str, Any]:
        """
        Full investigation pipeline.

        Args:
            incident_description: Current incident description.
            top_k: Number of similar incidents to retrieve.
            rag_enabled: Whether to use RAG (True) or direct LLM (False).

        Returns:
            Complete investigation results with report and metadata.
        """
        start = time.time()

        # Retrieve
        retrieved_results: dict[str, Any] = {"results": [], "count": 0, "query_time_ms": 0}
        if rag_enabled and self.vector_db.count() > 0:
            retrieved_results = self.retrieve(incident_description, top_k=top_k)

        # Generate
        report = self.generate_report(
            incident_description=incident_description,
            retrieved_results=retrieved_results,
            top_k=top_k,
        )

        processing_time_ms = (time.time() - start) * 1000

        return {
            "report": report,
            "retrieved_incidents": retrieved_results.get("results", []),
            "retrieval_count": retrieved_results.get("count", 0),
            "retrieval_time_ms": retrieved_results.get("query_time_ms", 0),
            "confidence_score": report.get("confidence_score", 0.0),
            "rag_enabled": rag_enabled and retrieved_results.get("count", 0) > 0,
            "processing_time_ms": round(processing_time_ms, 2),
        }

    # -----------------------------------------------------------------------
    # Mock LLM (for testing without real GPU - Phase 4-5 transition)
    # -----------------------------------------------------------------------

    def _mock_llm_generate(
        self,
        incident_description: str,
        retrieved_results: dict[str, Any],
    ) -> str:
        """
        Generate a mock report using retrieved incident data.
        Used for testing the RAG pipeline without a real LLM.

        In production (Phase 5), replace with actual Llama 3/Mistral inference.

        Args:
            incident_description: Current incident description.
            retrieved_results: Retrieved similar incidents.

        Returns:
            JSON report string.
        """
        # Extract relevant info from top retrieved incident
        top_incident = {}
        if retrieved_results.get("results"):
            top = retrieved_results["results"][0]
            top_incident = top.get("metadata", {})

        # Determine incident type from description
        incident_type = "Unknown"
        type_keywords = {
            "OOMKilled": ["oom", "memory", "killed", "out of memory"],
            "CrashLoopBackOff": ["crash", "loop", "backoff", "restart"],
            "ImagePullBackOff": ["imagepull", "pull", "image", "registry"],
            "ConnectionPoolExhaustion": ["connection pool", "max_connections", "pool exhaust"],
            "DNSFailure": ["dns", "resolve", "lookup", "coredns"],
            "CPUThrottling": ["cpu throttl", "throttle", "cpu limit"],
            "NetworkFailure": ["network", "connect", "timeout", "packet loss"],
        }
        desc_lower = incident_description.lower()
        for itype, keywords in type_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                incident_type = itype
                break

        # Generate deterministic mock report
        hash_digest = hashlib.md5(incident_description.encode()).hexdigest()
        incident_num = int(hash_digest[:6], 16) % 9000 + 1000
        incident_id = f"INC-{incident_num:04d}"
        confidence = round(75.0 + np.random.uniform(-10, 20), 1)
        confidence = max(0.0, min(100.0, confidence))

        report = {
            "incident_id": incident_id,
            "severity": top_incident.get("severity", "High"),
            "root_cause": (
                top_incident.get("root_cause", f"{incident_type} detected based on log patterns.")
            ),
            "evidence": [
                f"Error logs consistent with {incident_type} pattern",
                f"Similar to historical incident {top_incident.get('incident_id', 'N/A')}",
                "Affected services confirmed via monitoring alerts",
            ],
            "affected_services": (
                top_incident.get("affected_services", "N/A").split(", ")
                if isinstance(top_incident.get("affected_services"), str)
                else ["payment-service", "order-service"]
            ),
            "affected_pods": ["payment-service-7d4f", "payment-service-8e5g"],
            "business_impact": "Degraded service availability with potential customer impact",
            "timeline": f"{incident_type} detected at monitoring alert time, ongoing investigation",
            "confidence_score": confidence,
            "alternative_causes": [
                "Network connectivity issue between services",
                "Misconfiguration in deployment manifest",
            ],
            "recommended_fixes": [
                top_incident.get("resolution", "Restart affected pods and verify configuration"),
                "Review resource limits and update if necessary",
                "Enable additional monitoring alerts",
            ],
            "preventive_recommendations": [
                "Implement automated resource scaling policies",
                "Add pre-deployment validation checks",
                "Set up proactive monitoring dashboards",
            ],
            "retrieved_similar_incidents": [
                r.get("incident_id", "N/A")
                for r in retrieved_results.get("results", [])[:3]
            ],
            "generated_summary": (
                f"The {incident_type} incident was investigated using RAG-based analysis. "
                f"Root cause identified with {confidence}% confidence. "
                f"Similar historical incidents were retrieved to guide the investigation. "
                f"Recommended fixes have been provided based on proven resolutions."
            ),
        }

        return json.dumps(report, indent=2)

    def _local_llm_generate(self, prompts: dict[str, str]) -> str:
        """
        Generate report using a locally loaded LLM via HuggingFace transformers.

        Lazy-loads the LLM on first call to avoid loading when not needed.
        Supports Qwen, Phi-3, and Llama 3 models on CPU or GPU.

        Args:
            prompts: Dictionary with 'system' and 'user' prompts.

        Returns:
            LLM-generated text (should contain JSON report).
        """
        # Lazy-load the LLM (only when first used)
        if self._llm_generator is None:
            from models.llm_inference import LLMGenerator
            logger.info(f"Loading real LLM: {settings.LLM_MODEL_NAME}...")
            self._llm_generator = LLMGenerator(
                model_name=settings.LLM_MODEL_NAME,
                device=settings.LLM_DEVICE,
            )

        try:
            return self._llm_generator.generate_json(
                system_prompt=prompts["system"],
                user_prompt=prompts["user"],
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
            )
        except Exception as e:
            logger.error(f"Real LLM generation failed: {e}. Falling back to mock mode.")
            return self._mock_llm_generate(
                incident_description=prompts.get("user", ""),
                retrieved_results={"results": []},
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def investigate_incident(
    description: str,
    top_k: int = 5,
    rag_enabled: bool = True,
) -> dict[str, Any]:
    """
    Convenience function: run full investigation on an incident.

    Args:
        description: Incident description text.
        top_k: Number of similar incidents to retrieve.
        rag_enabled: Enable/disable RAG.

    Returns:
        Investigation results dictionary.
    """
    pipeline = RAGPipeline()
    return pipeline.investigate(description, top_k=top_k, rag_enabled=rag_enabled)


# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

def main() -> None:
    """Test the RAG pipeline with a sample incident."""
    sample_incident = (
        "Pod payment-service-7d4f in namespace production is in CrashLoopBackOff. "
        "Logs show: 'Error: connection refused' to PostgreSQL at 10.0.2.15:5432. "
        "The database pod is running but max_connections has been reached."
    )

    pipeline = RAGPipeline(llm_mode="mock")

    print("\n" + "=" * 60)
    print("KubeSage RAG Pipeline Test")
    print("=" * 60)
    print(f"\nIncident: {sample_incident[:100]}...")
    print(f"\nVector DB: {pipeline.vector_db.count()} indexed incidents\n")

    results = pipeline.investigate(sample_incident, top_k=3)

    print(f"Processing time: {results['processing_time_ms']:.1f}ms")
    print(f"Retrieved: {results['retrieval_count']} similar incidents")
    print(f"RAG enabled: {results['rag_enabled']}")

    if "error" not in results["report"]:
        print("\n" + ReportParser.format_for_display(results["report"]))
    else:
        print(f"\nError: {results['report']['error']}")


if __name__ == "__main__":
    main()
