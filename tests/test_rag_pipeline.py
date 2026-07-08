"""
Unit Tests — RAG Pipeline Orchestrator
======================================
Tests the PromptBuilder string interpolation, the ReportParser regex scrubbers 
(for stripping markdown blocks and extracting JSON), and mock execution of the orchestrator.
"""

import json
import pytest
from models.rag_pipeline import PromptBuilder, ReportParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> PromptBuilder:
    """Create a fresh PromptBuilder instance."""
    return PromptBuilder()


@pytest.fixture
def search_results_with_data() -> dict:
    """Sample search results with 3 retrieved incidents."""
    return {
        "results": [
            {
                "incident_id": "INC-1001",
                "similarity_score": 0.95,
                "metadata": {
                    "incident_type": "OOMKilled",
                    "severity": "Critical",
                    "root_cause": "Memory leak in payment service",
                    "resolution": "Increased memory limit to 512Mi",
                    "affected_services": "payment-service, order-service",
                },
                "document": "incident description 1",
            },
            {
                "incident_id": "INC-1002",
                "similarity_score": 0.87,
                "metadata": {
                    "incident_type": "NetworkFailure",
                    "severity": "High",
                    "root_cause": "NetworkPolicy blocking egress",
                    "resolution": "Updated NetworkPolicy rules",
                    "affected_services": "api-gateway, payment-service",
                },
                "document": "incident description 2",
            },
            {
                "incident_id": "INC-1003",
                "similarity_score": 0.72,
                "metadata": {
                    "incident_type": "CrashLoopBackOff",
                    "severity": "High",
                    "root_cause": "ConfigMap misconfiguration",
                    "resolution": "Fixed ConfigMap",
                    "affected_services": "notification-svc",
                },
                "document": "incident description 3",
            },
        ],
        "count": 3,
        "query_time_ms": 12.5,
        "top_k": 5,
    }


@pytest.fixture
def empty_search_results() -> dict:
    """Sample search results with no incidents retrieved."""
    return {
        "results": [],
        "count": 0,
        "query_time_ms": 5.0,
        "top_k": 5,
    }


# ===========================================================================
# PromptBuilder Tests
# ===========================================================================

class TestPromptBuilderFormatRetrieved:
    """Tests for format_retrieved_incidents method."""

    def test_formats_multiple_incidents(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """Should format all incidents with proper structure."""
        formatted = builder.format_retrieved_incidents(
            search_results_with_data, max_incidents=3,
        )
        assert "--- Incident 1" in formatted
        assert "--- Incident 2" in formatted
        assert "--- Incident 3" in formatted

    def test_includes_similarity_scores(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """Formatted output should show similarity scores."""
        formatted = builder.format_retrieved_incidents(
            search_results_with_data, max_incidents=1,
        )
        assert "Similarity: 0.95" in formatted

    def test_includes_metadata_fields(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """Formatted output should include type, severity, root cause, resolution."""
        formatted = builder.format_retrieved_incidents(
            search_results_with_data, max_incidents=1,
        )
        assert "Type: OOMKilled" in formatted
        assert "Severity: Critical" in formatted
        assert "Root Cause: Memory leak" in formatted
        assert "Resolution: Increased memory limit" in formatted

    def test_respects_max_incidents(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """Should limit results to max_incidents."""
        formatted = builder.format_retrieved_incidents(
            search_results_with_data, max_incidents=1,
        )
        assert "--- Incident 1" in formatted
        assert "--- Incident 2" not in formatted

    def test_empty_results_returns_placeholder(
        self, builder: PromptBuilder, empty_search_results: dict,
    ) -> None:
        """Empty results should return a clear placeholder message."""
        formatted = builder.format_retrieved_incidents(
            empty_search_results, max_incidents=5,
        )
        assert "No similar incidents" in formatted

    def test_no_results_key_handles_gracefully(self, builder: PromptBuilder) -> None:
        """Missing 'results' key should be handled gracefully."""
        formatted = builder.format_retrieved_incidents({}, max_incidents=5)
        assert "No similar incidents" in formatted


class TestPromptBuilderKnowledgeBase:
    """Tests for get_knowledge_base."""

    def test_returns_non_empty_string(self, builder: PromptBuilder) -> None:
        """Knowledge base should be a non-empty string."""
        kb = builder.get_knowledge_base()
        assert isinstance(kb, str)
        assert len(kb) > 100

    def test_covers_all_incident_types(self, builder: PromptBuilder) -> None:
        """Knowledge base should mention all 7 incident types."""
        kb = builder.get_knowledge_base()
        expected_types = [
            "OOMKilled", "CrashLoopBackOff", "ImagePullBackOff",
            "ConnectionPoolExhaustion", "DNSFailure", "CPUThrottling",
            "NetworkFailure",
        ]
        for itype in expected_types:
            assert itype in kb

    def test_includes_troubleshooting_commands(self, builder: PromptBuilder) -> None:
        """Knowledge base should include kubectl troubleshooting commands."""
        kb = builder.get_knowledge_base()
        assert "kubectl describe pod" in kb
        assert "kubectl logs" in kb
        assert "kubectl get events" in kb


class TestPromptBuilderBuildPrompt:
    """Tests for build_prompt method."""

    def test_returns_system_and_user_keys(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """Should return dict with 'system' and 'user' keys."""
        prompt = builder.build_prompt(
            incident_description="test incident",
            retrieved_incidents=search_results_with_data,
        )
        assert "system" in prompt
        assert "user" in prompt

    def test_system_prompt_is_non_empty(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """System prompt should be non-empty."""
        prompt = builder.build_prompt(
            incident_description="test",
            retrieved_incidents=search_results_with_data,
        )
        assert len(prompt["system"]) > 0

    def test_user_prompt_contains_incident_description(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """User prompt should contain the incident description."""
        prompt = builder.build_prompt(
            incident_description="Pod OOMKilled in production",
            retrieved_incidents=search_results_with_data,
        )
        assert "Pod OOMKilled in production" in prompt["user"]

    def test_user_prompt_contains_retrieved_incidents(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """User prompt should contain formatted retrieved incidents."""
        prompt = builder.build_prompt(
            incident_description="test",
            retrieved_incidents=search_results_with_data,
        )
        assert "INC-1001" in prompt["user"]

    def test_user_prompt_contains_json_instructions(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """User prompt should include JSON format instructions."""
        prompt = builder.build_prompt(
            incident_description="test",
            retrieved_incidents=search_results_with_data,
        )
        assert "incident_id" in prompt["user"]
        assert "severity" in prompt["user"]

    def test_respects_top_k(
        self, builder: PromptBuilder, search_results_with_data: dict,
    ) -> None:
        """Should limit included incidents to top_k."""
        prompt_k1 = builder.build_prompt(
            incident_description="test", retrieved_incidents=search_results_with_data, top_k=1,
        )
        prompt_k3 = builder.build_prompt(
            incident_description="test", retrieved_incidents=search_results_with_data, top_k=3,
        )
        assert "--- Incident 1" in prompt_k1["user"]
        assert "--- Incident 2" not in prompt_k1["user"]
        assert "--- Incident 3" in prompt_k3["user"]

    def test_empty_retrieval_does_not_crash(
        self, builder: PromptBuilder, empty_search_results: dict,
    ) -> None:
        """Building prompt with empty retrieval should not crash."""
        prompt = builder.build_prompt(
            incident_description="test",
            retrieved_incidents=empty_search_results,
        )
        assert "user" in prompt
        assert "No similar incidents" in prompt["user"]


# ===========================================================================
# ReportParser Tests
# ===========================================================================

class TestReportParserExtractJson:
    """Tests for extract_json method."""

    def test_extracts_plain_json(self) -> None:
        """Should parse plain JSON string."""
        text = '{"incident_id": "INC-001", "severity": "Critical"}'
        result = ReportParser.extract_json(text)
        assert result["incident_id"] == "INC-001"
        assert result["severity"] == "Critical"

    def test_extracts_markdown_wrapped_json(self) -> None:
        """Should extract JSON from markdown code block."""
        text = '```json\n{"incident_id": "INC-002", "severity": "High"}\n```'
        result = ReportParser.extract_json(text)
        assert result["incident_id"] == "INC-002"

    def test_extracts_json_without_lang_specifier(self) -> None:
        """Should extract JSON from markdown code block without language."""
        text = '```\n{"incident_id": "INC-003"}\n```'
        result = ReportParser.extract_json(text)
        assert result["incident_id"] == "INC-003"

    def test_invalid_json_returns_error(self) -> None:
        """Should return error dict for invalid JSON."""
        text = "This is not valid JSON at all"
        result = ReportParser.extract_json(text)
        assert "error" in result
        assert "raw_text" in result

    def test_nested_markdown_extraction(self) -> None:
        """Should find JSON within larger markdown text."""
        text = 'Here is my report:\n\n```json\n{"incident_id": "INC-005"}\n```\n\nDetails follow...'
        result = ReportParser.extract_json(text)
        assert result["incident_id"] == "INC-005"

    def test_handles_complex_json(self) -> None:
        """Should parse JSON with arrays and nested objects."""
        report = {
            "incident_id": "INC-001",
            "evidence": ["log1", "log2"],
            "alternative_causes": ["cause1", "cause2"],
            "confidence_score": 85.5,
        }
        text = json.dumps(report)
        result = ReportParser.extract_json(text)
        assert result["evidence"] == ["log1", "log2"]
        assert result["confidence_score"] == 85.5


class TestReportParserValidateReport:
    """Tests for validate_report method."""

    def test_all_required_fields_present(self) -> None:
        """Validated report should contain all REQUIRED_FIELDS."""
        report = {
            "incident_id": "INC-001",
            "severity": "Critical",
            "root_cause": "Memory leak",
            "evidence": ["log1"],
            "affected_services": ["svc1"],
            "confidence_score": 90.0,
            "recommended_fixes": ["fix1"],
            "generated_summary": "Summary text",
        }
        validated = ReportParser.validate_report(report)
        for field in ReportParser.REQUIRED_FIELDS:
            assert field in validated

    def test_missing_fields_get_default(self) -> None:
        """Missing required fields should get 'N/A' or defaults."""
        report = {"incident_id": "INC-001"}
        validated = ReportParser.validate_report(report)
        assert validated["incident_id"] == "INC-001"
        assert validated["severity"] == "N/A"
        assert validated["confidence_score"] == 0.0
        assert validated["evidence"] == []

    def test_confidence_score_clamped(self) -> None:
        """Confidence score should be clamped to [0, 100]."""
        r1 = ReportParser.validate_report({"confidence_score": 150})
        assert r1["confidence_score"] == 100.0

        r2 = ReportParser.validate_report({"confidence_score": -10})
        assert r2["confidence_score"] == 0.0

    def test_confidence_score_string_converted(self) -> None:
        """String confidence score should be converted to float."""
        validated = ReportParser.validate_report({"confidence_score": "87.5"})
        assert validated["confidence_score"] == 87.5
        assert isinstance(validated["confidence_score"], float)

    def test_invalid_confidence_defaults_to_zero(self) -> None:
        """Invalid confidence should default to 0.0."""
        validated = ReportParser.validate_report({"confidence_score": "N/A"})
        assert validated["confidence_score"] == 0.0

    def test_scalar_lists_converted(self) -> None:
        """Scalar values for list fields should be wrapped in list."""
        report = {
            "evidence": "single evidence item",
            "affected_services": "payment-service",
        }
        validated = ReportParser.validate_report(report)
        assert isinstance(validated["evidence"], list)
        assert validated["evidence"] == ["single evidence item"]
        assert validated["affected_services"] == ["payment-service"]

    def test_additional_fields_preserved(self) -> None:
        """Non-required fields from LLM should be preserved."""
        report = {"incident_id": "INC-001", "custom_field": "custom_value"}
        validated = ReportParser.validate_report(report)
        assert validated.get("custom_field") == "custom_value"


class TestReportParserFormatForDisplay:
    """Tests for format_for_display method."""

    def test_includes_all_sections(self) -> None:
        """Display output should contain all report sections."""
        report = {
            "incident_id": "INC-001",
            "severity": "Critical",
            "confidence_score": 93.0,
            "root_cause": "Memory leak",
            "evidence": ["Evidence 1", "Evidence 2"],
            "affected_services": ["payment-svc", "order-svc"],
            "business_impact": "Revenue loss",
            "recommended_fixes": ["Fix 1", "Fix 2"],
            "alternative_causes": ["Alt 1"],
            "preventive_recommendations": ["Prev 1"],
            "retrieved_similar_incidents": ["INC-100", "INC-200"],
            "generated_summary": "Summary",
        }
        formatted = ReportParser.format_for_display(report)

        assert "AI INCIDENT INVESTIGATION REPORT" in formatted
        assert "ROOT CAUSE" in formatted
        assert "EVIDENCE" in formatted
        assert "AFFECTED SERVICES" in formatted
        assert "BUSINESS IMPACT" in formatted
        assert "RECOMMENDED FIXES" in formatted
        assert "ALTERNATIVE CAUSES" in formatted
        assert "PREVENTIVE RECOMMENDATIONS" in formatted
        assert "RELATED INCIDENTS" in formatted
        assert "SUMMARY" in formatted

    def test_shows_incident_id_and_severity(self) -> None:
        """Should show incident ID, severity, and confidence."""
        report = {
            "incident_id": "INC-999",
            "severity": "High",
            "confidence_score": 85.0,
            "root_cause": "test",
            "evidence": [],
            "affected_services": [],
            "business_impact": "N/A",
            "recommended_fixes": [],
            "alternative_causes": [],
            "preventive_recommendations": [],
            "retrieved_similar_incidents": [],
            "generated_summary": "summary",
        }
        formatted = ReportParser.format_for_display(report)
        assert "INC-999" in formatted
        assert "High" in formatted
        assert "85" in formatted

    def test_multiline_output(self) -> None:
        """Output should be multi-line."""
        report = {
            "incident_id": "TEST",
            "severity": "Low",
            "confidence_score": 50.0,
            "root_cause": "test",
            "evidence": [],
            "affected_services": [],
            "business_impact": "",
            "recommended_fixes": [],
            "alternative_causes": [],
            "preventive_recommendations": [],
            "retrieved_similar_incidents": [],
            "generated_summary": "",
        }
        formatted = ReportParser.format_for_display(report)
        assert "\n" in formatted
        assert len(formatted.split("\n")) > 10

    def test_handles_missing_optional_fields(self) -> None:
        """Should handle reports with missing optional sections."""
        report = {
            "incident_id": "INC-001",
            "severity": "Low",
            "confidence_score": 50.0,
            "root_cause": "test",
            "evidence": [],
            "affected_services": [],
            "business_impact": "N/A",
            "recommended_fixes": [],
            "alternative_causes": [],
            "preventive_recommendations": [],
            "retrieved_similar_incidents": [],
            "generated_summary": "",
        }
        formatted = ReportParser.format_for_display(report)
        # Should not crash
        assert "AI INCIDENT INVESTIGATION REPORT" in formatted


# ===========================================================================
# RAGPipeline Tests (mock mode)
# ===========================================================================

class TestRAGPipelineMock:
    """Tests for RAGPipeline mock mode (no real LLM or embeddings needed)."""

    def test_mock_generate_returns_json(self) -> None:
        """Mock generation should return valid JSON."""
        from models.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(llm_mode="mock")
        json_str = pipeline._mock_llm_generate(
            incident_description="Pod OOMKilled in production",
            retrieved_results={"results": []},
        )
        # Should not raise
        parsed = json.loads(json_str)
        assert "incident_id" in parsed

    def test_mock_generate_with_retrieved_data(self) -> None:
        """Mock generation should use retrieved incident data."""
        from models.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(llm_mode="mock")
        json_str = pipeline._mock_llm_generate(
            incident_description="Pod OOMKilled due to memory leak",
            retrieved_results={
                "results": [
                    {
                        "incident_id": "INC-1001",
                        "similarity_score": 0.95,
                        "metadata": {
                            "incident_type": "OOMKilled",
                            "severity": "Critical",
                            "root_cause": "Memory leak in payment service",
                            "resolution": "Increased memory limit",
                            "affected_services": "payment-service, order-service",
                        },
                    },
                ],
            },
        )
        parsed = json.loads(json_str)
        assert parsed["severity"] == "Critical"
        # Should detect OOMKilled type from description
        assert "Memory" in parsed["root_cause"] or "OOMKilled" in parsed["root_cause"]

    def test_mock_generate_deterministic_id(self) -> None:
        """Same description should produce same incident ID."""
        from models.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(llm_mode="mock")
        desc = "Pod crash loop backoff"
        json1 = pipeline._mock_llm_generate(desc, {"results": []})
        json2 = pipeline._mock_llm_generate(desc, {"results": []})
        id1 = json.loads(json1)["incident_id"]
        id2 = json.loads(json2)["incident_id"]
        assert id1 == id2  # Deterministic via md5 hash

    def test_mock_generate_includes_all_report_fields(self) -> None:
        """Generated mock report should include all important fields."""
        from models.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(llm_mode="mock")
        json_str = pipeline._mock_llm_generate(
            incident_description="Network timeout to database",
            retrieved_results={"results": []},
        )
        parsed = json.loads(json_str)
        required = [
            "incident_id", "severity", "root_cause", "evidence",
            "affected_services", "confidence_score", "recommended_fixes",
            "alternative_causes", "preventive_recommendations",
            "generated_summary",
        ]
        for field in required:
            assert field in parsed, f"Missing field: {field}"

    def test_mock_confidence_in_range(self) -> None:
        """Confidence score should be in [0, 100]."""
        from models.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(llm_mode="mock")
        json_str = pipeline._mock_llm_generate("Test incident", {"results": []})
        parsed = json.loads(json_str)
        assert 0.0 <= parsed["confidence_score"] <= 100.0


class TestRAGPipelineInvestigate:
    """Tests for the investigate() convenience function."""

    def test_investigate_returns_structure(self) -> None:
        """investigate() should return expected result structure."""
        from models.rag_pipeline import investigate_incident
        result = investigate_incident(
            description="Pod OOMKilled in production",
            top_k=3,
            rag_enabled=True,
        )
        assert "report" in result
        assert "retrieved_incidents" in result
        assert "retrieval_count" in result
        assert "confidence_score" in result
        assert "rag_enabled" in result
        assert "processing_time_ms" in result

    def test_investigate_rag_disabled(self) -> None:
        """When RAG disabled, should still generate report without retrieval."""
        from models.rag_pipeline import investigate_incident
        result = investigate_incident(
            description="Test incident",
            rag_enabled=False,
        )
        assert result["rag_enabled"] is False
        assert result["retrieval_count"] == 0
        assert "report" in result

    def test_investigate_report_is_valid(self) -> None:
        """Generated report should pass validation."""
        from models.rag_pipeline import investigate_incident
        result = investigate_incident(
            description="Pod crash loop with database connection refused",
            top_k=3,
            rag_enabled=True,
        )
        report = result["report"]
        # Check required fields are present
        assert report.get("incident_id") is not None
        assert report.get("severity") is not None
        assert isinstance(report.get("evidence"), list)
        assert isinstance(report.get("recommended_fixes"), list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
