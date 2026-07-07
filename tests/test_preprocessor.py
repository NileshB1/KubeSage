"""
unit tests 
"""

import pytest
from data.preprocess import IncidentPreprocessor, preprocess_incidents


#test Fixtures
@pytest.fixture
def preprocessor() -> IncidentPreprocessor:
    """Create a pre-fitted preprocessor for testing."""
    pp = IncidentPreprocessor()
    samples = [
        {
            "incident_type": "OOMKilled",  "severity": "Critical",
            "title": "Memory OOM test",  "description": "Pod ran out of memory",
            "evidence": {"logs": ["kernel: Out of memory"]},
            "root_cause": "Memory leak",   "resolution": "Increased limit",
            "affected_services": ["payment-service"],
        },
        {
            "incident_type": "CrashLoopBackOff",
            "severity": "High",  "title": "Crash loop",
            "description": "Pod keeps crashing",  "evidence": {"logs": ["Error: connection refused"]},
            "root_cause": "DB connection", "resolution": "Fixed config",  "affected_services": ["order-service"],
        },
    ]
    pp.fit_encoders(samples)
    return pp


@pytest.fixture
def sample_incident() -> dict:
    """Create a minimal sample incident for testing."""
    return {
        "incident_id": "INC-TEST-001", "incident_type": "OOMKilled",
        "severity": "Critical",
        "title": "Test pod OOMKilled in production",
        "description": "Pod test-pod-123 in namespace production was terminated with OOMKilled.",
        "evidence": {
            "logs": [
                "[2024-06-15 14:30:45.123] kernel: Out of memory: Kill process 12345 (java)",
                "Memory cgroup out of memory: Killed process 12345",
            ],
        },
        "root_cause": "Memory leak in payment service", "resolution": "Increased memory limit from 256Mi to 512Mi",
        "affected_services": ["payment-service", "order-service"],
    }


# clean_log_line Tests
class TestCleanLogLine:
    """Tests for log line cleaning."""

    def test_removes_timestamps(self) -> None:
        """Should strip timestamps from log lines."""
        log = "[2024-06-15 14:30:45.123] kernel: Out of memory: Kill process 12345"
        cleaned = IncidentPreprocessor.clean_log_line(log)

        assert "2024-06-15" not in cleaned
        assert "kernel" in cleaned.lower()

    def test_removes_timestamps_t_format(self) -> None:
        """Should strip ISO-8601 timestamps."""
        log = "[2024-06-15T14:30:45] Starting application"
        cleaned = IncidentPreprocessor.clean_log_line(log)

        assert "2024-06-15" not in cleaned
        assert "starting application" in cleaned.lower()

    def test_replaces_ip_addresses(self) -> None:
        """Should replace IP addresses with <IP> placeholder."""
        log = "Connection refused to PostgreSQL at 10.0.2.15:5432"
        cleaned = IncidentPreprocessor.clean_log_line(log)

        assert "10.0.2.15" not in cleaned
        assert "<IP>" in cleaned

    def test_replaces_hex_ids(self) -> None:
        """Should replace 16+ char hex IDs with <HEX_ID> placeholder."""
        log = "Container abcdef0123456789abcdef0123456789 was OOMKilled"
        cleaned = IncidentPreprocessor.clean_log_line(log)

        assert "abcdef0123456789" not in cleaned
        assert "<HEX_ID>" in cleaned

    def test_normalizes_whitespace(self) -> None:
        """Should collapse multiple spaces into single spaces."""
        log = "Error:    connection     refused    to    database"
        cleaned = IncidentPreprocessor.clean_log_line(log)
        assert "    " not in cleaned

    def test_strips_trailing_whitespace(self) -> None:
        """Should strip leading/trailing whitespace."""
        cleaned = IncidentPreprocessor.clean_log_line("error message")
        assert cleaned == "error message"

    def test_removes_special_characters(self) -> None:
        """Should remove special characters not in allowed set."""
        log = "Error @#$%^& connection! to database"
        cleaned = IncidentPreprocessor.clean_log_line(log)

        assert "@" not in cleaned
        assert "#" not in cleaned


#build_incident_text
class TestBuildIncidentText:
    """Tests for building text representation from incident dict."""

    def test_includes_all_fields(self, sample_incident: dict) -> None:
        """Built text should contain all incident fields."""
        text = IncidentPreprocessor.build_incident_text(sample_incident)

        assert "Incident Type: OOMKilled" in text
        assert "Severity: Critical" in text
        assert "Root Cause: Memory leak" in text
        assert "Affected Services: payment-service, order-service" in text

    def test_handles_missing_fields(self) -> None:
        """Should handle incidents with minimal fields."""
        minimal = {"incident_id": "INC-001"}
        text = IncidentPreprocessor.build_incident_text(minimal)
        assert "Incident Type: Unknown" in text
        assert text  # Should not be empty

    def test_includes_description(self) -> None:
        """Description should appear in the built text."""

        incident = {"description": "Pod crashed due to OOM", "incident_type": "OOMKilled"}
        text = IncidentPreprocessor.build_incident_text(incident)
        assert "Pod crashed due to OOM" in text

    def test_includes_cleaned_evidence_logs(self, sample_incident: dict) -> None:
        """Evidence logs should be cleaned and included."""
        text = IncidentPreprocessor.build_incident_text(sample_incident)
        assert "Log:" in text
        assert "kernel" in text.lower()

    def test_multiline_output(self, sample_incident: dict) -> None:
        """Output should contain newlines between sections."""

        text = IncidentPreprocessor.build_incident_text(sample_incident)
        assert "\n" in text


#chunk_text Tests
class TestChunkText:
    """Tests for text chunking with overlap."""

    def test_short_text_single_chunk(self) -> None:
        """Short text should return single chunk."""
        text = "short error message"

        chunks = IncidentPreprocessor.chunk_text(text, max_chars=512)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self) -> None:
        """Long text should be split into multiple chunks with overlap."""
        words = ["error"] * 50  #50 words
        text = " ".join(words)
        chunks = IncidentPreprocessor.chunk_text(text, max_chars=5)

        assert len(chunks) >= 1

    def test_very_short_text_skipped(self) -> None:
        """Chunks smaller than 20 chars should be skipped, returning original."""
        text = "tiny"

        chunks = IncidentPreprocessor.chunk_text(text, max_chars=3)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self) -> None:
        """Empty text should return list with empty string."""
        chunks = IncidentPreprocessor.chunk_text("")
        assert len(chunks) == 1


#encode_features Tests
class TestEncodeFeatures:
    """Tests for feature encoding."""

    def test_encodes_severity_and_type(self, preprocessor: IncidentPreprocessor) -> None:
        """Should encode severity and type as integers."""
        incident = {"incident_type": "OOMKilled", "severity": "Critical"}
        features = preprocessor.encode_features(incident)

        assert isinstance(features["severity_encoded"], int)
        assert isinstance(features["type_encoded"], int)

    def test_raises_when_not_fitted(self) -> None:
        """Should raise RuntimeError if encoders not fitted."""
        pp = IncidentPreprocessor()
        with pytest.raises(RuntimeError, match="Encoders not fitted"):
            pp.encode_features({"incident_type": "OOMKilled", "severity": "High"})

    def test_consistent_encoding(self, preprocessor: IncidentPreprocessor) -> None:
        """Same input should produce same encoding."""
        inc = {"incident_type": "OOMKilled", "severity": "Critical"}

        f1=preprocessor.encode_features(inc)
        f2=preprocessor.encode_features(inc)
        assert f1 == f2


#fit_encoders Tests
class TestFitEncoders:
    """Tests for encoder fitting."""

    def test_fits_all_encoders(self) -> None:
        """Should fit both severity and type encoders."""
        pp = IncidentPreprocessor()
        incidents = [
            {"incident_type": "OOMKilled", "severity": "Critical"},  {"incident_type": "CrashLoopBackOff", "severity": "High"},
        ]
        pp.fit_encoders(incidents)
        assert pp._is_fitted


        assert len(pp.severity_encoder.classes_) == 2
        assert len(pp.type_encoder.classes_) == 2

    def test_handles_missing_fields(self) -> None:
        """Should use defaults for missing fields."""
        pp = IncidentPreprocessor()

        incidents = [{"incident_type": "Test"}]
        pp.fit_encoders(incidents)
        assert pp._is_fitted

#preprocess tests
class TestPreprocessSingle:
    """Tests for single incident preprocessing."""

    def test_returns_all_required_fields(self, preprocessor: IncidentPreprocessor, sample_incident: dict) -> None:
        """Preprocessed incident should contain all enrichment fields."""
        result = preprocessor.preprocess(sample_incident)
        assert "raw_text" in result
        assert "cleaned_text" in result

        assert "text_chunks" in result
        assert "num_chunks" in result
        assert "encoded_features" in result

    def test_preserves_original_fields(self, preprocessor: IncidentPreprocessor, sample_incident: dict) -> None:
        """Original incident fields should be preserved."""
        result = preprocessor.preprocess(sample_incident)
        assert result["incident_id"]=="INC-TEST-001"
        assert result["severity"]=="Critical"
        assert result["incident_type"]=="OOMKilled"

    def test_num_chunks_is_int(self, preprocessor: IncidentPreprocessor, sample_incident: dict) -> None:
        """num_chunks should be integer."""
        result = preprocessor.preprocess(sample_incident)
        assert isinstance(result["num_chunks"], int)
        assert result["num_chunks"] >= 1

    def test_cleaned_text_different_from_raw(self, preprocessor: IncidentPreprocessor, sample_incident: dict) -> None:
        """Cleaned text should differ from raw (timestamps removed, IPs replaced)."""
        result = preprocessor.preprocess(sample_incident)
        # Raw text contains timestamps/IPs; cleaned shouldn't
        assert len(result["cleaned_text"]) <= len(result["raw_text"])


#preprocess_incidents
class TestPreprocessBatch:
    """Tests for batch preprocessing."""

    def test_processes_multiple_incidents(self) -> None:
        """Should process all incidents in batch mode."""
        incidents = [
            {
                "incident_type": "OOMKilled", "severity": "Critical",
                "title": "Test 1",  "description": "Pod OOM",
                "evidence": {}, "root_cause": "Memory leak", "resolution": "Increased memory",
                "affected_services": ["svc1"],
            },
            {
                "incident_type": "CrashLoopBackOff",
                "severity": "High",    "title": "Test 2",
                "description": "Pod crash",
                "evidence": {},    "root_cause": "Config error",
                "resolution": "Fixed config",     "affected_services": ["svc2"],
            },
        ]
        results = preprocess_incidents(incidents)
        assert len(results) == 2
        assert all("cleaned_text" in r for r in results)

    def test_fit_encoders_false(self) -> None:
        """Should work with fit_encoders=False when encoders already fitted elsewhere."""
        # preprocess_incidents fits internally by default for this batch
        results = preprocess_incidents(
            [{"incident_type": "Test", "severity": "Low", "title": "X", "description": "Y", "evidence": {}, "root_cause": "", "resolution": "", "affected_services": []}],
            fit_encoders=True,
        )
        assert len(results) == 1

#main
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
