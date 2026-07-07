"""
Cleans and formats Kubernetes logs, pod names, IP addresses, and metadata.
Prepares unstructured incident tickets for downstream SBERT embedding generation
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

#ensure project root is on the path for cross-module imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from sklearn.preprocessing import LabelEncoder


class IncidentPreprocessor:
    """
    Preprocesses raw Kubernetes incident data for embedding generation.

    """
    
    def __init__(self) -> None:
        self.severity_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self._is_fitted = False
    

    #text cleaning
    @staticmethod
    def clean_log_line(log: str) -> str:
        """
        Clean a single log line.
        Remove timestamps, Remove IP addresses, Normalize whitespace, Convert to lowercase

        """
        #remove timestamps like [2024-01-15 14:30:45.123]
        log = re.sub(r'\[\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[.,\d]*\]?', '', log)
        #eemove hex IDs (container IDs, etc.)
        log = re.sub(r'[0-9a-f]{16,}', '<HEX_ID>', log)
        
        #normalize IP addresses
        log = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', log)
        #remove excessive punctuation
        log = re.sub(r'[^\w\s<>.,:;/=-]', ' ', log)
        
        #normalize whitespace
        log = re.sub(r'\s+', ' ', log).strip()
        
        return log
    
    @staticmethod
    def build_incident_text(incident: dict[str, Any]) -> str:
        """
        Build a single text representation from all incident fields. This text will be embedded 
        using Sentence Transformers
        """
        parts: list[str] = []
        
        #title and type
        parts.append(f"Incident Type: {incident.get('incident_type', 'Unknown')}")
        parts.append(f"Title: {incident.get('title', '')}")

        parts.append(f"Severity: {incident.get('severity', 'Unknown')}")
        
        #description
        desc = incident.get('description', '')
        if desc:
            parts.append(f"Description: {desc}")
        
        #evidence logs
        evidence = incident.get('evidence', {})
        logs = evidence.get('logs', [])
        if logs:
            for log in logs:
                cleaned = IncidentPreprocessor.clean_log_line(log)
                parts.append(f"Log: {cleaned}")
        
        # Root cause and resolution
        root_cause = incident.get('root_cause', '')
        if root_cause:
            parts.append(f"Root Cause: {root_cause}")
        

        resolution = incident.get('resolution', '')
        if resolution:
            parts.append(f"Resolution: {resolution}")
        
        #affected services
        services = incident.get('affected_services', [])
        if services:
            parts.append(f"Affected Services: {', '.join(services)}")
        
        return '\n'.join(parts)
    
    @staticmethod
    def chunk_text(text: str, max_chars: int = 512) -> list[str]:
        """
        Split long text into overlapping chunks for embedding.
 
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_chars // 2):  # 50% overlap
            chunk = ' '.join(words[i:i + max_chars])
            if len(chunk) < 20:  # Skip very short chunks
                break
            chunks.append(chunk)
        return chunks if chunks else [text]
    
 
    #feature enconding   
    def fit_encoders(self, incidents: list[dict[str, Any]]) -> None:
        """
        Fit label encoders on the incident dataset.
        
        Args:
            incidents: List of incident dictionaries.
        """
        severities = [inc.get('severity', 'Medium') for inc in incidents]
        types = [inc.get('incident_type', 'Unknown') for inc in incidents]
        
        self.severity_encoder.fit(severities)
        self.type_encoder.fit(types)
        self._is_fitted = True
        
        print(f"[OK] Fitted encoders: {len(self.severity_encoder.classes_)} severity levels, "
              f"{len(self.type_encoder.classes_)} incident types")
    
    def encode_features(self, incident: dict[str, Any]) -> dict[str, int]:
        """
        Encode categorical features for a single incident
        """
        if not self._is_fitted:
            raise RuntimeError("Encoders not fitted. Call fit_encoders() first.")
        
        return {
            "severity_encoded": int(
                self.severity_encoder.transform([incident.get('severity', 'Medium')])[0]
            ),
            "type_encoded": int(
                self.type_encoder.transform([incident.get('incident_type', 'Unknown')])[0]
            ),
        }
    
    #pipeline
    def preprocess(self, incident: dict[str, Any]) -> dict[str, Any]:
        """
        Run the full preprocessing pipeline on a single incident.

        """
        # Build raw text
        raw_text = self.build_incident_text(incident)
        
        # Clean the text

        cleaned_text = self.clean_log_line(raw_text)
        

        # Create chunks
        chunks = self.chunk_text(cleaned_text)
        
        #encode features
        features = self.encode_features(incident)
        
        # Combine
        preprocessed = {
            **incident, "raw_text": raw_text,
            "cleaned_text": cleaned_text,  "text_chunks": chunks,
            "num_chunks": len(chunks), "encoded_features": features,
        }
        
        return preprocessed


def preprocess_incidents(
    incidents: list[dict[str, Any]],
    fit_encoders: bool = True,
) -> list[dict[str, Any]]:
    """
    Convenience function: preprocess a list of incidents
    """
    preprocessor = IncidentPreprocessor()
    
    if fit_encoders:
        preprocessor.fit_encoders(incidents)
    
    preprocessed = [preprocessor.preprocess(inc) for inc in incidents]
    print(f"Preprocessed: {len(preprocessed)} incidents")
    
    return preprocessed


#cli
def main() -> None:
    """Load raw incidents, preprocess, and save."""
    import argparse
    
    parser = argparse.ArgumentParser(description="preprocess K8S incident data")

    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"error: input file not found: {input_path}")
        return
    
    with open(input_path, "r") as f:
        raw_incidents = json.load(f)
    
    print(f"loaded {len(raw_incidents)} raw incidents")
    
    preprocessed = preprocess_incidents(raw_incidents, fit_encoders=True)
    
    with open(output_path, "w") as f:
        json.dump(preprocessed, f, indent=2)
    
    print(f"Saved preprocessed dataset to: {output_path}")
    print(f"size: {output_path.stat().st_size / 1024:.1f} KB")

#main method
if __name__ == "__main__":
    main()
