"""
TuneKit - Automated LLM Fine-Tuning Pipeline
=============================================

A LangGraph-powered workflow that automates the entire fine-tuning process:
- Ingest data (CSV, JSON, JSONL)
- Validate quality
- Analyze dataset and detect task type
- Select optimal model and training config
- (Coming soon) Estimate cost, human review, training, monitoring
"""

from .state import TuneKitState
from .tools import (
    ingest_data,
    validate_quality,
    analyze_dataset,
    recommend_model,
    generate_package,
    enrich_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "TuneKitState",
    "ingest_data",
    "validate_quality",
    "analyze_dataset",
    "recommend_model",
    "generate_package",
    "enrich_dataset",
]
