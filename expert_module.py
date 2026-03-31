"""
UNICC AI Safety Lab — Project 2 | Step 4
src/expert_module.py

Abstract base class for all three expert modules.
Defined in the Step 1 Module Specification Document (Section 4.2).

Every expert module MUST implement:
  - assess(agent_input) -> SafetyVerdict
  - health_check() -> bool
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib, json, datetime


@dataclass
class SafetyVerdict:
    """
    Standardized output schema for all three expert modules.
    Defined in Step 1 Module Specification Document Section 4.1.
    """
    module_id:        str
    verdict:          str            # PASS | WARN | FAIL
    composite_score:  float          # 0.0 – 5.0
    confidence:       float          # 0.0 – 1.0
    dimensions:       dict           # per-dimension scores + rationales
    governance_flags: list           # triggered rule IDs
    input_hash:       str            # SHA-256 of raw input
    output_hash:      str            # SHA-256 of this verdict JSON
    timestamp_utc:    str            # ISO 8601
    model_version:    str            # SLM checkpoint identifier
    raw_output:       str            # full untruncated model output

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    @property
    def flagged(self) -> bool:
        return len(self.governance_flags) > 0


def hash_input(data: dict) -> str:
    """SHA-256 hash of the input payload for audit trail integrity."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def hash_output(verdict_dict: dict) -> str:
    """
    SHA-256 hash of the SafetyVerdict JSON (excluding output_hash field).
    Prevents post-hoc alteration of evaluation records.
    """
    # Exclude output_hash from the hash computation to avoid circularity
    d = {k: v for k, v in verdict_dict.items() if k != "output_hash"}
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def utc_now() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class ExpertModule(ABC):
    """
    Abstract base class for all Council of Experts modules.

    Subclasses: ScoringExpert, GovernanceExpert, RedTeamExpert
    """

    def __init__(self, module_id: str, model_version: str = "llama-3.1-8b-instruct"):
        self.module_id     = module_id
        self.model_version = model_version
        self._loaded       = False

    @abstractmethod
    def assess(self, agent_input: dict) -> SafetyVerdict:
        """
        Evaluate an AI agent input and return a structured safety verdict.

        Args:
            agent_input: dict with keys:
                - "transcript": str — the agent conversation to evaluate
                - "agent_id":   str — identifier for the agent being tested
                - "context":    str — optional deployment context

        Returns:
            SafetyVerdict — fully populated verdict with hashes
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Verify SLM endpoint is reachable and adapter is loaded."""
        ...

    def _build_verdict(
        self,
        agent_input:      dict,
        verdict:          str,
        composite_score:  float,
        confidence:       float,
        dimensions:       dict,
        governance_flags: list,
        raw_output:       str,
    ) -> SafetyVerdict:
        """
        Helper that assembles a SafetyVerdict with integrity hashes.
        Called by each subclass after parsing model output.
        """
        input_h = hash_input(agent_input)

        # Build partial dict to compute output hash
        partial = {
            "module_id":        self.module_id,
            "verdict":          verdict,
            "composite_score":  composite_score,
            "confidence":       confidence,
            "dimensions":       dimensions,
            "governance_flags": governance_flags,
            "input_hash":       input_h,
            "timestamp_utc":    utc_now(),
            "model_version":    self.model_version,
            "raw_output":       raw_output,
        }
        output_h = hash_output(partial)

        return SafetyVerdict(
            module_id        = self.module_id,
            verdict          = verdict,
            composite_score  = round(composite_score, 3),
            confidence       = round(confidence, 3),
            dimensions       = dimensions,
            governance_flags = governance_flags,
            input_hash       = input_h,
            output_hash      = output_h,
            timestamp_utc    = partial["timestamp_utc"],
            model_version    = self.model_version,
            raw_output       = raw_output,
        )

    def __repr__(self):
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(id={self.module_id}, model={self.model_version}, {status})"
