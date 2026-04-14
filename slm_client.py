"""
UNICC AI Safety Lab — Project 2
src/slm_client.py

LLM Client — supports Claude API (production) and mock mode (testing).

Usage:
    # Set environment variable before running:
    export ANTHROPIC_API_KEY=your_key_here

    # Or mock mode (no API key needed):
    export MOCK_MODE=1
"""

import os, json, re, logging
import time 
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 1024
DEFAULT_MODEL      = "claude-sonnet-4-20250514"


class SLMClient:
    """
    LLM inference client.
    - If MOCK_MODE=1 → returns simulated outputs (no API key needed)
    - Otherwise → uses Claude API via ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, use_mock: bool = None):
        if use_mock is None:
            use_mock = os.environ.get("MOCK_MODE", "0") == "1"
        self.use_mock = use_mock
        self._client  = None

    def load(self):
        if self.use_mock:
            log.info("SLMClient: MOCK MODE (set MOCK_MODE=1 or unset to use Claude API)")
            return

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "Set it with: export ANTHROPIC_API_KEY=your_key_here\n"
                "Or run in mock mode: export MOCK_MODE=1"
            )

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            log.info(f"SLMClient: Claude API ready (model={DEFAULT_MODEL})")
        except ImportError:
            raise ImportError("Run: pip install anthropic")

    def generate(self, messages: list, adapter_name: str = None,
                 max_new_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        if self.use_mock:
            return self._mock_response(adapter_name)

        if not self._client:
            raise RuntimeError("Call client.load() first")

        # Separate system prompt from user/assistant messages
        system_prompt = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                filtered.append(m)

        try:
            time.sleep(2)  
            response = self._client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=max_new_tokens,
                system=system_prompt,
                messages=filtered,
                temperature=0, 
            )
            return response.content[0].text
        except Exception as e:
            log.error(f"Claude API error: {e}")
            raise

    def health_check(self) -> bool:
        if self.use_mock:
            return True
        return self._client is not None

    def _mock_response(self, adapter_name: str) -> str:
        import random
        random.seed(42)
        if adapter_name == "adapter_scoring":
            return json.dumps({
                "verdict": "WARN", "composite_score": 2.8, "confidence": 0.85,
                "dimensions": {
                    "harmfulness":      {"score": 2.5, "rationale": "Mock: moderate risk detected."},
                    "bias_fairness":    {"score": 1.2, "rationale": "Mock: minimal bias detected."},
                    "transparency":     {"score": 3.1, "rationale": "Mock: transparency concerns."},
                    "legal_compliance": {"score": 2.0, "rationale": "Mock: review recommended."},
                    "manipulation":     {"score": 1.5, "rationale": "Mock: low manipulation risk."},
                    "self_preservation":{"score": 0.8, "rationale": "Mock: aligned with user interests."},
                },
                "governance_flags": ["NIST-600-1-HAL-01"]
            }, indent=2)
        elif adapter_name == "adapter_governance":
            return json.dumps({
                "rule_id": "NIST-600-1-HAL-01", "violated": True,
                "severity": "MEDIUM", "verdict": "WARN",
                "evidence": "Mock: agent produces unverified outputs.",
                "rationale": "Mock: hallucination risk identified.",
                "recommendation": "Add output verification layer."
            }, indent=2)
        else:
            return json.dumps({
                "attack_prompt": "Mock adversarial probe.", "attack_technique": "role_play",
                "harm_category": "Mock category", "target_response": "Mock response.",
                "refusal_quality": 3.5, "refusal_verdict": "WEAK_REFUSAL",
                "judge_rationale": "Mock: weak refusal detected.",
                "safety_verdict": "WARN"
            }, indent=2)

    @staticmethod
    def parse_json_output(raw: str):
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        clean = re.sub(r"```(?:json)?\s*", "", raw)
        clean = re.sub(r"```\s*", "", clean).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        log.warning("Could not parse JSON from LLM output")
        return None
