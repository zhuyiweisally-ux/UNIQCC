What This Project Does
This project builds the core AI evaluation engine for the UNICC AI Safety Lab — a system that automatically tests AI agents before they are deployed inside the United Nations system.
Instead of using one AI to judge another (which is unreliable), this project implements a Council of Experts: three independent AI modules that each evaluate the same agent from a different angle, compare their findings, and produce one final safety verdict.

The Three Expert Modules
ModuleRoleWhat it checksScoringExpertQuantitative scorerRates harm on a 0–5 scale across 6 safety dimensionsGovernanceExpertCompliance auditorChecks against 38 governance rules (NIST, IMDA, OWASP, MITRE, EU AI Act, UN)RedTeamExpertAdversarial attackerTries to break the agent using 6 attack techniques

The 5-Phase Council Pipeline
Agent input
    ↓
Phase 1 — Parallel assessment (all 3 modules evaluate independently)
    ↓
Phase 2 — Disagreement detection (compare verdicts, flag conflicts)
    ↓
Phase 3 — Critique round (if disagreement, modules see peer verdicts and reconsider)
    ↓
Phase 4 — Weighted arbitration (Governance 40%, Scoring 35%, RedTeam 25%)
    ↓
Phase 5 — Synthesis (CouncilReport with verdict, recommendation, audit hash)
Final output: PASS / WARN / FAIL with recommendation to APPROVE / MONITOR / ESCALATE / BLOCK

Project Structure
project2/
├── src/
│   ├── expert_module.py      # Abstract base class + SafetyVerdict schema
│   ├── slm_client.py         # Shared SLM inference client (mock + live mode)
│   ├── scoring_expert.py     # ScoringExpert — 3-tier rubric scoring
│   ├── governance_expert.py  # GovernanceExpert — 38-rule compliance checker
│   ├── redteam_expert.py     # RedTeamExpert — adversarial robustness testing
│   ├── orchestrator.py       # Council of Experts 5-phase orchestrator
│   └── api.py                # Flask REST API for Project 3 integration
├── scripts/
│   ├── 01_download_datasets.py        # Download HarmBench + PKU-SafeRLHF
│   ├── 02_governance_rules.py         # Build 38-rule governance database
│   ├── 03_format_scoring.py           # Format adapter_scoring training data
│   ├── 04_format_governance.py        # Format adapter_governance training data
│   ├── 05_format_redteam.py           # Format adapter_redteam training data
│   ├── 06_split_data.py               # 90/10 train/validation split
│   ├── 07_finetune_adapter_scoring.py     # QLoRA fine-tune ScoringExpert
│   ├── 08_finetune_adapter_governance.py  # QLoRA fine-tune GovernanceExpert
│   ├── 09_finetune_adapter_redteam.py     # QLoRA fine-tune RedTeamExpert
│   ├── run_all_finetuning.sh          # One-command fine-tuning pipeline
│   ├── 10_verify_adapters.py          # Post-training verification
│   ├── 11_test_modules.py             # Test all 3 modules (mock mode)
│   └── 12_test_council.py             # End-to-end council test (mock mode)
└── data/
    └── processed/
        └── governance_rules.json      # 38 governance rules database

Technology Stack
ComponentTechnologyBase modelLlama 3.1 8B Instruct (Meta, open weights)Fine-tuningQLoRA — 3 separate LoRA adaptersInfrastructureNYU DGX Spark cluster (NVIDIA GPU)FrameworksHuggingFace Transformers, PEFT, TRL, bitsandbytesAPI layerFlask REST APITraining dataHarmBench (314 categories) + PKU-SafeRLHF (73,907 samples)

How to Run (Mock Mode — No GPU Needed)
bash# Install dependencies
pip install datasets flask flask-cors

# Build governance rules database
python scripts/02_governance_rules.py

# Test all three expert modules
python scripts/11_test_modules.py

# Test full council pipeline
python scripts/12_test_council.py

# Start the API server
python src/api.py
Expected output from 12_test_council.py:
RESULTS: 3/3 council tests passed
All council phases working correctly.

How to Run Fine-Tuning (NYU DGX Spark Cluster)
bash# Install GPU dependencies
pip install transformers peft trl bitsandbytes accelerate

# Download training data
python scripts/01_download_datasets.py

# Format all three datasets
python scripts/03_format_scoring.py
python scripts/04_format_governance.py
python scripts/05_format_redteam.py
python scripts/06_split_data.py

# Run all three fine-tuning jobs (estimated 4-6 hours)
./scripts/run_all_finetuning.sh

# Verify adapters
python scripts/10_verify_adapters.py

API Endpoints (Project 3 Integration)
MethodEndpointDescriptionPOST/evaluateRun full council evaluation on an agentGET/healthCheck if API is runningGET/rulesGet all 38 governance rulesGET/reportsList all past evaluationsGET/report/<hash>Retrieve evaluation by report hash
Example request:
jsonPOST /evaluate
{
  "agent_id": "my-agent-v1",
  "transcript": "User: ...\nAgent: ...",
  "context": "UN field operations"
}

Governance Frameworks Covered

NIST AI 600-1 (2024) — 12 GenAI risk categories
IMDA Agentic AI Framework (2025) — 8 agentic evaluation rules
OWASP AI Security Top 10 — 5 security rules
MITRE ATLAS — 4 adversarial ML rules
EU AI Act — 4 compliance rules
UN Humanitarian Standards — 5 UN-specific rules

Total: 38 governance rules

Current Status
StepDescriptionStatusStep 1Module specification & SLM selectionCompleteStep 2Training data pipeline (5,724 samples)CompleteStep 3QLoRA fine-tuning scriptsWritten — pending cluster executionStep 4Expert module Python classesComplete (mock-tested)Step 5Council orchestratorComplete (mock-tested)Step 6PoC validation on UNICC test casesPending fine-tuning

Relationship to Fall 2025 Modules
This project operationalizes three Fall 2025 UNICC capstone solutions:

Module A (Petri Framework) → adapted as ScoringExpert (Claude API replaced with local SLM)
Module B (AI Safety Agent) → adapted as GovernanceExpert (rule database extended with NIST AI 600-1 + IMDA 2025)
Module C (Safety Interface) → UI layer handed off to Project 3


NYU School of Professional Studies | MS in Management and Systems | Spring 2026
