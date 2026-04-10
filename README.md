# UNICC AI Safety Lab — Project 2
## Council of Experts: Fine-Tuning the SLM and Building the Multi-Module Ensemble

**Student:** Yiwei Zhu | NetID: yz9422 | NYU SPS MASY GC-4100 | Spring 2026  
**Sponsor:** Dr. Andrés Fortino (NYU) / Ms. Anusha Dandapani (UNICC)

---

## Quick Start (Evaluator Instructions)

### Step 1 — Clone the repository
```bash
git clone https://github.com/zhuyiweisally-ux/UNICC
cd UNICC
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set your API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Step 4 — Run evaluation against VeriMedia
```bash
python3 main.py --agent_url https://github.com/FlashCarrot/VeriMedia
```

### Step 4 (alternative) — Mock mode, no API key needed
```bash
python3 main.py --agent_url https://github.com/FlashCarrot/VeriMedia --mock
# or
MOCK_MODE=1 python3 main.py --agent_url https://github.com/FlashCarrot/VeriMedia
```
### Expected output
════════════════════════════════════════════════
UNICC AI SAFETY LAB — COUNCIL OF EXPERTS REPORT
Agent:          VeriMedia
Final Verdict:  WARN
Safety Score:   2.4/5.0
Recommendation: REVIEW
════════════════════════════════════════════════

### Save output as JSON
```bash
python3 main.py --agent_url https://github.com/FlashCarrot/VeriMedia --output report.json
```

---

## What This Project Does

This is **Project 2** of the UNICC AI Safety Lab — the core evaluation engine. It implements a **Council of Experts**: three independent AI modules that each evaluate the same AI agent from a different perspective, then synthesize a final safety verdict.

### The Three Expert Modules

| Module | Role | Framework |
|--------|------|-----------|
| **ScoringExpert** | Rates harm 0–5 across 6 dimensions | EU AI Act + NIST AI RMF |
| **GovernanceExpert** | Checks against 38 governance rules | NIST AI 600-1, IMDA 2025, OWASP, MITRE, UN standards |
| **RedTeamExpert** | Tests adversarial robustness (6 attack techniques) | HarmBench methodology |

### The 5-Phase Council Pipeline

```
Agent input (GitHub URL or description)
    ↓
Phase 1 — Parallel assessment (3 modules evaluate independently)
    ↓
Phase 2 — Disagreement detection (flags conflicts between modules)
    ↓
Phase 3 — Critique round (modules see peer verdicts and reconsider)
    ↓
Phase 4 — Weighted arbitration (Governance 40%, Scoring 35%, RedTeam 25%)
    ↓
Phase 5 — Synthesis (structured CouncilReport with APPROVE/REVIEW/REJECT)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (live mode) | Your Anthropic API key |
| `MOCK_MODE` | No | Set to `1` to skip API calls and return simulated outputs |

---

## Project Structure

```
UNICC/
├── main.py                    ← Entry point — run this
├── requirements.txt           ← pip install -r requirements.txt
├── expert_module.py           ← Abstract base class + SafetyVerdict schema
├── slm_client.py              ← LLM client (Claude API + mock mode)
├── scoring_expert.py          ← ScoringExpert module
├── governance_expert.py       ← GovernanceExpert module
├── redteam_expert.py          ← RedTeamExpert module
├── orchestrator.py            ← Council of Experts orchestrator
├── api.py                     ← Flask REST API (for Project 3 UI)
├── governance_rules.json      ← 38 governance rules database
├── 11_test_modules.py     ← Test all 3 modules (mock mode)
└── 12_test_council.py     ← End-to-end council test (mock mode)
```
> **Note for evaluators:** Scripts `01_download_datasets.py` through 
> `10_verify_adapters.py` are data pipeline and fine-tuning scripts 
> for the DGX Spark cluster. To run the evaluation, use only: 
> `python3 main.py --agent_url https://github.com/FlashCarrot/VeriMedia`
---

## Flask API (for Project 3 Integration)

```bash
export ANTHROPIC_API_KEY=your_key_here
python3 api.py
```

Then POST to `http://localhost:5050/evaluate`:
```json
{
  "agent_id": "VeriMedia",
  "transcript": "...",
  "context": "UN deployment evaluation"
}
```

---

## No API Keys in This Repository

All API keys are read from environment variables. No secrets are hardcoded anywhere in this codebase.

---

*NYU School of Professional Studies | MS in Management and Systems | Spring 2026*
