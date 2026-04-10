"""
UNICC AI Safety Lab — Project 2 | Project 3 Handoff
src/api.py

Flask REST API that wraps the Council of Experts orchestrator.
This is the connection point between Project 2 (evaluation engine)
and Project 3 (user interface).

Endpoints:
    POST /evaluate   — run full council evaluation on an agent
    GET  /health     — check if the lab is running
    GET  /rules      — get the full governance rules database
    GET  /report/<id> — retrieve a past evaluation by report hash

Usage (run on DGX Spark or local):
    pip install flask flask-cors
    python src/api.py

Project 3 connects to:
    http://localhost:5050
"""

import json, logging, traceback
from datetime import datetime
from pathlib import Path
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from slm_client import SLMClient
from scoring_expert import ScoringExpert
from governance_expert import GovernanceExpert
from redteam_expert import RedTeamExpert
from orchestrator import CouncilOrchestrator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow Project 3 frontend to call this API from any origin

# ── Global state ─────────────────────────────────────────────────────────────
# Model loaded once at startup, reused for all requests
_client       = None
_orchestrator = None
_report_store = {}   # in-memory store: report_hash -> CouncilReport dict
_start_time   = datetime.utcnow()

# ── Config ───────────────────────────────────────────────────────────────────
USE_MOCK      = False   # Set False on DGX Spark with real adapters
PORT          = 5050
RULES_PATH    = "data/processed/governance_rules.json"


def get_orchestrator():
    """Lazy-load the orchestrator on first request."""
    global _client, _orchestrator
    if _orchestrator is None:
        log.info("Initializing Council of Experts...")
        _client = SLMClient(use_mock=USE_MOCK)
        _client.load()
        scoring    = ScoringExpert(_client)
        governance = GovernanceExpert(_client)
        redteam    = RedTeamExpert(_client)
        _orchestrator = CouncilOrchestrator(
            scoring_expert    = scoring,
            governance_expert = governance,
            redteam_expert    = redteam,
            run_parallel      = False,   # sequential for stability
        )
        log.info("Council of Experts ready")
    return _orchestrator


def api_response(data=None, error=None, status=200):
    """Standardized API response wrapper."""
    if error:
        return jsonify({"success": False, "error": error}), status
    return jsonify({"success": True, "data": data}), status


def require_json(f):
    """Decorator — returns 400 if request body is not valid JSON."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return api_response(error="Request must be JSON (Content-Type: application/json)", status=400)
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """
    GET /health
    Project 3 calls this to check if the lab is running before sending evaluations.

    Response:
    {
        "success": true,
        "data": {
            "status": "ok",
            "mode": "mock" | "live",
            "uptime_seconds": 123,
            "evaluations_run": 5,
            "modules": ["module_a_scoring", "module_b_governance", "module_c_redteam"]
        }
    }
    """
    uptime = (datetime.utcnow() - _start_time).seconds
    return api_response({
        "status":           "ok",
        "mode":             "mock" if USE_MOCK else "live",
        "uptime_seconds":   uptime,
        "evaluations_run":  len(_report_store),
        "modules": [
            "module_a_scoring",
            "module_b_governance",
            "module_c_redteam"
        ]
    })


@app.route("/evaluate", methods=["POST"])
@require_json
def evaluate():
    """
    POST /evaluate
    Run the full 5-phase Council of Experts evaluation on an agent.

    Request body (JSON):
    {
        "transcript": "User: ...\nAgent: ...",   REQUIRED
        "agent_id":   "my-agent-v1",             REQUIRED
        "context":    "UN field operations"      optional
    }

    Response:
    {
        "success": true,
        "data": {
            "final_verdict":         "FAIL",
            "final_score":           2.836,
            "final_confidence":      0.737,
            "recommendation":        "ESCALATE",
            "requires_human_review": true,
            "disagreement_level":    "MEDIUM",
            "critique_triggered":    true,
            "module_verdicts": {
                "module_a_scoring":    "FAIL",
                "module_b_governance": "FAIL",
                "module_c_redteam":    "WARN"
            },
            "module_scores": { ... },
            "disagreements":       [ "ScoringExpert says FAIL but RedTeamExpert says WARN" ],
            "all_governance_flags": [ "NIST-600-1-HAL-01", ... ],
            "critical_flags":       [],
            "synthesis_rationale":  "The Council of Experts evaluated...",
            "agent_id":             "my-agent-v1",
            "input_hash":           "abc123...",
            "report_hash":          "def456...",
            "timestamp_utc":        "2026-03-24T19:00:00Z",
            "processing_time_ms":   1243
        }
    }
    """
    body = request.get_json()

    # Validate required fields
    if not body.get("transcript"):
        return api_response(error="'transcript' field is required", status=400)
    if not body.get("agent_id"):
        return api_response(error="'agent_id' field is required", status=400)

    agent_input = {
        "transcript": body["transcript"],
        "agent_id":   body["agent_id"],
        "context":    body.get("context", "General UN operational context"),
    }

    log.info(f"POST /evaluate — agent_id={agent_input['agent_id']}")

    try:
        orchestrator = get_orchestrator()
        report       = orchestrator.evaluate(agent_input)
        report_dict  = report.to_dict()

        # Store report for retrieval by Project 3
        _report_store[report.report_hash] = report_dict

        log.info(f"Evaluation complete — verdict={report.final_verdict} "
                 f"score={report.final_score} "
                 f"hash={report.report_hash[:12]}...")

        return api_response(report_dict)

    except Exception as e:
        log.error(f"Evaluation failed: {e}\n{traceback.format_exc()}")
        return api_response(error=f"Evaluation failed: {str(e)}", status=500)


@app.route("/report/<report_hash>", methods=["GET"])
def get_report(report_hash):
    """
    GET /report/<report_hash>
    Retrieve a previously run evaluation report by its hash.
    Project 3 uses this to display past evaluations.

    Response: same schema as /evaluate
    """
    report = _report_store.get(report_hash)
    if not report:
        return api_response(error=f"Report not found: {report_hash}", status=404)
    return api_response(report)


@app.route("/reports", methods=["GET"])
def list_reports():
    """
    GET /reports
    List all evaluation report hashes and their key metadata.
    Project 3 uses this to show evaluation history.

    Response:
    {
        "success": true,
        "data": {
            "total": 3,
            "reports": [
                {
                    "report_hash":    "abc123...",
                    "agent_id":       "my-agent-v1",
                    "final_verdict":  "FAIL",
                    "final_score":    2.836,
                    "timestamp_utc":  "2026-03-24T19:00:00Z"
                },
                ...
            ]
        }
    }
    """
    summaries = []
    for report_hash, report in _report_store.items():
        summaries.append({
            "report_hash":   report_hash,
            "agent_id":      report.get("agent_id"),
            "final_verdict": report.get("final_verdict"),
            "final_score":   report.get("final_score"),
            "timestamp_utc": report.get("timestamp_utc"),
            "recommendation": report.get("recommendation"),
        })
    # Sort by timestamp, newest first
    summaries.sort(key=lambda x: x.get("timestamp_utc", ""), reverse=True)
    return api_response({"total": len(summaries), "reports": summaries})


@app.route("/rules", methods=["GET"])
def get_rules():
    """
    GET /rules
    Return the full governance rules database.
    Project 3 uses this to display rule descriptions alongside flags.

    Optional query params:
        ?framework=NIST AI 600-1   filter by framework
        ?severity=CRITICAL         filter by severity

    Response:
    {
        "success": true,
        "data": {
            "total": 38,
            "rules": [ { rule_id, framework, category, severity, description, ... } ]
        }
    }
    """
    rules_path = Path(RULES_PATH)
    if not rules_path.exists():
        return api_response(
            error=f"Governance rules not found at {RULES_PATH}. Run: python scripts/02_governance_rules.py",
            status=503
        )

    with open(rules_path) as f:
        rules = json.load(f)

    # Optional filtering
    framework = request.args.get("framework")
    severity  = request.args.get("severity")

    if framework:
        rules = [r for r in rules if r.get("framework", "").lower() == framework.lower()]
    if severity:
        rules = [r for r in rules if r.get("severity", "").upper() == severity.upper()]

    return api_response({"total": len(rules), "rules": rules})


@app.route("/rules/<rule_id>", methods=["GET"])
def get_rule(rule_id):
    """
    GET /rules/<rule_id>
    Get details for one specific governance rule.
    Project 3 uses this to show full rule info when a flag is clicked.
    """
    rules_path = Path(RULES_PATH)
    if not rules_path.exists():
        return api_response(error="Governance rules not found", status=503)

    with open(rules_path) as f:
        rules = json.load(f)

    rule = next((r for r in rules if r["rule_id"] == rule_id), None)
    if not rule:
        return api_response(error=f"Rule not found: {rule_id}", status=404)

    return api_response(rule)


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return api_response(error="Endpoint not found", status=404)

@app.errorhandler(405)
def method_not_allowed(e):
    return api_response(error="Method not allowed", status=405)

@app.errorhandler(500)
def internal_error(e):
    return api_response(error="Internal server error", status=500)


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" UNICC AI Safety Lab — Council of Experts API")
    print("="*60)
    print(f" Mode:     {'MOCK (no GPU needed)' if USE_MOCK else 'LIVE (DGX Spark)'}")
    print(f" Port:     {PORT}")
    print(f" Base URL: http://localhost:{PORT}")
    print()
    print(" Endpoints for Project 3:")
    print(f"   POST http://localhost:{PORT}/evaluate")
    print(f"   GET  http://localhost:{PORT}/health")
    print(f"   GET  http://localhost:{PORT}/rules")
    print(f"   GET  http://localhost:{PORT}/reports")
    print(f"   GET  http://localhost:{PORT}/report/<hash>")
    print("="*60 + "\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)
