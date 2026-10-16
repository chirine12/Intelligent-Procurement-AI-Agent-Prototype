"""Flask web app for the Intelligent Procurement AI Agent.

Run with:  python app.py        (then open http://localhost:5000)

The app reuses the same code as the CLI (src/). It adds no logic of its own:
scores come from risk_engine, actions from agent.decide_actions, and the
guardrail from agent.check_guardrail (rules in config.GUARDRAIL_RULES).
Approvals are given here instead of the terminal; every sensitive action starts
as "Pending" and nothing happens to it until a human approves or rejects it.
"""

import json
import os
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET

from flask import Flask, Response, abort, redirect, render_template, request, url_for

from src import agent, config, data_processing, generate_data, llm, risk_engine
from src import mock_sap_api as sap
from src.audit import AuditLog, read_audit_log

llm.load_env_file()
app = Flask(__name__)

# Human-readable labels for the field names returned by the mock SAP tools.
FIELD_LABELS = {
    "po_id": "PO number", "supplier_id": "Supplier ID", "material": "Material",
    "material_description": "Description", "quantity": "Quantity",
    "order_value_eur": "Order value (EUR)", "promised_delivery_date": "Promised delivery",
    "days_until_delivery": "Days until delivery", "is_overdue": "Overdue",
    "criticality_of_material": "Material criticality", "plant": "Plant", "name": "Supplier",
    "on_time_rate": "On-time rate", "avg_delay_days": "Average delay (days)",
    "past_deliveries": "Past deliveries", "is_reliable": "Reliable",
    "current_stock": "Current stock", "safety_stock": "Safety stock",
    "stock_coverage_days": "Stock coverage (days)", "below_safety_stock": "Below safety stock",
}
# Boolean fields where "True" is bad news (shown in red) or good news (green).
BAD_WHEN_TRUE = {"is_overdue", "below_safety_stock"}
GOOD_WHEN_TRUE = {"is_reliable"}

GUARDRAIL_LABELS = {"allowed": "Automatic", "needs_approval": "Needs approval", "blocked": "Blocked by policy"}
# Audit events that carry a decision get a tone, so they stand out in the log.
EVENT_TONES = {
    "approval_granted": "good", "action_executed": "good",
    "approval_denied": "bad", "action_blocked": "bad",
    "approval_requested": "warn", "guardrail_decision": "info",
    "recommendation": "info", "action_handed_over": "info",
}

TEST_DESCRIPTIONS = {
    "test_cleaning_handles_every_injected_anomaly":
        "Every injected data-quality issue is found and handled, with exact counts",
    "test_scores_stay_within_their_bounds":
        "All scores stay in range; priority never exceeds risk; composite follows the configured weights",
    "test_prioritization_ranks_by_priority_and_criticality":
        "Ranking is correct; a critical material outranks an identical non-critical one",
    "test_mock_sap_tools_return_structured_and_safe_results":
        "Mock SAP tools return structured results, report errors as data, change nothing",
    "test_sensitive_actions_need_approval_and_work_without_llm":
        "Refused approval means no action; unknown action is blocked; LLM outage uses templates",
    "test_each_outcome_and_value_tier":
        "Guardrail: automatic, blocked and each approval tier (Manager, Director)",
    "test_boundary_values":
        "Guardrail: boundary values around 1,000 and 10,000 EUR",
    "test_unknown_action_is_blocked_by_default":
        "Guardrail: an action in no list is blocked (default-deny)",
    "test_large_order_makes_an_automatic_action_need_approval":
        "Guardrail: an automatic action on a very large order needs approval",
    "test_pending_then_rejected_action_is_logged_and_never_executed":
        "Pending does nothing; an explicit Reject is logged with its reason; nothing executed",
}

# Investigations in progress, kept in memory for this server process only.
INVESTIGATIONS = {}
LAST_TEST_RUN = {}


# --- Helpers -----------------------------------------------------------------

def data_ready():
    return (config.PROCESSED_DIR / "open_orders.csv").exists()


def run_setup():
    """Same as `python -m src.main setup`."""
    generate_data.generate_all()
    data_processing.run()
    sap.reload_data()


def read_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scored_orders():
    return agent.detect_and_prioritize()


def field_rows(data):
    """(label, display text, tone) for each field of a mock SAP tool result."""
    rows = []
    for key, value in data.items():
        tone = ""
        if isinstance(value, bool):
            if key in BAD_WHEN_TRUE and value:
                tone = "bad"
            elif key in GOOD_WHEN_TRUE:
                tone = "good" if value else "bad"
            text = "Yes" if value else "No"
        elif isinstance(value, float):
            text = f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
        else:
            text = "-" if value is None else str(value)
        rows.append((FIELD_LABELS.get(key, key.replace("_", " ").capitalize()), text, tone))
    return rows


def audit_rows(entries):
    return [{
        "time": e["timestamp"][:19].replace("T", " "), "run": e["run_id"], "event": e["event"],
        "tone": EVENT_TONES.get(e["event"], ""), "po": e["po_id"] or "",
        "subject": e["details"].get("action") or e["details"].get("tool") or "",
        "decision": e["details"].get("decision") or "", "approver": e["details"].get("approver") or "",
        "details": json.dumps(e["details"], indent=2, default=str),
    } for e in entries]


@app.context_processor
def globals_for_templates():
    rules = config.GUARDRAIL_RULES
    return {"as_of": config.AS_OF_DATE, "has_llm_key": bool(os.environ.get("GROQ_API_KEY")),
            "tier_low": rules["sensitive_automatic_below"], "tier_high": rules["manager_approval_up_to"]}


@app.before_request
def require_setup():
    if not data_ready() and request.endpoint not in ("setup", "setup_run", "tests", "tests_run", "static"):
        return redirect(url_for("setup"))


# --- Pages -------------------------------------------------------------------

@app.get("/setup")
def setup():
    return render_template("setup.html", page="overview")


@app.post("/setup")
def setup_run():
    run_setup()
    return redirect(url_for("overview", regenerated=1))


@app.get("/")
def overview():
    scored = scored_orders()
    summary = read_json(config.PROCESSED_DIR / "data_quality_summary.json")
    levels = scored["risk_level"].value_counts()

    datasets, issues = [], []
    if summary:
        datasets = [(name.replace("_", " ").capitalize(), c["raw"], c["clean"], c["raw"] - c["clean"])
                    for name, c in summary["row_counts"].items()]
        issues = [(dataset.replace("_", " ").capitalize(), issue.replace("_", " ").capitalize(), count)
                  for dataset in ["purchase_orders", "deliveries_history", "suppliers", "stock_levels"]
                  for issue, count in summary[dataset].items() if count]
    return render_template(
        "overview.html", page="overview", queue=scored.head(5).to_dict("records"),
        total=len(scored), high=int(levels.get("High", 0)), medium=int(levels.get("Medium", 0)),
        low=int(levels.get("Low", 0)), overdue=int(scored["is_overdue"].sum()),
        datasets=datasets, issues=issues, regenerated=request.args.get("regenerated"))


LEVELS = ["High", "Medium", "Low"]
# Risk reasons start with one of these phrases (see risk_engine.apply_rules).
RISK_DRIVERS = [("Overdue", "Overdue"), ("Stock below safety stock", "Stock below safety stock"),
                ("Stock covers", "Stock runs out early"), ("Unreliable supplier", "Unreliable supplier"),
                ("No risk factor", "No risk factor")]


def bars(counts, order=None):
    """[(label, value, percent of the largest value)] for a horizontal bar chart."""
    items = [(k, counts.get(k, 0)) for k in order] if order else sorted(counts.items(), key=lambda i: -i[1])
    top = max((v for _, v in items), default=0) or 1
    return [(k, v, v / top * 100) for k, v in items]


@app.get("/ranking")
def ranking():
    scored = scored_orders()
    chosen = request.args.getlist("level") or LEVELS
    view = scored[scored["risk_level"].isin(chosen)]

    levels = view["risk_level"].value_counts().to_dict()
    drivers = {label: 0 for _, label in RISK_DRIVERS}
    for reasons in view["risk_reasons"]:
        for reason in reasons.split("; "):
            label = next((l for prefix, l in RISK_DRIVERS if reason.startswith(prefix)), None)
            if label:
                drivers[label] += 1
    at_risk = view[view["risk_level"].isin(["High", "Medium"])]
    value_by_category = at_risk.groupby("category")["order_value"].sum().to_dict()
    # Rows of criticality, each split into risk-level segments (percent of the row).
    criticality = []
    for crit in ["High", "Medium", "Low"]:
        rows = view[view["criticality_of_material"] == crit]
        counts = rows["risk_level"].value_counts().to_dict()
        criticality.append((crit, len(rows), [(lvl, counts.get(lvl, 0), counts.get(lvl, 0) / max(len(rows), 1) * 100)
                                              for lvl in LEVELS]))

    return render_template(
        "ranking.html", page="ranking", orders=view.to_dict("records"), top=view.head(15).to_dict("records"),
        chosen=chosen, total=len(scored), shown=len(view),
        levels=[(lvl, levels.get(lvl, 0), levels.get(lvl, 0) / max(len(view), 1) * 100) for lvl in LEVELS],
        overdue=int(view["is_overdue"].sum()), value_at_risk=float(at_risk["order_value"].sum()),
        drivers=[d for d in bars(drivers) if d[1]], categories=bars(value_by_category),
        criticality=criticality)


@app.get("/agent")
def agent_page():
    scored = scored_orders()
    rules = config.GUARDRAIL_RULES
    actions = rules["automatic_actions"] + rules["sensitive_actions"] + rules["blocked_actions"]
    try_action = request.args.get("try_action", actions[0])
    try_value = request.args.get("try_value", 5000, type=int)
    return render_template(
        "agent.html", page="agent", orders=scored.to_dict("records"),
        selected=request.args.get("po", scored.iloc[0]["po_id"]),
        actions=actions, try_action=try_action, try_value=try_value,
        try_result=agent.check_guardrail(try_action, try_value), rules=rules)


@app.post("/agent/investigate")
def investigate():
    scored = scored_orders()
    po_id = request.form["po"]
    match = scored[scored["po_id"] == po_id]
    if match.empty:
        abort(404)
    order = match.iloc[0]
    use_llm = bool(request.form.get("use_llm")) and bool(os.environ.get("GROQ_API_KEY"))
    audit = AuditLog()
    actions, facts, narrative = agent.prepare_order(order, use_llm, audit)
    token = uuid.uuid4().hex[:12]
    INVESTIGATIONS[token] = {"order": order, "actions": actions, "facts": facts,
                             "narrative": narrative, "audit": audit, "outcomes": None}
    return redirect(url_for("investigation", token=token))


def get_investigation(token):
    prepared = INVESTIGATIONS.get(token)
    if prepared is None:
        abort(404, "This investigation is no longer in memory (the server was restarted). Start a new one.")
    return prepared


def read_decisions(form, prepared):
    """{action: {"approved": True/False/None, "reason": str}} from the posted form."""
    choices = {}
    for action in prepared["actions"]:
        name = action["action"]
        choice = form.get(f"decision-{name}", "pending")
        choices[name] = {"approved": {"approve": True, "reject": False}.get(choice),
                         "reason": form.get(f"reason-{name}", "").strip()}
    return choices


@app.get("/agent/<token>")
def investigation(token):
    prepared = get_investigation(token)
    order, facts = prepared["order"], prepared["facts"]
    cards = []
    for action in prepared["actions"]:
        result = agent.check_guardrail(action["action"], order["order_value"])
        outcome = (prepared["outcomes"] or {}).get(action["action"])
        cards.append({"name": action["action"], "reason": action["reason"], "guardrail": result,
                      "label": GUARDRAIL_LABELS[result.decision], "outcome": outcome,
                      "decided": outcome is not None and outcome["status"] != "pending"})
    outcomes = prepared["outcomes"]
    entries, counts = [], None
    if outcomes is not None:
        counts = agent.summarize_outcomes(outcomes.values())
        audit = prepared["audit"]
        entries = audit_rows([e for e in read_audit_log(audit.path) if e["run_id"] == audit.run_id])
    return render_template(
        "investigation.html", page="agent", token=token, facts=facts, narrative=prepared["narrative"],
        po=facts["purchase_order"], scores=facts["scores"], cards=cards, outcomes=outcomes,
        counts=counts, entries=entries, formula=risk_engine.composite_formula(),
        tabs=[("Purchase order", field_rows(facts["purchase_order"])),
              ("Supplier", field_rows(facts["supplier"])), ("Stock", field_rows(facts["stock"]))],
        pending=[o for o in (outcomes or {}).values() if o["status"] == "pending"])


@app.post("/agent/<token>/execute")
def execute(token):
    prepared = get_investigation(token)
    if prepared["outcomes"] is None:
        choices = read_decisions(request.form, prepared)
        order, narrative, audit = prepared["order"], prepared["narrative"], prepared["audit"]
        prepared["outcomes"] = {
            action["action"]: agent.handle_action(
                action, order, narrative,
                lambda question, answer=choices.get(action["action"]): answer, audit)
            for action in prepared["actions"]}
    return redirect(url_for("investigation", token=token) + "#outcome")


@app.post("/agent/<token>/resolve")
def resolve(token):
    prepared = get_investigation(token)
    choices = read_decisions(request.form, prepared)
    for outcome in (prepared["outcomes"] or {}).values():
        answer = choices.get(outcome["action"], {})
        if outcome["status"] == "pending" and answer.get("approved") is not None:
            agent.resolve_approval(outcome, answer["approved"], answer["reason"], prepared["order"],
                                   prepared["narrative"], prepared["audit"])
    return redirect(url_for("investigation", token=token) + "#outcome")


@app.get("/audit")
def audit_page():
    entries = audit_rows(read_audit_log())
    runs = list(dict.fromkeys(reversed([e["run"] for e in entries])))
    run = request.args.get("run", runs[0] if runs else "all")
    in_run = entries if run == "all" else [e for e in entries if e["run"] == run]
    po = request.args.get("po", "all")
    events = request.args.getlist("event")
    view = [e for e in in_run if (po == "all" or e["po"] == po) and (not events or e["event"] in events)]

    if request.args.get("format") == "csv":
        import csv
        import io
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Time (UTC)", "Run", "Event", "PO", "Action / tool", "Decision", "Approver", "Details"])
        for e in view:
            writer.writerow([e["time"], e["run"], e["event"], e["po"], e["subject"], e["decision"],
                             e["approver"], e["details"].replace("\n", " ")])
        return Response(buffer.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=audit_log_filtered.csv"})

    return render_template(
        "audit.html", page="audit", entries=list(reversed(view)), runs=runs, run=run, po=po,
        pos=sorted({e["po"] for e in in_run if e["po"]}), events=events,
        all_events=sorted({e["event"] for e in entries}), path=config.AUDIT_LOG_PATH,
        granted=sum(e["event"] == "approval_granted" for e in view),
        denied=sum(e["event"] == "approval_denied" for e in view))


@app.get("/tests")
def tests():
    return render_template("tests.html", page="tests", descriptions=TEST_DESCRIPTIONS,
                           result=LAST_TEST_RUN.get("result"))


@app.post("/tests")
def tests_run():
    LAST_TEST_RUN["result"] = run_pytest()
    return redirect(url_for("tests") + "#results")


def run_pytest():
    """Run the test suite in a subprocess and read the JUnit XML report."""
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "report.xml")
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "-p", "no:warnings", "-p", "no:cacheprovider",
             f"--junitxml={report}"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True,
        )
        tests = []
        if os.path.exists(report):
            for case in ET.parse(report).getroot().iter("testcase"):
                status = "passed"
                for tag in ("failure", "error", "skipped"):
                    if case.find(tag) is not None:
                        status = "failed" if tag == "failure" else tag
                name = case.get("name")
                tests.append({"name": name, "time": float(case.get("time", 0)), "status": status,
                              "description": TEST_DESCRIPTIONS.get(name.split("[")[0], "")})
    passed = sum(t["status"] == "passed" for t in tests)
    return {"tests": tests, "passed": passed, "output": process.stdout + process.stderr}


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", page="", message=error.description), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
