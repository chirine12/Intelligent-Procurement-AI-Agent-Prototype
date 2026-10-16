"""Procurement risk agent: an explicit, step-by-step workflow.

    1. Detect risks            (risk engine scores every open order)
    2. Prioritize              (orders ranked by priority_score)
    3. Select                  (top N orders)
    For each selected order:
    4. Investigate             (mock SAP tools)
    5. Analyze                 (risk factors from the tool results)
    6. Explain and recommend   (actions decided by rules, wording by LLM or template)
    7. Check the guardrail     (automatic, approval required, or blocked)
    8. Ask for human approval  (when required)
    9. Execute and log         (simulated action + audit log)

The LLM never chooses or authorises an action: actions are decided in step 6
by decide_actions() and gated in step 7 by check_guardrail(), whose rules
are all in config.GUARDRAIL_RULES.
"""

from dataclasses import asdict, dataclass

import pandas as pd

from src import config, llm, risk_engine
from src import mock_sap_api as sap
from src.audit import AuditLog

# Actions that have a simulated SAP tool. "replace_supplier" has none on
# purpose: even when approved, it is handed over to a human buyer.
ACTION_TOOLS = {
    "send_supplier_reminder": sap.send_supplier_reminder,
    "reschedule_delivery": sap.reschedule_delivery,
}

# The three guardrail decisions.
ALLOWED, NEEDS_APPROVAL, BLOCKED = "allowed", "needs_approval", "blocked"


# --- Human approval --------------------------------------------------------

def terminal_approver(question):
    """Ask the user in the terminal. Only an explicit 'y' or 'yes' approves."""
    try:
        answer = input(f"{question} [y/N]: ")
    except EOFError:  # no terminal available: treat as a refusal
        return {"approved": False, "reason": "no terminal available"}
    # Input piped from PowerShell starts with an invisible byte-order mark.
    answer = answer.replace("﻿", "").strip().lower()
    return {"approved": answer in ("y", "yes"), "reason": f"terminal answer: '{answer}'"}


def deny_all_approver(question):
    """Approver for unattended runs: every sensitive action is refused."""
    return {"approved": False, "reason": "unattended run (--approval deny)"}


# --- Steps 1-3: detect, prioritize, select ---------------------------------

def detect_and_prioritize():
    """Score all open orders and return them ranked by priority."""
    open_orders = pd.read_csv(config.PROCESSED_DIR / "open_orders.csv", dtype={"po_id": str})
    return risk_engine.score_open_orders(open_orders)


# --- Step 4: investigate ---------------------------------------------------

def call_tool(audit, po_id, tool, *args, **kwargs):
    """Call a mock SAP tool, log the call, and raise if the tool reports an error."""
    result = tool(*args, **kwargs)
    audit.record("tool_call", po_id, tool=tool.__name__, arguments=list(args),
                 success=result["success"], error=result.get("error"))
    if not result["success"]:
        raise RuntimeError(f"{tool.__name__} failed: {result['error']}")
    return result["data"]


def investigate(order, audit):
    """Retrieve the facts about one order with the mock SAP tools."""
    po_id = order["po_id"]
    purchase_order = call_tool(audit, po_id, sap.get_purchase_order, po_id)
    supplier = call_tool(audit, po_id, sap.get_supplier_performance, purchase_order["supplier_id"])
    stock = call_tool(audit, po_id, sap.get_stock_level, purchase_order["material"])

    # Alternatives are only worth looking up when the current supplier is unreliable.
    alternatives = []
    if not supplier["is_reliable"]:
        found = call_tool(audit, po_id, sap.find_alternative_suppliers,
                          purchase_order["material"], exclude_supplier_id=supplier["supplier_id"])
        alternatives = found["alternatives"]

    return {"purchase_order": purchase_order, "supplier": supplier, "stock": stock,
            "alternatives": alternatives}


# --- Steps 5-6: analyze, decide, explain -----------------------------------

def decide_actions(order, findings):
    """Choose the recommended actions with simple, deterministic rules.

    Returns a list of {"action", "reason"}.
    """
    if order["risk_level"] == "Low":
        return [{"action": "monitor", "reason": "The risk level is low."}]

    actions = [{
        "action": "send_supplier_reminder",
        "reason": "The delivery date needs to be confirmed by the supplier.",
    }]
    if order["risk_level"] == "High":
        reliable = [a for a in findings["alternatives"] if a["is_reliable"]]
        if findings["stock"]["below_safety_stock"] and not findings["supplier"]["is_reliable"] and reliable:
            best = reliable[0]
            actions.append({
                "action": "replace_supplier",
                "reason": (f"Stock is below safety stock and the supplier is unreliable; "
                           f"{best['name']} delivers {best['on_time_rate']:.0%} on time."),
            })
        else:
            actions.append({
                "action": "reschedule_delivery",
                "reason": "The promised date is no longer realistic; planning needs a confirmed date.",
            })
    return actions


def build_facts(order, findings, actions):
    """Everything the explanation may use. Nothing else is given to the LLM."""
    po, supplier, stock = findings["purchase_order"], findings["supplier"], findings["stock"]
    return {
        "purchase_order": {
            "po_id": po["po_id"],
            "material": po["material"],
            "material_description": stock["material_description"],
            "quantity": po["quantity"],
            "order_value_eur": po["order_value"],
            "promised_delivery_date": po["promised_delivery_date"],
            "days_until_delivery": po["days_until_delivery"],
            "is_overdue": po["is_overdue"],
            "criticality_of_material": po["criticality_of_material"],
            "plant": po["plant"],
        },
        "supplier": {
            "name": supplier["name"],
            "on_time_rate": f"{supplier['on_time_rate']:.0%}",
            "avg_delay_days": supplier["avg_delay_days"],
            "past_deliveries": supplier["past_deliveries"],
            "is_reliable": supplier["is_reliable"],
        },
        "stock": {
            "current_stock": stock["current_stock"],
            "safety_stock": stock["safety_stock"],
            "stock_coverage_days": stock["stock_coverage_days"],
            "below_safety_stock": stock["below_safety_stock"],
        },
        "alternative_suppliers": [
            {"name": a["name"], "on_time_rate": f"{a['on_time_rate']:.0%}",
             "is_reliable": a["is_reliable"]}
            for a in findings["alternatives"]
        ],
        "scores": {
            "rank": int(order["rank"]),
            "supplier_late_rate": float(order["supplier_late_rate"]),
            "supplier_late_rate_text": f"{order['supplier_late_rate']:.0%}",
            "observed_status": order["observed_status"],
            "rule_risk_score": int(order["rule_risk_score"]),
            "composite_risk_score": float(order["composite_risk_score"]),
            "risk_level": order["risk_level"],
            "priority_score": float(order["priority_score"]),
        },
        "risk_reasons": order["risk_reasons"].split("; "),
        "recommended_actions": actions,
    }


# --- Step 7: guardrail -----------------------------------------------------

@dataclass(frozen=True)
class GuardrailResult:
    decision: str         # "allowed", "needs_approval" or "blocked"
    rule_fired: str       # the rule that decided, in plain words
    approver: str = None  # role that must approve, or None

    def as_dict(self):
        return asdict(self)


def check_guardrail(action, order_value):
    """Decide how an action may proceed, using config.GUARDRAIL_RULES.

    Rules are applied in a fixed order; an action in no list is blocked.
    """
    rules = config.GUARDRAIL_RULES
    value = f"{order_value:,.2f} EUR"
    if action in rules["blocked_actions"]:
        return GuardrailResult(BLOCKED, f"'{action}' is on the blocked list")

    if action in rules["sensitive_actions"]:
        low, high = rules["sensitive_automatic_below"], rules["manager_approval_up_to"]
        if order_value < low:
            return GuardrailResult(ALLOWED, f"sensitive action, order value {value} is under {low:,} EUR")
        if order_value <= high:
            return GuardrailResult(
                NEEDS_APPROVAL, f"sensitive action, order value {value} is between {low:,} and {high:,} EUR",
                rules["manager_role"])
        return GuardrailResult(NEEDS_APPROVAL, f"sensitive action, order value {value} is over {high:,} EUR",
                               rules["director_role"])

    if action in rules["automatic_actions"]:
        limit = rules["automatic_value_limit"]
        if action not in rules["automatic_limit_exempt"] and order_value > limit:
            return GuardrailResult(
                NEEDS_APPROVAL, f"automatic action, but order value {value} is over {limit:,} EUR",
                rules["manager_role"])
        return GuardrailResult(ALLOWED, f"'{action}' is an automatic action")

    return GuardrailResult(BLOCKED, f"'{action}' is not in any list of allowed actions (default-deny)")


def authorization_label(result):
    """How a guardrail result is worded in explanations."""
    if result.decision == ALLOWED:
        return "automatic"
    if result.decision == NEEDS_APPROVAL:
        return f"requires approval by the {result.approver}"
    return "blocked by policy"


# --- Steps 8-9 for one action ----------------------------------------------

def _read_answer(answer):
    """An approver returns True / False / None (still pending), or a dict
    {"approved": ..., "reason": ...}. Returns (approved, reason)."""
    if isinstance(answer, dict):
        return answer.get("approved"), answer.get("reason") or ""
    return answer, ""


def handle_action(action, order, narrative, approver, audit):
    """Apply the guardrail, ask for approval if needed, execute, and log.

    Returns a dictionary describing what happened to this action. Its
    "status" is executed, monitoring, handed_over, pending, rejected or blocked.
    """
    name, po_id = action["action"], order["po_id"]
    guardrail = check_guardrail(name, order["order_value"])
    outcome = {"action": name, "reason": action["reason"], "guardrail": guardrail.decision,
               "rule_fired": guardrail.rule_fired, "approver": guardrail.approver,
               "approved": None, "approval_id": None, "approval_reason": None,
               "executed": False, "result": None, "status": None}
    audit.record("guardrail_decision", po_id, action=name, decision=guardrail.decision,
                 rule_fired=guardrail.rule_fired, approver=guardrail.approver,
                 reason=action["reason"], simulated=True)

    if guardrail.decision == BLOCKED:
        audit.record("action_blocked", po_id, action=name, reason=guardrail.rule_fired, simulated=True)
        outcome["status"] = "blocked"
        return outcome

    if guardrail.decision == NEEDS_APPROVAL:
        request = call_tool(audit, po_id, sap.create_approval_request, po_id, name)
        outcome["approval_id"] = request["approval_id"]
        audit.record("approval_requested", po_id, action=name, approval_id=request["approval_id"],
                     approver=guardrail.approver, rule_fired=guardrail.rule_fired, simulated=True)
        question = (f"  APPROVAL NEEDED for PO {po_id}: {name} - approver: {guardrail.approver}\n"
                    f"  Rule: {guardrail.rule_fired}\n  Reason: {action['reason']}\n  Approve?")
        approved, reason = _read_answer(approver(question))
        if approved is None:  # no decision yet: nothing happens until a human decides
            outcome["status"] = "pending"
            return outcome
        return resolve_approval(outcome, approved, reason, order, narrative, audit)

    return execute_action(outcome, order, narrative, audit)


def resolve_approval(outcome, approved, reason, order, narrative, audit):
    """Record a human decision on a pending action; execute it if approved."""
    po_id, name = order["po_id"], outcome["action"]
    outcome["approved"], outcome["approval_reason"] = bool(approved), reason
    if not approved:
        audit.record("approval_denied", po_id, action=name, approval_id=outcome["approval_id"],
                     approver=outcome["approver"], reason=reason, simulated=True)
        audit.record("action_blocked", po_id, action=name, reason="approval denied", simulated=True)
        outcome["status"] = "rejected"
        return outcome
    audit.record("approval_granted", po_id, action=name, approval_id=outcome["approval_id"],
                 approver=outcome["approver"], reason=reason, simulated=True)
    return execute_action(outcome, order, narrative, audit)


def execute_action(outcome, order, narrative, audit):
    """Run an authorised action (automatically or after approval). All simulated."""
    po_id, name = order["po_id"], outcome["action"]
    if name in ("monitor", "read_order"):
        outcome["result"] = "No action taken; the order stays under monitoring."
        outcome["status"] = "monitoring"
    elif name not in ACTION_TOOLS:
        outcome["result"] = "Approved and handed over to a buyer; not executed by the agent."
        outcome["status"] = "handed_over"
        audit.record("action_handed_over", po_id, action=name, simulated=True)
    else:
        arguments = {"message": narrative["reminder_email"]} if name == "send_supplier_reminder" else {}
        outcome["result"] = call_tool(audit, po_id, ACTION_TOOLS[name], po_id, **arguments)
        outcome["executed"] = True
        outcome["status"] = "executed"
        audit.record("action_executed", po_id, action=name, simulated=True, result=outcome["result"])
    return outcome


def summarize_outcomes(outcomes):
    """Count actions per status, for the one-line run summary."""
    counts = {"executed": 0, "pending": 0, "rejected": 0, "blocked": 0, "other": 0}
    for outcome in outcomes:
        counts[outcome["status"] if outcome["status"] in counts else "other"] += 1
    return counts


# --- One order, then the whole run -----------------------------------------

def prepare_order(order, use_llm, audit):
    """Steps 4 to 6 for one order: investigate, decide, explain. Nothing is executed.

    Returns (actions, facts, narrative).
    """
    po_id = order["po_id"]
    findings = investigate(order, audit)                           # step 4
    actions = decide_actions(order, findings)                      # steps 5-6
    # Tell the explanation which actions are gated, so the text says so.
    for action in actions:
        action["authorization"] = authorization_label(check_guardrail(action["action"], order["order_value"]))
    facts = build_facts(order, findings, actions)
    narrative = llm.generate_narrative(facts, use_llm)             # step 6 (wording)
    audit.record("recommendation", po_id, actions=actions, scores=facts["scores"],
                 supplier_late_rate=facts["scores"]["supplier_late_rate"],
                 observed_status=facts["scores"]["observed_status"],
                 risk_reasons=facts["risk_reasons"], narrative_source=narrative["source"],
                 fallback_reason=narrative["fallback_reason"],
                 explanation=narrative["explanation"],
                 recommendation=narrative["recommendation"])
    return actions, facts, narrative


def process_order(order, use_llm, approver, audit):
    """Steps 4 to 9 for one order. Returns a result dictionary for the report."""
    po_id = order["po_id"]
    result = {"po_id": po_id, "rank": int(order["rank"]), "status": "ok"}
    try:
        actions, facts, narrative = prepare_order(order, use_llm, audit)

        print_analysis(facts, narrative)
        outcomes = []
        for action in actions:                                     # steps 7-9
            outcome = handle_action(action, order, narrative, approver, audit)
            print_outcome(outcome)
            outcomes.append(outcome)
        result.update(facts=facts, narrative=narrative, outcomes=outcomes)
    except Exception as error:
        # One failing order must not stop the run, and must leave a trace.
        result.update(status="error", error=str(error))
        audit.record("error", po_id, error=str(error))
        print(f"  ERROR while processing PO {po_id}: {error}")
    return result


def run_agent(top_n=config.DEFAULT_TOP_N, use_llm=True, approver=terminal_approver, audit=None):
    """Run the full workflow. Returns (all scored orders, results of the top N)."""
    llm.load_env_file()
    audit = audit or AuditLog()
    scored = detect_and_prioritize()                               # steps 1-2
    selected = scored.head(top_n)                                  # step 3
    audit.record("run_started", top_n=top_n, use_llm=use_llm, open_orders=len(scored),
                 selected_po_ids=list(selected["po_id"]))

    print(f"{len(scored)} open orders scored. Investigating the top {len(selected)}.")
    results = [process_order(order, use_llm, approver, audit) for _, order in selected.iterrows()]

    audit.record("run_finished", processed=len(results),
                 errors=sum(r["status"] == "error" for r in results))
    print(f"\nAudit log: {audit.path} (run {audit.run_id})")
    return scored, results


# --- Console output --------------------------------------------------------

def print_analysis(facts, narrative):
    po, scores = facts["purchase_order"], facts["scores"]
    print(f"\n{'=' * 78}")
    print(f"#{scores['rank']}  PO {po['po_id']}  |  {po['material_description']}  |  "
          f"{facts['supplier']['name']}  |  {po['order_value_eur']:,.2f} EUR")
    print(f"Priority {scores['priority_score']}  |  Composite risk {scores['composite_risk_score']} "
          f"({scores['risk_level']})  |  Supplier late rate {scores['supplier_late_rate']:.0%} "
          f"(statistic)  |  Observed: {scores['observed_status']}  |  Rule score {scores['rule_risk_score']}")
    print("Risk factors:")
    for reason in facts["risk_reasons"]:
        print(f"  - {reason}")
    print(f"Explanation [{narrative['source']}]: {narrative['explanation']}")
    print(f"Recommendation [{narrative['source']}]: {narrative['recommendation']}")
    if narrative["fallback_reason"]:
        print(f"  (template used: {narrative['fallback_reason']})")
    print("Actions:")


def print_outcome(outcome):
    status = {
        "blocked": "BLOCKED by policy",
        "rejected": "NOT EXECUTED (approval denied)",
        "pending": "PENDING approval",
        "executed": "EXECUTED (simulated)",
    }.get(outcome["status"], outcome["result"])
    approval = f"approved by {outcome['approver']}" if outcome["approved"] else outcome["guardrail"]
    print(f"  - {outcome['action']}: {status}  [{approval}]  Rule: {outcome['rule_fired']}")
