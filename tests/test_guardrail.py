"""Unit tests for the guardrail (agent.check_guardrail and config.GUARDRAIL_RULES).

The guardrail only looks at the action name and the order value, so most tests
need no data. The last one uses a real open order from the test pipeline,
because the (mock) approval request must find the order.
"""

import pytest

from src import agent, config
from src.audit import AuditLog, read_audit_log

MANAGER = config.GUARDRAIL_RULES["manager_role"]
DIRECTOR = config.GUARDRAIL_RULES["director_role"]


@pytest.mark.parametrize("action, value, decision, approver", [
    # Automatic actions
    ("send_supplier_reminder", 500, "allowed", None),
    ("read_order", 500, "allowed", None),
    ("monitor", 500, "allowed", None),
    # Blocked list, whatever the value
    ("delete_supplier", 10, "blocked", None),
    ("modify_payment_terms", 10, "blocked", None),
    ("delete_supplier", 50000, "blocked", None),
    # Value tiers of a sensitive action
    ("cancel_order", 500, "allowed", None),
    ("cancel_order", 5000, "needs_approval", MANAGER),
    ("cancel_order", 15000, "needs_approval", DIRECTOR),
    ("replace_supplier", 11642.96, "needs_approval", DIRECTOR),
    ("reschedule_delivery", 6045.86, "needs_approval", MANAGER),
])
def test_each_outcome_and_value_tier(action, value, decision, approver):
    result = agent.check_guardrail(action, value)
    assert (result.decision, result.approver) == (decision, approver)
    assert result.rule_fired  # always explains which rule decided


@pytest.mark.parametrize("value, decision, approver", [
    (999.99, "allowed", None),
    (1000, "needs_approval", MANAGER),       # 1,000 is already in the Manager tier
    (1000.01, "needs_approval", MANAGER),
    (9999.99, "needs_approval", MANAGER),
    (10000, "needs_approval", MANAGER),      # 10,000 is still the Manager tier
    (10000.01, "needs_approval", DIRECTOR),  # above 10,000 -> Director
])
def test_boundary_values(value, decision, approver):
    result = agent.check_guardrail("cancel_order", value)
    assert (result.decision, result.approver) == (decision, approver)


def test_unknown_action_is_blocked_by_default():
    result = agent.check_guardrail("transfer_money", 1)
    assert result.decision == "blocked"
    assert "default-deny" in result.rule_fired


def test_large_order_makes_an_automatic_action_need_approval():
    limit = config.GUARDRAIL_RULES["automatic_value_limit"]
    assert agent.check_guardrail("send_supplier_reminder", limit).decision == "allowed"
    assert agent.check_guardrail("send_supplier_reminder", limit + 0.01).approver == MANAGER
    assert agent.check_guardrail("monitor", limit * 10).decision == "allowed"  # exempt: changes nothing


def test_pending_then_rejected_action_is_logged_and_never_executed(pipeline, tmp_path):
    """Pending means nothing happens; only an explicit Reject is logged as a refusal."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    scored = pipeline["scored"]
    order = scored[scored["order_value"].between(1000, 10000)].iloc[0]  # Manager tier
    action = {"action": "cancel_order", "reason": "test"}
    narrative = {"reminder_email": ""}

    outcome = agent.handle_action(action, order, narrative, lambda question: None, audit)
    assert outcome["status"] == "pending" and outcome["executed"] is False

    entries = read_audit_log(audit.path)
    decision = next(e for e in entries if e["event"] == "guardrail_decision")["details"]
    assert decision["decision"] == "needs_approval" and decision["approver"] == MANAGER
    assert decision["rule_fired"] and decision["simulated"] is True
    assert not any(e["event"] == "approval_denied" for e in entries)

    agent.resolve_approval(outcome, False, "supplier already confirmed", order, narrative, audit)
    denied = [e for e in read_audit_log(audit.path) if e["event"] == "approval_denied"]
    assert outcome["status"] == "rejected" and outcome["executed"] is False
    assert denied[0]["details"]["reason"] == "supplier already confirmed"
