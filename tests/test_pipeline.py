"""Five tests covering data cleaning, scoring, prioritization, mock SAP tools,
and the protection of sensitive actions (including the no-LLM fallback)."""

import pandas as pd

from src import agent, config, llm, mock_sap_api, risk_engine
from src.audit import AuditLog, read_audit_log
from src.generate_data import ANOMALY_COUNTS


def test_cleaning_handles_every_injected_anomaly(pipeline):
    summary = pipeline["summary"]
    po = pipeline["outputs"]["purchase_orders"]
    rejected = pipeline["outputs"]["rejected_purchase_orders"]

    assert summary["purchase_orders"]["duplicate_rows_removed"] == ANOMALY_COUNTS["po_duplicate_rows"]
    assert summary["deliveries_history"]["duplicate_rows_removed"] == ANOMALY_COUNTS["history_duplicate_rows"]
    assert summary["deliveries_history"]["supplier_names_normalized"] == \
        ANOMALY_COUNTS["history_supplier_name_variants"]
    assert summary["suppliers"]["duplicate_supplier_ids_removed"] == ANOMALY_COUNTS["supplier_duplicate_rows"]

    # Unrepairable orders are rejected with a reason, never kept.
    assert len(rejected) == (ANOMALY_COUNTS["po_unknown_supplier"]
                             + ANOMALY_COUNTS["po_promised_before_order"]
                             + ANOMALY_COUNTS["po_delivered_without_actual_date"])
    assert "S999" not in set(po["supplier_id"])
    assert (po["promised_delivery_date"] >= po["order_date"]).all()

    # Clean orders: one row per PO and no missing values (except the delivery
    # date of orders that have not been delivered yet).
    assert po["po_id"].is_unique
    assert po.drop(columns="actual_delivery_date").isna().sum().sum() == 0
    assert po.loc[po["status"] == "Delivered", "actual_delivery_date"].notna().all()


def test_scores_stay_within_their_bounds(pipeline):
    scored = pipeline["scored"]
    assert scored["supplier_late_rate"].between(0, 1).all()
    # The statistic is the complement of the on-time rate used by the supplier rule.
    assert ((scored["supplier_late_rate"] + scored["supplier_on_time_rate"] - 1).abs() < 0.0015).all()
    for column in ["rule_risk_score", "composite_risk_score", "priority_score"]:
        assert scored[column].between(0, 100).all(), column

    # The priority multiplier is at most 1, so priority never exceeds risk.
    assert (scored["priority_score"] <= scored["composite_risk_score"]).all()
    # Supplier statistic and observed overdue status are separate inputs of the composite.
    w = config.COMPOSITE_WEIGHTS
    assert abs(sum(w.values()) - 1) < 1e-9
    expected = 100 * (w["supplier_late_rate"] * scored["supplier_late_rate"]
                      + w["overdue"] * scored["is_overdue"].astype(int)
                      + w["rules"] * scored["rule_risk_score"] / 100)
    assert ((scored["composite_risk_score"] - expected).abs() <= 0.051).all()  # rounded to 0.1
    assert (scored["observed_status"].str.startswith("Overdue") == scored["is_overdue"]).all()
    assert (scored.loc[~scored["is_overdue"], "observed_status"] == "On track").all()
    assert set(scored["risk_level"]) <= {"High", "Medium", "Low"}

    # With these weights only an overdue order can reach High risk (max 25 + 30 = 55 otherwise).
    assert (scored.loc[scored["risk_level"] == "High", "is_overdue"]).all()


def test_prioritization_ranks_by_priority_and_criticality(pipeline):
    scored = pipeline["scored"]
    assert list(scored["rank"]) == list(range(1, len(scored) + 1))
    assert scored["priority_score"].is_monotonic_decreasing

    # Two identical overdue orders that differ only in material criticality:
    # same risk, but the critical one must be ranked first.
    base = pipeline["outputs"]["open_orders"].iloc[0].to_dict()
    base.update(is_overdue=True, days_until_delivery=-2)
    low = dict(base, po_id="TEST-LOW", criticality_of_material="Low")
    high = dict(base, po_id="TEST-HIGH", criticality_of_material="High")
    ranked = risk_engine.score_open_orders(pd.DataFrame([low, high]))

    assert list(ranked["po_id"]) == ["TEST-HIGH", "TEST-LOW"]
    assert ranked["composite_risk_score"].nunique() == 1


def test_mock_sap_tools_return_structured_and_safe_results(pipeline):
    scored = pipeline["scored"]
    top = scored.iloc[0]

    # Read tool: same numbers as the features used by the risk engine.
    performance = mock_sap_api.get_supplier_performance(top["supplier_id"])
    assert performance["success"] is True
    assert performance["data"]["on_time_rate"] == top["supplier_on_time_rate"]

    # Errors are returned as data, not raised.
    assert mock_sap_api.get_supplier_performance("S999") == {
        "success": False, "error": "Supplier S999 not found"}

    # Alternatives never include the current supplier.
    alternatives = mock_sap_api.find_alternative_suppliers(top["material"], top["supplier_id"])
    assert top["supplier_id"] not in [a["supplier_id"] for a in alternatives["data"]["alternatives"]]

    # Actions are simulated, only allowed on open orders, and change nothing.
    po_file = config.PROCESSED_DIR / "purchase_orders.csv"
    before = po_file.read_bytes()
    result = mock_sap_api.reschedule_delivery(top["po_id"])
    assert result["success"] and result["data"]["simulated"] is True
    assert po_file.read_bytes() == before

    po = pipeline["outputs"]["purchase_orders"]
    delivered_po = po.loc[po["status"] == "Delivered", "po_id"].iloc[0]
    assert mock_sap_api.send_supplier_reminder(delivered_po)["success"] is False


def test_sensitive_actions_need_approval_and_work_without_llm(pipeline, no_api_key, monkeypatch, tmp_path):
    # The guardrail itself is unit-tested in test_guardrail.py.
    assert agent.check_guardrail("delete_supplier", 100).decision == "blocked"

    # Fallback: the LLM is enabled but its call fails -> templates are used.
    def failing_call(facts):
        raise ConnectionError("simulated outage")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "call_groq", failing_call)

    # Run 1: every approval is refused -> no sensitive action is executed.
    denied_log = tmp_path / "denied.jsonl"
    _, results = agent.run_agent(top_n=3, use_llm=True, approver=agent.deny_all_approver,
                                 audit=AuditLog(denied_log))
    assert all(r["status"] == "ok" for r in results)
    assert all(r["narrative"]["source"] == "template" for r in results)
    assert all("simulated outage" in r["narrative"]["fallback_reason"] for r in results)

    entries = read_audit_log(denied_log)
    executed = [e["details"]["action"] for e in entries if e["event"] == "action_executed"]
    assert "reschedule_delivery" not in executed and "replace_supplier" not in executed
    denied = [e for e in entries if e["event"] == "approval_denied"]
    assert len(denied) == sum(e["event"] == "approval_requested" for e in entries) > 0

    # Run 2: approvals granted -> the reschedule is executed (simulated) and logged.
    approved_log = tmp_path / "approved.jsonl"
    agent.run_agent(top_n=3, use_llm=False, approver=lambda question: True,
                    audit=AuditLog(approved_log))
    entries = read_audit_log(approved_log)
    assert any(e["event"] == "approval_granted" for e in entries)
    assert any(e["event"] == "action_executed" and e["details"]["action"] == "reschedule_delivery"
               for e in entries)
