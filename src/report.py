"""Write the outputs of an agent run: prioritized_risks.csv and risk_report.md."""

import json
from datetime import datetime, timezone

from src import config, risk_engine

CSV_COLUMNS = [
    "rank", "po_id", "supplier_id", "material", "criticality_of_material", "order_value",
    "promised_delivery_date", "days_until_delivery", "is_overdue", "stock_coverage_days",
    "below_safety_stock", "supplier_on_time_rate", "supplier_late_rate",
    "observed_status", "rule_risk_score", "composite_risk_score", "risk_level",
    "priority_score", "risk_reasons",
]


def write_prioritized_csv(scored, path=None):
    path = path or config.OUTPUT_DIR / "prioritized_risks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    scored[CSV_COLUMNS].to_csv(path, index=False)
    return path


def _read_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summary_section(scored, results):
    levels = scored["risk_level"].value_counts()
    return [
        "## Summary",
        "",
        f"- Open purchase orders scored: **{len(scored)}**",
        f"- Risk levels: **{levels.get('High', 0)} High**, {levels.get('Medium', 0)} Medium, "
        f"{levels.get('Low', 0)} Low",
        f"- Orders already overdue: **{int(scored['is_overdue'].sum())}**",
        f"- Orders investigated by the agent: **{len(results)}**",
        "",
    ]


def _ranking_section(scored, top=10):
    lines = [
        f"## Top {top} orders by priority",
        "",
        "| Rank | PO | Material | Criticality | Value (EUR) | Supplier late | Rule score "
        "| Composite | Level | Priority |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for _, o in scored.head(top).iterrows():
        lines.append(
            f"| {o['rank']} | {o['po_id']} | {o['material']} | {o['criticality_of_material']} "
            f"| {o['order_value']:,.2f} | {o['supplier_late_rate']:.0%} | {o['rule_risk_score']} "
            f"| {o['composite_risk_score']} | {o['risk_level']} | {o['priority_score']} |"
        )
    return lines + [""]


def _describe_outcome(outcome):
    rule = f" Rule: {outcome['rule_fired']}."
    if outcome["status"] == "blocked":
        return "blocked by policy." + rule
    if outcome["status"] == "rejected":
        return f"approval denied ({outcome['approver']}) - not executed." + rule
    if outcome["status"] == "pending":
        return f"pending approval by the {outcome['approver']}." + rule
    approval = f"approved by the {outcome['approver']}" if outcome["approved"] else "automatic"
    if outcome["executed"]:
        return f"executed (simulated, {approval})." + rule
    return f"{approval} - {outcome['result']}" + rule


def _investigation_section(results):
    lines = ["## Investigated orders", ""]
    for result in results:
        if result["status"] == "error":
            lines += [f"### PO {result['po_id']} - error", "", f"{result['error']}", ""]
            continue
        facts, narrative = result["facts"], result["narrative"]
        po, supplier, stock, scores = (facts["purchase_order"], facts["supplier"],
                                       facts["stock"], facts["scores"])
        lines += [
            f"### #{scores['rank']} - PO {po['po_id']} ({po['material_description']})",
            "",
            f"- Supplier: {supplier['name']} - {supplier['on_time_rate']} on time over "
            f"{supplier['past_deliveries']} deliveries",
            f"- Value: {po['order_value_eur']:,.2f} EUR - criticality {po['criticality_of_material']}"
            f" - promised {po['promised_delivery_date']}",
            f"- Stock: {stock['current_stock']:.0f} (safety stock {stock['safety_stock']:.0f}, "
            f"coverage {stock['stock_coverage_days']} days)",
            f"- Scores: priority **{scores['priority_score']}**, composite "
            f"{scores['composite_risk_score']} ({scores['risk_level']}), supplier late rate "
            f"{scores['supplier_late_rate']:.0%} (statistic), observed: {scores['observed_status']}, "
            f"rule score {scores['rule_risk_score']}",
            "",
            "**Risk factors**",
            "",
        ]
        lines += [f"- {reason}" for reason in facts["risk_reasons"]]
        source = "LLM" if narrative["source"] == "llm" else "template"
        lines += [
            "",
            f"**Explanation** ({source}): {narrative['explanation']}",
            "",
            f"**Recommendation** ({source}): {narrative['recommendation']}",
            "",
            "**Actions**",
            "",
        ]
        lines += [f"- `{o['action']}`: {_describe_outcome(o)}" for o in result["outcomes"]]
        lines.append("")
    return lines


def _quality_section(summary):
    if summary is None:
        return []
    po = summary["purchase_orders"]
    history = summary["deliveries_history"]
    rejected = (po["rejected_unknown_supplier"] + po["rejected_promised_date_before_order_date"]
                + po["rejected_delivered_without_actual_date"])
    return [
        "## Data quality",
        "",
        f"- Purchase orders: {po['duplicate_rows_removed']} duplicates removed, {rejected} rejected "
        f"(unknown supplier, inconsistent dates, delivered without date), "
        f"{po['missing_unit_price_imputed'] + po['missing_quantity_imputed']} missing values imputed",
        f"- Delivery history: {history['duplicate_rows_removed']} duplicates removed, "
        f"{history['supplier_names_normalized']} supplier names normalized, "
        f"{history['missing_actual_date_rows_removed']} rows without delivery date removed",
        f"- Suppliers: {summary['suppliers']['duplicate_supplier_ids_removed']} duplicate removed",
        "",
    ]


def _method_section():
    p = config.RULE_POINTS
    return [
        "## How the scores are computed",
        "",
        f"- `rule_risk_score` (0-100): below safety stock {p['below_safety_stock']}, stock runs out "
        f"before delivery {p['stock_out_before_delivery']}, supplier on time < "
        f"{config.UNRELIABLE_SUPPLIER_ON_TIME_RATE:.0%} {p['unreliable_supplier']}.",
        "- `supplier_late_rate` (0-1): share of the supplier's deliveries before today that were late.",
        "- `observed_status`: \"Overdue by N day(s)\" or \"On track\" - a fact, not a prediction.",
        f"- `composite_risk_score`: {risk_engine.composite_formula()}",
        f"- `priority_score` = composite x ({config.PRIORITY_BASE} + "
        f"{config.PRIORITY_WEIGHT_CRITICALITY} x criticality + {config.PRIORITY_WEIGHT_VALUE} x value "
        f"+ {config.PRIORITY_WEIGHT_URGENCY} x urgency). A ranking score, not a probability.",
        "",
    ]


def write_risk_report(scored, results, run_info, path=None):
    """Write the Markdown report of one agent run."""
    path = path or config.OUTPUT_DIR / "risk_report.md"
    lines = [
        "# Procurement Risk Report",
        "",
        f"- Data as of: {config.AS_OF_DATE}",
        f"- Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC - run {run_info['run_id']}",
        f"- Explanations: {run_info['explanations']}",
        f"- Approval mode: {run_info['approval_mode']}",
        "- All SAP interactions are simulated. No email was sent and no SAP data was changed.",
        "",
    ]
    lines += _summary_section(scored, results)
    lines += _ranking_section(scored)
    lines += _investigation_section(results)
    lines += _quality_section(_read_json(config.PROCESSED_DIR / "data_quality_summary.json"))
    lines += _method_section()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
