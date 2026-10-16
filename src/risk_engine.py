"""Risk engine: business rules + a supplier statistic + transparent scores.

For every open purchase order it computes:
    supplier_late_rate    - share of the supplier's past deliveries that were late (0-1)
    observed_status       - what is already known today ("Overdue by N day(s)" / "On track")
    rule_risk_score       - points from the business rules (0-100)
    composite_risk_score  - weighted mix of supplier statistic, overdue flag and rules (0-100)
    risk_level            - High / Medium / Low, from the composite score
    priority_score        - composite risk weighted by criticality, value and urgency
                            (0-100). It is a ranking score, not a probability.
    risk_reasons          - the reasons, in plain words

No model is trained: every number can be recomputed by hand from the data.

Run with:  python -m src.risk_engine
"""

import pandas as pd

from src import config


# --- Business rules --------------------------------------------------------

def apply_rules(order):
    """Evaluate the business rules for one open order.

    Returns (rule_risk_score, list of human-readable reasons).
    """
    points, reasons = 0, []

    # Overdue is listed as a reason, but scored separately (COMPOSITE_WEIGHTS).
    if order["is_overdue"]:
        reasons.append(f"Overdue by {-order['days_until_delivery']} day(s)")

    if order["below_safety_stock"]:
        points += config.RULE_POINTS["below_safety_stock"]
        reasons.append(
            f"Stock below safety stock ({order['current_stock']:.0f} < {order['safety_stock']:.0f})"
        )

    # Not yet overdue, but the stock will run out before the promised date.
    if not order["is_overdue"] and order["stock_coverage_days"] < order["days_until_delivery"]:
        points += config.RULE_POINTS["stock_out_before_delivery"]
        reasons.append(
            f"Stock covers {order['stock_coverage_days']:.0f} day(s), "
            f"delivery promised in {order['days_until_delivery']} day(s)"
        )

    if order["supplier_on_time_rate"] < config.UNRELIABLE_SUPPLIER_ON_TIME_RATE:
        points += config.RULE_POINTS["unreliable_supplier"]
        reasons.append(
            f"Unreliable supplier ({order['supplier_on_time_rate']:.0%} on time "
            f"over {order['supplier_past_deliveries']} deliveries)"
        )

    return min(points, 100), reasons


# --- Scoring ---------------------------------------------------------------

def composite_formula():
    """The composite formula with the weights from config, for captions and reports."""
    w = config.COMPOSITE_WEIGHTS
    return (f"Composite risk = 100 x ({w['supplier_late_rate']} x supplier late rate + "
            f"{w['overdue']} x overdue (0/1) + {w['rules']} x rule score / 100). "
            f"High >= {config.RISK_LEVEL_HIGH}, Medium >= {config.RISK_LEVEL_MEDIUM}.")


def risk_level(composite_score):
    if composite_score >= config.RISK_LEVEL_HIGH:
        return "High"
    if composite_score >= config.RISK_LEVEL_MEDIUM:
        return "Medium"
    return "Low"


def observed_status(days_until_delivery):
    """What is already known today."""
    if days_until_delivery < 0:
        return f"Overdue by {-days_until_delivery} day(s)"
    return "On track"


def score_open_orders(open_orders):
    """Score and rank the open purchase orders. Highest priority first."""
    scored = open_orders.copy()
    scored["observed_status"] = scored["days_until_delivery"].map(observed_status)

    rule_results = [apply_rules(order) for _, order in scored.iterrows()]
    scored["rule_risk_score"] = [score for score, _ in rule_results]
    scored["risk_reasons"] = ["; ".join(reasons) if reasons else "No risk factor identified"
                              for _, reasons in rule_results]

    weights = config.COMPOSITE_WEIGHTS
    scored["composite_risk_score"] = (100 * (
        weights["supplier_late_rate"] * scored["supplier_late_rate"]
        + weights["overdue"] * scored["is_overdue"].astype(int)
        + weights["rules"] * scored["rule_risk_score"] / 100
    )).round(1)
    scored["risk_level"] = scored["composite_risk_score"].map(risk_level)

    # Priority factors, each between 0 and 1.
    criticality = scored["criticality_of_material"].map(config.CRITICALITY_FACTOR)
    value = (scored["order_value"] / config.VALUE_CAP).clip(upper=1)
    urgency = (1 - scored["days_until_delivery"] / config.URGENCY_HORIZON_DAYS).clip(0, 1)
    scored["priority_score"] = (scored["composite_risk_score"] * (
        config.PRIORITY_BASE
        + config.PRIORITY_WEIGHT_CRITICALITY * criticality
        + config.PRIORITY_WEIGHT_VALUE * value
        + config.PRIORITY_WEIGHT_URGENCY * urgency
    )).round(1)

    scored = scored.sort_values(["priority_score", "po_id"], ascending=[False, True])
    scored.insert(0, "rank", range(1, len(scored) + 1))
    return scored.reset_index(drop=True)


# --- Entry point -----------------------------------------------------------

def main():
    open_orders = pd.read_csv(config.PROCESSED_DIR / "open_orders.csv", dtype={"po_id": str})
    scored = score_open_orders(open_orders)
    columns = ["rank", "po_id", "supplier_late_rate", "observed_status", "rule_risk_score",
               "composite_risk_score", "risk_level", "priority_score", "risk_reasons"]
    print(scored[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
