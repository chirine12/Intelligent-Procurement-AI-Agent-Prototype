"""Central configuration for the Intelligent Procurement AI Agent prototype.

Every path, date, size and threshold lives here so that the behaviour of the
prototype can be explained (and changed) from a single file.
"""

from datetime import date
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- Reproducibility -------------------------------------------------------
SEED = 42

# "Today" for the whole prototype. It is fixed on purpose: features such as
# days_until_delivery and is_overdue depend on the current date, so a moving
# date would make the demo results change from one day to the next.
AS_OF_DATE = date(2026, 10, 12)

# --- Synthetic data volumes ------------------------------------------------
N_SUPPLIERS = 25
N_PURCHASE_ORDERS = 500
N_HISTORY_DELIVERIES = 1200

# Historical deliveries are ordered in [HISTORY_START, PO_WINDOW_START).
# Purchase orders are ordered in [PO_WINDOW_START, AS_OF_DATE).
HISTORY_START = date(2025, 1, 1)
PO_WINDOW_START = date(2026, 4, 15)

# --- Delay label -----------------------------------------------------------
# A delivery is labelled "late" when it arrives more than this many days
# after the promised delivery date. 0 means: any day after the promise.
LATE_TOLERANCE_DAYS = 0

# --- Data processing -------------------------------------------------------
# When safety stock is missing, assume this many days of consumption.
DEFAULT_SAFETY_STOCK_DAYS = 10

# --- Business rules --------------------------------------------------------
# Points added to rule_risk_score when a rule fires. They sum to 100.
# "Overdue" is not a rule here: it is a separate input of the composite score
# (see COMPOSITE_WEIGHTS), so it is never counted twice.
# The supplier rule is deliberately small: supplier reliability also enters
# the composite score through the supplier late rate.
RULE_POINTS = {
    "below_safety_stock": 40,
    "stock_out_before_delivery": 35,
    "unreliable_supplier": 25,
}
UNRELIABLE_SUPPLIER_ON_TIME_RATE = 0.70  # below this, the supplier rule fires

# --- Composite risk score (0-100) ------------------------------------------
# composite = 100 * (  w["supplier_late_rate"] * supplier late rate (0-1)
#                    + w["overdue"]            * overdue flag (0 or 1)
#                    + w["rules"]              * rule_risk_score / 100 )
# The supplier late rate is a plain statistic: the share of the supplier's past
# deliveries (before today) that arrived after the promised date.
# The weights sum to 1. Overdue has the largest weight: it is a fact about this
# order, while the late rate describes the supplier's past behaviour.
COMPOSITE_WEIGHTS = {"supplier_late_rate": 0.25, "overdue": 0.45, "rules": 0.30}

RISK_LEVEL_HIGH = 60    # composite score from which an order is "High" risk
RISK_LEVEL_MEDIUM = 35  # composite score from which an order is "Medium" risk

# --- Priority score (0-100) ------------------------------------------------
# priority = composite * (PRIORITY_BASE + sum of weight * factor), factors in [0, 1].
# The weights sum to 1, so priority can never exceed the composite score.
PRIORITY_BASE = 0.50
PRIORITY_WEIGHT_CRITICALITY = 0.25
PRIORITY_WEIGHT_VALUE = 0.15
PRIORITY_WEIGHT_URGENCY = 0.10

CRITICALITY_FACTOR = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
VALUE_CAP = 20000          # order value (EUR) at which the value factor reaches 1
URGENCY_HORIZON_DAYS = 30  # deliveries due later than this have urgency 0

# --- Mock SAP API ----------------------------------------------------------
MAX_ALTERNATIVE_SUPPLIERS = 3
RESCHEDULE_DAYS = 7  # a simulated reschedule proposes today + this many days
APPROVER_ROLE = "Procurement Manager"

# --- Agent guardrails ------------------------------------------------------
AUDIT_LOG_PATH = OUTPUT_DIR / "audit_log.jsonl"
DEFAULT_TOP_N = 3

# All guardrail rules in one place. agent.check_guardrail() applies them in
# this order: blocked list, sensitive actions (by value tier), automatic
# actions (with a value limit). An action that is in no list is blocked
# (default-deny).
GUARDRAIL_RULES = {
    "blocked_actions": ["delete_supplier", "modify_payment_terms"],
    "sensitive_actions": ["cancel_order", "replace_supplier", "reschedule_delivery"],
    "automatic_actions": ["monitor", "read_order", "send_supplier_reminder"],
    # Value tiers for sensitive actions (order value in EUR):
    #   value < 1,000            -> automatic
    #   1,000 <= value <= 10,000 -> Procurement Manager approval
    #   value > 10,000           -> Director approval
    "sensitive_automatic_below": 1000,
    "manager_approval_up_to": 10000,
    "manager_role": "Procurement Manager",
    "director_role": "Director",
    # Even an automatic action needs approval on a very large order
    # (monitor and read_order never change anything, so they are exempt).
    "automatic_value_limit": 15000,
    "automatic_limit_exempt": ["monitor", "read_order"],
}

# --- LLM (optional) --------------------------------------------------------
# The API key is read from the GROQ_API_KEY environment variable (or .env).
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"  # override with GROQ_MODEL
LLM_TIMEOUT_SECONDS = 20
