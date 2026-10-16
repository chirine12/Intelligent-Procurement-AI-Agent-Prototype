"""Mock SAP API: the tools the agent uses to read data and simulate actions.

Read tools (no side effect):
    get_purchase_order, get_supplier_performance, get_stock_level,
    find_alternative_suppliers
Action tools (simulated only - nothing is sent, nothing is modified):
    send_supplier_reminder, create_approval_request, reschedule_delivery

Every function returns a dictionary:
    {"success": True, "data": {...}}  or  {"success": False, "error": "..."}

In a real landscape these would be calls to SAP OData services. Here they read
the cleaned CSV files in data/processed/.
"""

from functools import lru_cache

import pandas as pd

from src import config
from src.data_processing import build_closed_deliveries, supplier_stats_before

DATE_COLUMNS = ["order_date", "promised_delivery_date", "actual_delivery_date"]


@lru_cache(maxsize=1)
def _data():
    """Load the cleaned datasets once and keep them in memory."""
    directory = config.PROCESSED_DIR
    clean = {
        "suppliers": pd.read_csv(directory / "suppliers.csv"),
        "stock_levels": pd.read_csv(directory / "stock_levels.csv"),
        "purchase_orders": pd.read_csv(directory / "purchase_orders.csv", dtype={"po_id": str},
                                       parse_dates=DATE_COLUMNS),
        "deliveries_history": pd.read_csv(directory / "deliveries_history.csv",
                                          parse_dates=DATE_COLUMNS),
    }
    clean["closed_deliveries"] = build_closed_deliveries(clean)
    return clean


def reload_data():
    """Forget the cached datasets (used after the data has been regenerated)."""
    _data.cache_clear()


def _ok(data):
    return {"success": True, "data": data}


def _error(message):
    return {"success": False, "error": message}


def _find_po(po_id):
    """Return the purchase order row, or None when it does not exist."""
    po = _data()["purchase_orders"]
    match = po[po["po_id"] == str(po_id)]
    return None if match.empty else match.iloc[0]


def _find_open_po(po_id):
    """Return (row, None) for an open order, or (None, error response)."""
    order = _find_po(po_id)
    if order is None:
        return None, _error(f"Purchase order {po_id} not found")
    if order["status"] != "Open":
        return None, _error(f"Purchase order {po_id} is {order['status']}, not Open")
    return order, None


# --- Read tools ------------------------------------------------------------

def get_purchase_order(po_id):
    """Header and item data of one purchase order."""
    order = _find_po(po_id)
    if order is None:
        return _error(f"Purchase order {po_id} not found")

    as_of = pd.Timestamp(config.AS_OF_DATE)
    is_open = order["status"] == "Open"
    days_until_delivery = int((order["promised_delivery_date"] - as_of).days) if is_open else None
    actual = order["actual_delivery_date"]
    return _ok({
        "po_id": order["po_id"],
        "supplier_id": order["supplier_id"],
        "material": order["material"],
        "quantity": float(order["quantity"]),
        "unit_price": float(order["unit_price"]),
        "order_value": round(float(order["quantity"] * order["unit_price"]), 2),
        "order_date": str(order["order_date"].date()),
        "promised_delivery_date": str(order["promised_delivery_date"].date()),
        "actual_delivery_date": None if pd.isna(actual) else str(actual.date()),
        "status": order["status"],
        "plant": order["plant"],
        "criticality_of_material": order["criticality_of_material"],
        "days_until_delivery": days_until_delivery,
        "is_overdue": bool(is_open and days_until_delivery < 0),
    })


def get_supplier_performance(supplier_id):
    """Master data and delivery reliability of one supplier, as of today."""
    suppliers = _data()["suppliers"]
    match = suppliers[suppliers["supplier_id"] == supplier_id]
    if match.empty:
        return _error(f"Supplier {supplier_id} not found")
    supplier = match.iloc[0]

    # Same calculation as the ML features, so the agent sees the same numbers.
    stats = supplier_stats_before(
        _data()["closed_deliveries"], supplier_id, pd.Timestamp(config.AS_OF_DATE)
    )
    on_time_rate = stats["supplier_on_time_rate"]
    return _ok({
        "supplier_id": supplier_id,
        "name": supplier["name"],
        "country": supplier["country"],
        "category": supplier["category"],
        "payment_terms": supplier["payment_terms"],
        "past_deliveries": int(stats["supplier_past_deliveries"]),
        "on_time_rate": None if on_time_rate is None else float(on_time_rate),
        "avg_delay_days": None if on_time_rate is None else float(stats["supplier_avg_delay"]),
        "is_reliable": bool(
            on_time_rate is not None and on_time_rate >= config.UNRELIABLE_SUPPLIER_ON_TIME_RATE
        ),
    })


def get_stock_level(material):
    """Current stock situation of one material."""
    stock = _data()["stock_levels"]
    match = stock[stock["material"] == material]
    if match.empty:
        return _error(f"Material {material} not found")
    row = match.iloc[0]
    return _ok({
        "material": material,
        "material_description": row["material_description"],
        "current_stock": float(row["current_stock"]),
        "safety_stock": float(row["safety_stock"]),
        "daily_consumption": float(row["daily_consumption"]),
        "stock_coverage_days": round(float(row["current_stock"] / row["daily_consumption"]), 1),
        "below_safety_stock": bool(row["current_stock"] < row["safety_stock"]),
    })


def find_alternative_suppliers(material, exclude_supplier_id=None):
    """Other suppliers that have already delivered this material, best first."""
    if get_stock_level(material)["success"] is False:
        return _error(f"Material {material} not found")

    closed = _data()["closed_deliveries"]
    supplier_ids = sorted(closed.loc[closed["material"] == material, "supplier_id"].unique())
    candidates = []
    for supplier_id in supplier_ids:
        if supplier_id == exclude_supplier_id:
            continue
        performance = get_supplier_performance(supplier_id)["data"]
        candidates.append({key: performance[key] for key in [
            "supplier_id", "name", "country", "past_deliveries", "on_time_rate",
            "avg_delay_days", "is_reliable",
        ]})

    candidates.sort(key=lambda c: (-c["on_time_rate"], c["supplier_id"]))
    return _ok({
        "material": material,
        "excluded_supplier_id": exclude_supplier_id,
        "alternatives": candidates[:config.MAX_ALTERNATIVE_SUPPLIERS],
    })


# --- Action tools (simulated) ----------------------------------------------

def send_supplier_reminder(po_id, message=None):
    """Simulate sending a delivery reminder to the supplier. No email is sent."""
    order, error = _find_open_po(po_id)
    if error:
        return error
    supplier = get_supplier_performance(order["supplier_id"])["data"]
    if message is None:
        message = (
            f"Reminder: purchase order {order['po_id']} ({order['material']}) was promised "
            f"for {order['promised_delivery_date'].date()}. Please confirm the delivery date."
        )
    return _ok({
        "action": "send_supplier_reminder",
        "simulated": True,
        "reminder_id": f"REM-{order['po_id']}",
        "po_id": order["po_id"],
        "supplier_id": order["supplier_id"],
        "recipient": supplier["name"],
        "message": message,
    })


def create_approval_request(po_id, action):
    """Simulate creating an approval work item for a sensitive action."""
    order, error = _find_open_po(po_id)
    if error:
        return error
    return _ok({
        "action": "create_approval_request",
        "simulated": True,
        "approval_id": f"APR-{order['po_id']}-{action}",
        "po_id": order["po_id"],
        "requested_action": action,
        "approver_role": config.APPROVER_ROLE,
        "status": "PENDING",
    })


def reschedule_delivery(po_id):
    """Simulate moving the promised delivery date. The data is not modified."""
    order, error = _find_open_po(po_id)
    if error:
        return error
    new_date = config.AS_OF_DATE + pd.Timedelta(days=config.RESCHEDULE_DAYS)
    return _ok({
        "action": "reschedule_delivery",
        "simulated": True,
        "po_id": order["po_id"],
        "old_promised_delivery_date": str(order["promised_delivery_date"].date()),
        "new_promised_delivery_date": str(new_date),
    })
