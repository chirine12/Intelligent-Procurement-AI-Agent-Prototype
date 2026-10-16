"""Load, validate and clean the raw data, then build the tables used for scoring.

Files written to data/processed/:
    suppliers.csv, purchase_orders.csv, deliveries_history.csv, stock_levels.csv
                                  - cleaned versions of the raw files
    rejected_purchase_orders.csv  - orders that could not be repaired, with a reason
    data_quality_summary.json     - what was found and how it was handled
    closed_orders.csv             - deliveries with a known outcome and the late flag
    open_orders.csv               - open orders with today's situation, for the risk assessment

Run with:  python -m src.data_processing
"""

import json

import pandas as pd

from src import config

# Expected columns per raw file (schema validation).
REQUIRED_COLUMNS = {
    "suppliers": ["supplier_id", "name", "country", "category", "payment_terms"],
    "purchase_orders": [
        "po_id", "supplier_id", "material", "quantity", "unit_price", "order_date",
        "promised_delivery_date", "actual_delivery_date", "status", "plant",
        "criticality_of_material",
    ],
    "deliveries_history": [
        "delivery_id", "supplier_id", "supplier_name", "material", "quantity",
        "order_date", "promised_delivery_date", "actual_delivery_date", "delay_days",
    ],
    "stock_levels": [
        "material", "material_description", "current_stock", "safety_stock",
        "daily_consumption",
    ],
}
DATE_COLUMNS = ["order_date", "promised_delivery_date", "actual_delivery_date"]
NUMERIC_COLUMNS = ["quantity", "unit_price", "delay_days", "current_stock", "safety_stock",
                   "daily_consumption"]


# --- Loading and schema validation -----------------------------------------

def load_raw(raw_dir=config.RAW_DIR):
    """Read the four raw CSV files as text; types are set in validate_schema."""
    return {name: pd.read_csv(raw_dir / f"{name}.csv", dtype=str) for name in REQUIRED_COLUMNS}


def validate_schema(frames):
    """Check required columns and convert dates and numbers to proper types.

    Values that cannot be converted become missing (NaT / NaN) and are then
    handled like any other missing value by the cleaning functions.
    """
    for name, columns in REQUIRED_COLUMNS.items():
        missing = [c for c in columns if c not in frames[name].columns]
        if missing:
            raise ValueError(f"{name}.csv is missing required columns: {missing}")

    for frame in frames.values():
        for column in frame.columns:
            if column in DATE_COLUMNS:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
            elif column in NUMERIC_COLUMNS:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frames


def normalize_name(name):
    """Remove leading/trailing spaces and collapse repeated spaces."""
    return " ".join(str(name).split())


# --- Cleaning --------------------------------------------------------------

def clean_suppliers(suppliers, summary):
    suppliers = suppliers.copy()
    suppliers["name"] = suppliers["name"].map(normalize_name)

    # supplier_id is the key: a second row with the same id is a duplicate,
    # even when the name is spelled differently. The first row is kept.
    duplicates = suppliers.duplicated(subset="supplier_id", keep="first")
    suppliers = suppliers[~duplicates].reset_index(drop=True)

    missing_terms = suppliers["payment_terms"].isna()
    suppliers.loc[missing_terms, "payment_terms"] = "UNKNOWN"

    summary["suppliers"] = {
        "duplicate_supplier_ids_removed": int(duplicates.sum()),
        "missing_payment_terms_set_to_unknown": int(missing_terms.sum()),
    }
    return suppliers


def clean_stock(stock, summary):
    stock = stock.copy()
    missing = stock["safety_stock"].isna()
    stock.loc[missing, "safety_stock"] = (
        stock.loc[missing, "daily_consumption"] * config.DEFAULT_SAFETY_STOCK_DAYS
    )
    stock["safety_stock_imputed"] = missing
    summary["stock_levels"] = {"missing_safety_stock_imputed": int(missing.sum())}
    return stock


def clean_history(history, suppliers, summary):
    history = history.copy()

    duplicates = history.duplicated(keep="first")
    history = history[~duplicates]

    unknown_supplier = ~history["supplier_id"].isin(suppliers["supplier_id"])
    history = history[~unknown_supplier]

    # Supplier names: the master data is the reference. A name is replaced
    # only when it is the same name apart from case and spacing.
    master_name = history["supplier_id"].map(suppliers.set_index("supplier_id")["name"])
    spelled_differently = history["supplier_name"] != master_name
    same_supplier = history["supplier_name"].map(normalize_name).str.lower() == master_name.str.lower()
    names_fixed = spelled_differently & same_supplier
    history.loc[names_fixed, "supplier_name"] = master_name[names_fixed]

    # Without an actual delivery date the delay cannot be measured.
    missing_actual = history["actual_delivery_date"].isna()
    history = history[~missing_actual].copy()

    # Recompute the delay from the dates instead of trusting the stored value.
    history["delay_days"] = (
        history["actual_delivery_date"] - history["promised_delivery_date"]
    ).dt.days

    summary["deliveries_history"] = {
        "duplicate_rows_removed": int(duplicates.sum()),
        "unknown_supplier_rows_removed": int(unknown_supplier.sum()),
        "supplier_names_normalized": int(names_fixed.sum()),
        "supplier_names_not_matching_master": int((spelled_differently & ~same_supplier).sum()),
        "missing_actual_date_rows_removed": int(missing_actual.sum()),
    }
    return history.reset_index(drop=True)


def clean_purchase_orders(po, suppliers, summary):
    """Return (clean orders, rejected orders with a reason)."""
    po = po.copy()

    duplicates = po.duplicated(keep="first")
    po = po[~duplicates].copy()

    # Orders that cannot be repaired are rejected, not silently dropped.
    po["reject_reason"] = None
    unknown_supplier = ~po["supplier_id"].isin(suppliers["supplier_id"])
    promised_before_order = po["promised_delivery_date"] < po["order_date"]
    delivered_without_date = (po["status"] == "Delivered") & po["actual_delivery_date"].isna()
    po.loc[unknown_supplier, "reject_reason"] = "unknown_supplier"
    po.loc[promised_before_order, "reject_reason"] = "promised_date_before_order_date"
    po.loc[delivered_without_date, "reject_reason"] = "delivered_without_actual_date"

    rejected = po[po["reject_reason"].notna()].reset_index(drop=True)
    po = po[po["reject_reason"].isna()].drop(columns="reject_reason").copy()

    # Missing price / quantity: use the median of the same material and keep
    # a flag, so that an imputed order value can be recognised later.
    for column in ["unit_price", "quantity"]:
        missing = po[column].isna()
        po[f"{column}_imputed"] = missing
        material_median = po.groupby("material")[column].transform("median")
        po.loc[missing, column] = material_median[missing]

    # Criticality is a property of the material, so it is copied from the
    # other orders of the same material.
    missing_criticality = po["criticality_of_material"].isna()
    known = po.dropna(subset=["criticality_of_material"]).groupby("material")
    criticality_by_material = known["criticality_of_material"].agg(lambda s: s.mode().iloc[0])
    po.loc[missing_criticality, "criticality_of_material"] = (
        po.loc[missing_criticality, "material"].map(criticality_by_material)
    )

    summary["purchase_orders"] = {
        "duplicate_rows_removed": int(duplicates.sum()),
        "rejected_unknown_supplier": int(unknown_supplier.sum()),
        "rejected_promised_date_before_order_date": int(promised_before_order.sum()),
        "rejected_delivered_without_actual_date": int(delivered_without_date.sum()),
        "missing_unit_price_imputed": int(po["unit_price_imputed"].sum()),
        "missing_quantity_imputed": int(po["quantity_imputed"].sum()),
        "missing_criticality_filled": int(missing_criticality.sum()),
        "remaining_missing_values": int(
            po.drop(columns="actual_delivery_date").isna().sum().sum()
        ),
    }
    return po.reset_index(drop=True), rejected


def clean_all(frames):
    """Clean every dataset. Returns (clean frames, rejected orders, summary)."""
    summary = {}
    suppliers = clean_suppliers(frames["suppliers"], summary)
    stock = clean_stock(frames["stock_levels"], summary)
    history = clean_history(frames["deliveries_history"], suppliers, summary)
    po, rejected = clean_purchase_orders(frames["purchase_orders"], suppliers, summary)

    clean = {
        "suppliers": suppliers,
        "purchase_orders": po,
        "deliveries_history": history,
        "stock_levels": stock,
    }
    summary["row_counts"] = {
        name: {"raw": len(frames[name]), "clean": len(clean[name])} for name in clean
    }
    return clean, rejected, summary


# --- Feature engineering ---------------------------------------------------

def build_closed_deliveries(clean):
    """All deliveries whose outcome is known: history + delivered purchase orders."""
    po = clean["purchase_orders"]
    delivered = po[po["status"] == "Delivered"].copy()
    delivered["delay_days"] = (
        delivered["actual_delivery_date"] - delivered["promised_delivery_date"]
    ).dt.days
    delivered = delivered.rename(columns={"po_id": "order_ref"}).assign(source="purchase_order")
    history = clean["deliveries_history"].rename(columns={"delivery_id": "order_ref"})
    history = history.assign(source="history")

    columns = ["order_ref", "source", "supplier_id", "material", "quantity", "order_date",
               "promised_delivery_date", "actual_delivery_date", "delay_days"]
    closed = pd.concat([history[columns], delivered[columns]], ignore_index=True)
    closed["delivery_delay_label"] = (closed["delay_days"] > config.LATE_TOLERANCE_DAYS).astype(int)
    return closed


def supplier_stats_before(closed, supplier_id, cutoff_date):
    """Supplier reliability using only deliveries completed before cutoff_date.

    This is the leakage guard: a delivery that had not arrived yet on the
    cutoff date cannot influence the statistics.
    """
    past = closed[
        (closed["supplier_id"] == supplier_id) & (closed["actual_delivery_date"] < cutoff_date)
    ]
    if past.empty:
        return {"supplier_past_deliveries": 0, "supplier_avg_delay": None,
                "supplier_on_time_rate": None}
    return {
        "supplier_past_deliveries": len(past),
        # Average days late; early deliveries count as 0, not as negative delay.
        "supplier_avg_delay": round(past["delay_days"].clip(lower=0).mean(), 2),
        "supplier_on_time_rate": round(1 - past["delivery_delay_label"].mean(), 3),
    }


def add_supplier_features(orders, closed, clean, cutoff):
    """Add the supplier category and its delivery statistics as of the cutoff date."""
    orders = orders.copy()
    suppliers = clean["suppliers"].set_index("supplier_id")
    orders["category"] = orders["supplier_id"].map(suppliers["category"])
    stats = [supplier_stats_before(closed, supplier_id, cutoff) for supplier_id in orders["supplier_id"]]
    return pd.concat([orders, pd.DataFrame(stats, index=orders.index)], axis=1)


def build_closed_orders(clean, closed):
    """Closed orders with their supplier category, sorted by order date."""
    suppliers = clean["suppliers"].set_index("supplier_id")
    closed = closed.assign(category=closed["supplier_id"].map(suppliers["category"]))
    return closed.sort_values("order_date").reset_index(drop=True)


def build_open_orders(clean, closed):
    """Open orders: supplier statistics as of today + today's operational situation."""
    as_of = pd.Timestamp(config.AS_OF_DATE)
    po = clean["purchase_orders"]
    open_po = po[po["status"] == "Open"].drop(columns="actual_delivery_date")

    open_po = add_supplier_features(open_po, closed, clean, cutoff=as_of)
    # The statistic used by the risk score: how often this supplier was late before today.
    open_po["supplier_late_rate"] = (1 - open_po["supplier_on_time_rate"]).round(3)

    open_po["order_value"] = (open_po["quantity"] * open_po["unit_price"]).round(2)
    open_po["days_until_delivery"] = (open_po["promised_delivery_date"] - as_of).dt.days
    open_po["is_overdue"] = open_po["days_until_delivery"] < 0

    stock_columns = ["material", "current_stock", "safety_stock", "daily_consumption"]
    open_po = open_po.merge(clean["stock_levels"][stock_columns], on="material", how="left")
    open_po["stock_coverage_days"] = (open_po["current_stock"] / open_po["daily_consumption"]).round(1)
    open_po["below_safety_stock"] = open_po["current_stock"] < open_po["safety_stock"]
    return open_po.sort_values("po_id").reset_index(drop=True)


# --- Entry point -----------------------------------------------------------

def run(raw_dir=config.RAW_DIR, processed_dir=config.PROCESSED_DIR):
    """Full pipeline: load, validate, clean, build features, write files."""
    frames = validate_schema(load_raw(raw_dir))
    clean, rejected, summary = clean_all(frames)

    closed = build_closed_deliveries(clean)
    closed_orders = build_closed_orders(clean, closed)
    open_orders = build_open_orders(clean, closed)

    status_counts = clean["purchase_orders"]["status"].value_counts().to_dict()
    summary["feature_tables"] = {
        "closed_orders_rows": len(closed_orders),
        "closed_orders_late_rate": round(float(closed_orders["delivery_delay_label"].mean()), 3),
        "open_orders_rows": len(open_orders),
        "open_orders_overdue": int(open_orders["is_overdue"].sum()),
        "cancelled_orders_excluded": int(status_counts.get("Cancelled", 0)),
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = dict(clean, rejected_purchase_orders=rejected, closed_orders=closed_orders,
                   open_orders=open_orders)
    for name, frame in outputs.items():
        frame.to_csv(processed_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    with open(processed_dir / "data_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return outputs, summary


def main():
    _, summary = run()
    print(f"Processed data written to {config.PROCESSED_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
