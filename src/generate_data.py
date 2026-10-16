"""Generate synthetic SAP-like procurement data (raw layer).

Files written to data/raw/:
    suppliers.csv            - supplier master data (inspired by SAP LFA1)
    purchase_orders.csv      - purchase orders (inspired by SAP EKKO/EKPO)
    deliveries_history.csv   - older, closed deliveries per supplier
    stock_levels.csv         - stock, safety stock and daily consumption
    injected_anomalies.json  - exact list of the data-quality issues injected

Run with:  python -m src.generate_data
"""

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from src import config

# --- Reference data --------------------------------------------------------

# 25 suppliers: 5 per category, so every material has alternative suppliers.
SUPPLIERS = [
    # (name, country, category)
    ("Rheinstahl GmbH", "DE", "Raw Materials"),
    ("Nordic Alloys AB", "SE", "Raw Materials"),
    ("Iberia Metals SA", "ES", "Raw Materials"),
    ("Anatolia Steel AS", "TR", "Raw Materials"),
    ("Silesia Aluminium Sp", "PL", "Raw Materials"),
    ("Shenzhen Microtech Ltd", "CN", "Electronics"),
    ("Taipei Circuits Co", "TW", "Electronics"),
    ("Bavaria Sensorik GmbH", "DE", "Electronics"),
    ("Penang Semicon Sdn", "MY", "Electronics"),
    ("Lyon Electronics Ltd", "FR", "Electronics"),
    ("Danube Packaging Kft", "HU", "Packaging"),
    ("Flanders Carton NV", "BE", "Packaging"),
    ("Lombardia Imballaggi Srl", "IT", "Packaging"),
    ("Baltic Pack OU", "EE", "Packaging"),
    ("Tunis Packaging Ltd", "TN", "Packaging"),
    ("Schwaben Praezision GmbH", "DE", "Mechanical Parts"),
    ("Bohemia Bearings sro", "CZ", "Mechanical Parts"),
    ("Pune Forgings Pvt Ltd", "IN", "Mechanical Parts"),
    ("Torino Meccanica SpA", "IT", "Mechanical Parts"),
    ("Midlands Fasteners Ltd", "GB", "Mechanical Parts"),
    ("Rotterdam Chemie BV", "NL", "Chemicals"),
    ("Basel Coatings AG", "CH", "Chemicals"),
    ("Gujarat Polymers Ltd", "IN", "Chemicals"),
    ("Texas Lubricants Inc", "US", "Chemicals"),
    ("Catalonia Resins SL", "ES", "Chemicals"),
]

# 20 materials: 4 per category.
MATERIALS = [
    # (material, description, category, base_price, typical_qty, criticality)
    ("MAT-1001", "Steel Sheet 2mm", "Raw Materials", 42.0, 400, "High"),
    ("MAT-1002", "Aluminium Profile", "Raw Materials", 18.5, 600, "Medium"),
    ("MAT-1003", "Copper Wire Coil", "Raw Materials", 95.0, 150, "High"),
    ("MAT-1004", "Stainless Rod 10mm", "Raw Materials", 27.0, 300, "Low"),
    ("MAT-2001", "Control Board PCB", "Electronics", 135.0, 120, "High"),
    ("MAT-2002", "Proximity Sensor", "Electronics", 48.0, 250, "High"),
    ("MAT-2003", "Power Connector", "Electronics", 3.2, 2000, "Medium"),
    ("MAT-2004", "LED Indicator", "Electronics", 0.9, 5000, "Low"),
    ("MAT-3001", "Cardboard Box L", "Packaging", 1.4, 3000, "Low"),
    ("MAT-3002", "Protective Foam", "Packaging", 2.1, 1500, "Low"),
    ("MAT-3003", "Wooden Pallet", "Packaging", 11.0, 400, "Medium"),
    ("MAT-3004", "Stretch Film Roll", "Packaging", 6.5, 500, "Low"),
    ("MAT-4001", "Ball Bearing 6204", "Mechanical Parts", 7.8, 1200, "High"),
    ("MAT-4002", "Gear Shaft", "Mechanical Parts", 64.0, 180, "High"),
    ("MAT-4003", "Hex Bolt M8", "Mechanical Parts", 0.25, 10000, "Low"),
    ("MAT-4004", "Hydraulic Seal", "Mechanical Parts", 4.6, 800, "Medium"),
    ("MAT-5001", "Industrial Adhesive", "Chemicals", 23.0, 200, "Medium"),
    ("MAT-5002", "Powder Coating", "Chemicals", 15.5, 350, "Medium"),
    ("MAT-5003", "Machine Lubricant", "Chemicals", 9.8, 400, "High"),
    ("MAT-5004", "Cleaning Solvent", "Chemicals", 5.4, 300, "Low"),
]

PLANTS = ["PL01", "PL02", "PL03"]
PAYMENT_TERMS = ["NET30", "NET45", "NET60", "NET90"]

# Number of rows affected by each injected data-quality issue.
ANOMALY_COUNTS = {
    "po_duplicate_rows": 8,
    "po_missing_unit_price": 10,
    "po_missing_quantity": 5,
    "po_missing_criticality": 6,
    "po_promised_before_order": 5,
    "po_delivered_without_actual_date": 4,
    "po_unknown_supplier": 3,
    "supplier_duplicate_rows": 1,
    "supplier_missing_payment_terms": 2,
    "history_duplicate_rows": 10,
    "history_missing_actual_date": 8,
    "history_supplier_name_variants": 60,
    "stock_missing_safety_stock": 1,
}


# --- Clean data generation -------------------------------------------------

def build_suppliers(rng):
    """Return the supplier master and each supplier's hidden late probability.

    The hidden probability drives the simulation but is never written to a
    CSV: the risk engine has to estimate reliability from delivery history.
    """
    suppliers = pd.DataFrame(SUPPLIERS, columns=["name", "country", "category"])
    suppliers.insert(0, "supplier_id", [f"S{i:03d}" for i in range(1, len(suppliers) + 1)])
    suppliers["payment_terms"] = rng.choice(PAYMENT_TERMS, size=len(suppliers))

    # Most suppliers are reasonably reliable; 4 are clearly unreliable.
    late_prob = rng.uniform(0.08, 0.30, size=len(suppliers))
    unreliable = rng.choice(len(suppliers), size=4, replace=False)
    late_prob[unreliable] = rng.uniform(0.50, 0.70, size=4)
    hidden_late_prob = dict(zip(suppliers["supplier_id"], late_prob))
    return suppliers, hidden_late_prob


def simulate_orders(rng, n, start, end, suppliers, hidden_late_prob):
    """Simulate n orders placed between start and end (inclusive).

    Every order gets a "true" actual delivery date. The caller decides whether
    that date is already known (delivered) or still in the future (open).
    """
    materials = pd.DataFrame(
        MATERIALS,
        columns=["material", "description", "category", "base_price", "typical_qty", "criticality"],
    )
    span_days = (end - start).days
    rows = []
    for _ in range(n):
        supplier = suppliers.iloc[rng.integers(len(suppliers))]
        candidates = materials[materials["category"] == supplier["category"]]
        material = candidates.iloc[rng.integers(len(candidates))]

        order_date = start + timedelta(days=int(rng.integers(0, span_days + 1)))
        lead_time = int(rng.integers(7, 61))  # promised lead time in days
        quantity = max(1, int(material["typical_qty"] * rng.lognormal(0, 0.4)))
        unit_price = round(material["base_price"] * rng.uniform(0.9, 1.1), 2)

        # Late probability = supplier reliability, adjusted by order context.
        # Work on the logit scale so the result always stays between 0 and 1.
        base = hidden_late_prob[supplier["supplier_id"]]
        logit = np.log(base / (1 - base))
        if lead_time <= 14:  # tight promises are harder to keep
            logit += 0.9
        if quantity > 1.5 * material["typical_qty"]:  # unusually large orders
            logit += 0.5
        if supplier["category"] == "Electronics":  # more volatile supply chain
            logit += 0.4
        late_probability = 1 / (1 + np.exp(-logit))

        if rng.random() < late_probability:
            delay_days = min(45, 1 + int(rng.gamma(1.5, 4.0)))
        else:
            delay_days = -int(rng.integers(0, 4))  # on time or up to 3 days early

        promised = order_date + timedelta(days=lead_time)
        rows.append({
            "supplier_id": supplier["supplier_id"],
            "supplier_name": supplier["name"],
            "material": material["material"],
            "quantity": quantity,
            "unit_price": unit_price,
            "order_date": pd.Timestamp(order_date),
            "promised_delivery_date": pd.Timestamp(promised),
            "actual_delivery_date": pd.Timestamp(promised + timedelta(days=delay_days)),
            "criticality_of_material": material["criticality"],
        })
    return pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)


def build_purchase_orders(rng, suppliers, hidden_late_prob):
    """Purchase orders of the current window, with a status as of AS_OF_DATE."""
    po = simulate_orders(
        rng, config.N_PURCHASE_ORDERS, config.PO_WINDOW_START,
        config.AS_OF_DATE - timedelta(days=2), suppliers, hidden_late_prob,
    )
    po.insert(0, "po_id", [str(4500000001 + i) for i in range(len(po))])
    po["plant"] = rng.choice(PLANTS, size=len(po))

    # Status follows from the fixed "today": a delivery in the future is not
    # known yet, so the order is still open and its actual date is blank.
    as_of = pd.Timestamp(config.AS_OF_DATE)
    delivered = po["actual_delivery_date"] <= as_of
    po["status"] = np.where(delivered, "Delivered", "Open")
    po.loc[~delivered, "actual_delivery_date"] = pd.NaT

    # A few open orders were cancelled.
    open_idx = po.index[po["status"] == "Open"]
    cancelled = rng.choice(open_idx, size=6, replace=False)
    po.loc[cancelled, "status"] = "Cancelled"

    return po[[
        "po_id", "supplier_id", "material", "quantity", "unit_price", "order_date",
        "promised_delivery_date", "actual_delivery_date", "status", "plant",
        "criticality_of_material",
    ]]


def build_deliveries_history(rng, suppliers, hidden_late_prob):
    """Older, fully delivered orders used to measure supplier reliability."""
    history = simulate_orders(
        rng, config.N_HISTORY_DELIVERIES, config.HISTORY_START,
        config.PO_WINDOW_START - timedelta(days=1), suppliers, hidden_late_prob,
    )
    history.insert(0, "delivery_id", [f"D{i:06d}" for i in range(1, len(history) + 1)])
    history["delay_days"] = (
        history["actual_delivery_date"] - history["promised_delivery_date"]
    ).dt.days
    return history[[
        "delivery_id", "supplier_id", "supplier_name", "material", "quantity",
        "order_date", "promised_delivery_date", "actual_delivery_date", "delay_days",
    ]]


def build_stock_levels(rng):
    """One row per material: stock, safety stock and daily consumption."""
    rows = []
    for material, description, _, _, typical_qty, _ in MATERIALS:
        daily_consumption = max(1, round(typical_qty / 15))
        rows.append({
            "material": material,
            "material_description": description,
            # Between 3 and 35 days of coverage, so some materials are short.
            "current_stock": int(daily_consumption * rng.uniform(3, 35)),
            "safety_stock": int(daily_consumption * rng.uniform(7, 14)),
            "daily_consumption": daily_consumption,
        })
    return pd.DataFrame(rows)


# --- Data-quality issues ---------------------------------------------------

def spelling_variant(name, rng):
    """Return the same supplier name written slightly differently."""
    variants = [name.upper(), name.lower(), name + " ", "  " + name, name.replace(" ", "  ", 1)]
    return variants[rng.integers(len(variants))]


def inject_anomalies(rng, suppliers, po, history, stock):
    """Inject a small, documented set of realistic data-quality issues.

    Each issue hits different rows (taken from one shuffled list of indices),
    so the counts in ANOMALY_COUNTS are exact and can be checked in tests.
    """
    counts = ANOMALY_COUNTS
    log = {}

    # --- Purchase orders ---
    free_rows = list(rng.permutation(po.index))

    def take(n, mask=None):
        """Pick n unused PO rows, optionally restricted to rows where mask is True."""
        picked = [i for i in free_rows if mask is None or mask[i]][:n]
        for i in picked:
            free_rows.remove(i)
        return picked

    idx = take(counts["po_missing_unit_price"])
    po.loc[idx, "unit_price"] = np.nan
    log["po_missing_unit_price"] = sorted(po.loc[idx, "po_id"])

    idx = take(counts["po_missing_quantity"])
    po["quantity"] = po["quantity"].astype("float")
    po.loc[idx, "quantity"] = np.nan
    log["po_missing_quantity"] = sorted(po.loc[idx, "po_id"])

    idx = take(counts["po_missing_criticality"])
    po.loc[idx, "criticality_of_material"] = np.nan
    log["po_missing_criticality"] = sorted(po.loc[idx, "po_id"])

    idx = take(counts["po_promised_before_order"])
    po.loc[idx, "promised_delivery_date"] = po.loc[idx, "order_date"] - pd.Timedelta(days=5)
    log["po_promised_before_order"] = sorted(po.loc[idx, "po_id"])

    idx = take(counts["po_delivered_without_actual_date"], mask=po["status"] == "Delivered")
    po.loc[idx, "actual_delivery_date"] = pd.NaT
    log["po_delivered_without_actual_date"] = sorted(po.loc[idx, "po_id"])

    idx = take(counts["po_unknown_supplier"])
    po.loc[idx, "supplier_id"] = "S999"
    log["po_unknown_supplier"] = sorted(po.loc[idx, "po_id"])

    idx = take(counts["po_duplicate_rows"])
    log["po_duplicate_rows"] = sorted(po.loc[idx, "po_id"])
    po = pd.concat([po, po.loc[idx]]).sort_values("po_id", kind="stable").reset_index(drop=True)

    # --- Suppliers ---
    idx = rng.choice(suppliers.index, size=counts["supplier_missing_payment_terms"], replace=False)
    suppliers.loc[idx, "payment_terms"] = np.nan
    log["supplier_missing_payment_terms"] = sorted(suppliers.loc[idx, "supplier_id"])

    duplicate = suppliers.loc[[rng.integers(len(suppliers))]].copy()
    duplicate["name"] = duplicate["name"].str.upper() + " "
    log["supplier_duplicate_rows"] = sorted(duplicate["supplier_id"])
    suppliers = pd.concat([suppliers, duplicate]).reset_index(drop=True)

    # --- Deliveries history ---
    history_rows = rng.permutation(history.index)
    n_missing = counts["history_missing_actual_date"]
    n_variants = counts["history_supplier_name_variants"]
    n_duplicates = counts["history_duplicate_rows"]

    idx = history_rows[:n_missing]
    history["delay_days"] = history["delay_days"].astype("float")
    history.loc[idx, ["actual_delivery_date", "delay_days"]] = [pd.NaT, np.nan]
    log["history_missing_actual_date"] = sorted(history.loc[idx, "delivery_id"])

    idx = history_rows[n_missing:n_missing + n_variants]
    history.loc[idx, "supplier_name"] = [
        spelling_variant(name, rng) for name in history.loc[idx, "supplier_name"]
    ]
    log["history_supplier_name_variants"] = sorted(history.loc[idx, "delivery_id"])

    idx = history_rows[n_missing + n_variants:n_missing + n_variants + n_duplicates]
    log["history_duplicate_rows"] = sorted(history.loc[idx, "delivery_id"])
    history = (
        pd.concat([history, history.loc[idx]])
        .sort_values("delivery_id", kind="stable")
        .reset_index(drop=True)
    )

    # --- Stock levels ---
    idx = rng.choice(stock.index, size=counts["stock_missing_safety_stock"], replace=False)
    stock["safety_stock"] = stock["safety_stock"].astype("float")
    stock.loc[idx, "safety_stock"] = np.nan
    log["stock_missing_safety_stock"] = sorted(stock.loc[idx, "material"])

    return suppliers, po, history, stock, log


# --- Entry point -----------------------------------------------------------

def generate_all(raw_dir=config.RAW_DIR, seed=config.SEED):
    """Generate every raw dataset and write it to raw_dir. Returns the frames."""
    rng = np.random.default_rng(seed)

    suppliers, hidden_late_prob = build_suppliers(rng)
    po = build_purchase_orders(rng, suppliers, hidden_late_prob)
    history = build_deliveries_history(rng, suppliers, hidden_late_prob)
    stock = build_stock_levels(rng)
    suppliers, po, history, stock, anomaly_log = inject_anomalies(rng, suppliers, po, history, stock)

    raw_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "suppliers": suppliers,
        "purchase_orders": po,
        "deliveries_history": history,
        "stock_levels": stock,
    }
    for name, frame in datasets.items():
        frame.to_csv(raw_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    with open(raw_dir / "injected_anomalies.json", "w", encoding="utf-8") as f:
        json.dump(anomaly_log, f, indent=2)
    return datasets


def main():
    datasets = generate_all()
    print(f"Raw data written to {config.RAW_DIR} (seed={config.SEED}, as of {config.AS_OF_DATE})")
    for name, frame in datasets.items():
        print(f"  {name:<20} {len(frame):>5} rows")
    print("  purchase order status:", datasets["purchase_orders"]["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
