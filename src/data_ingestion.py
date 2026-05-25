"""
Phase 1 — Data Ingestion, Validation & Master DataFrame
========================================================
Loads all 9 Olist CSV files, validates schema and referential integrity,
merges into a single master dataframe with engineered time/delivery columns,
and saves to /outputs/master_df.csv.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import io

warnings.filterwarnings("ignore")

# Fix Windows console encoding for special characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD ALL 9 CSV FILES
# ═══════════════════════════════════════════════════════════════════════════════

def load_datasets():
    """Load all 9 Olist CSV files with correct dtypes and date parsing."""

    date_cols_orders = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    orders = pd.read_csv(
        os.path.join(DATA_DIR, "olist_orders_dataset.csv"),
        parse_dates=date_cols_orders,
    )

    order_items = pd.read_csv(
        os.path.join(DATA_DIR, "olist_order_items_dataset.csv"),
        parse_dates=["shipping_limit_date"],
        dtype={"price": float, "freight_value": float},
    )

    payments = pd.read_csv(
        os.path.join(DATA_DIR, "olist_order_payments_dataset.csv"),
        dtype={"payment_installments": int, "payment_value": float},
    )

    customers = pd.read_csv(
        os.path.join(DATA_DIR, "olist_customers_dataset.csv"),
        dtype={"customer_zip_code_prefix": str},
    )

    sellers = pd.read_csv(
        os.path.join(DATA_DIR, "olist_sellers_dataset.csv"),
        dtype={"seller_zip_code_prefix": str},
    )

    products = pd.read_csv(
        os.path.join(DATA_DIR, "olist_products_dataset.csv"),
        dtype={
            "product_name_lenght": "Int64",
            "product_description_lenght": "Int64",
            "product_photos_qty": "Int64",
            "product_weight_g": "Int64",
            "product_length_cm": "Int64",
            "product_height_cm": "Int64",
            "product_width_cm": "Int64",
        },
    )

    reviews = pd.read_csv(
        os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv"),
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )

    geolocation = pd.read_csv(
        os.path.join(DATA_DIR, "olist_geolocation_dataset.csv"),
        dtype={"geolocation_zip_code_prefix": str},
    )

    translations = pd.read_csv(
        os.path.join(DATA_DIR, "product_category_name_translation.csv"),
        encoding="utf-8",
    )

    datasets = {
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "customers": customers,
        "sellers": sellers,
        "products": products,
        "reviews": reviews,
        "geolocation": geolocation,
        "translations": translations,
    }

    print("=" * 70)
    print("STEP 1 — DATASETS LOADED")
    print("=" * 70)
    for name, df in datasets.items():
        print(f"\n  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        print(f"    Columns: {list(df.columns)}")

    return datasets


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — VALIDATE EACH TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def validate_tables(datasets):
    """Print null counts, duplicate counts, and dtype issues for every table."""

    print("\n" + "=" * 70)
    print("STEP 2 — TABLE VALIDATION REPORT")
    print("=" * 70)

    for name, df in datasets.items():
        print(f"\n{'─' * 50}")
        print(f"  TABLE: {name} ({df.shape[0]:,} rows × {df.shape[1]} cols)")
        print(f"{'─' * 50}")

        null_counts = df.isnull().sum()
        nulls_with_values = null_counts[null_counts > 0]
        if len(nulls_with_values) > 0:
            print("  Nulls:")
            for col, cnt in nulls_with_values.items():
                pct = cnt / len(df) * 100
                print(f"    {col}: {cnt:,} ({pct:.1f}%)")
        else:
            print("  Nulls: None")

        dup_count = df.duplicated().sum()
        print(f"  Exact duplicate rows: {dup_count:,}")

        for col in df.columns:
            if df[col].dtype == object:
                numeric_check = pd.to_numeric(df[col], errors="coerce")
                non_null_original = df[col].notna().sum()
                non_null_numeric = numeric_check.notna().sum()
                if non_null_original > 0 and non_null_numeric / non_null_original > 0.9:
                    failed = non_null_original - non_null_numeric
                    if failed > 0:
                        print(f"  ⚠ Column '{col}' looks numeric but has {failed} non-numeric values")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FIX KNOWN ISSUES
# ═══════════════════════════════════════════════════════════════════════════════

def fix_known_issues(datasets):
    """Merge translations into products, fix dtypes, handle encoding."""

    products = datasets["products"]
    translations = datasets["translations"]

    products = products.merge(translations, on="product_category_name", how="left")

    untranslated = products["product_category_name_english"].isnull().sum()
    if untranslated > 0:
        products["product_category_name_english"] = products[
            "product_category_name_english"
        ].fillna(products["product_category_name"])
        print(f"\n  Filled {untranslated} untranslated category names with Portuguese originals")

    datasets["products"] = products

    reviews = datasets["reviews"]
    reviews["review_comment_title"] = reviews["review_comment_title"].fillna("")
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("")
    datasets["reviews"] = reviews

    print("\n" + "=" * 70)
    print("STEP 3 — KNOWN ISSUES FIXED")
    print("=" * 70)
    print("  ✓ Product category translations merged into products table")
    print("  ✓ Review comment nulls filled with empty strings")
    print("  ✓ All date columns parsed as datetime")
    print(f"  ✓ Products table now has {products.shape[1]} columns (added product_category_name_english)")

    return datasets


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — REFERENTIAL INTEGRITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_referential_integrity(datasets):
    """Verify foreign key relationships across all joined tables."""

    orders = datasets["orders"]
    order_items = datasets["order_items"]
    customers = datasets["customers"]
    sellers = datasets["sellers"]
    payments = datasets["payments"]
    reviews = datasets["reviews"]

    print("\n" + "=" * 70)
    print("STEP 4 — REFERENTIAL INTEGRITY REPORT")
    print("=" * 70)

    orphans_total = 0

    # order_items.order_id → orders.order_id
    items_orders = set(order_items["order_id"]) - set(orders["order_id"])
    print(f"\n  order_items → orders:  {len(items_orders)} orphaned order_ids")
    orphans_total += len(items_orders)

    # orders.customer_id → customers.customer_id
    orders_customers = set(orders["customer_id"]) - set(customers["customer_id"])
    print(f"  orders → customers:   {len(orders_customers)} orphaned customer_ids")
    orphans_total += len(orders_customers)

    # order_items.seller_id → sellers.seller_id
    items_sellers = set(order_items["seller_id"]) - set(sellers["seller_id"])
    print(f"  order_items → sellers: {len(items_sellers)} orphaned seller_ids")
    orphans_total += len(items_sellers)

    # order_items.product_id → products.product_id
    items_products = set(order_items["product_id"]) - set(datasets["products"]["product_id"])
    print(f"  order_items → products: {len(items_products)} orphaned product_ids")
    orphans_total += len(items_products)

    # payments.order_id → orders.order_id
    pay_orders = set(payments["order_id"]) - set(orders["order_id"])
    print(f"  payments → orders:    {len(pay_orders)} orphaned order_ids")
    orphans_total += len(pay_orders)

    # reviews.order_id → orders.order_id
    rev_orders = set(reviews["order_id"]) - set(orders["order_id"])
    print(f"  reviews → orders:     {len(rev_orders)} orphaned order_ids")
    orphans_total += len(rev_orders)

    print(f"\n  TOTAL: {orphans_total} orphaned records found across all tables")
    if orphans_total == 0:
        print("  ✓ All foreign key relationships are intact")

    return orphans_total


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD MASTER DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def build_master_df(datasets):
    """
    Merge all tables into a single master dataframe.

    Join logic (documented):
      1. orders LEFT JOIN customers        ON customer_id
      2. result LEFT JOIN order_items       ON order_id
      3. result LEFT JOIN products          ON product_id
      4. result LEFT JOIN sellers           ON seller_id
      5. result LEFT JOIN payments (agg)    ON order_id  (aggregated to order level)
      6. result LEFT JOIN reviews (dedup)   ON order_id  (one review per order)

    LEFT joins preserve all orders even if items/payments/reviews are missing.
    Payments are aggregated per order to avoid row explosion from multiple installments.
    Reviews are deduplicated keeping the most recent per order.
    """

    orders = datasets["orders"]
    order_items = datasets["order_items"]
    customers = datasets["customers"]
    sellers = datasets["sellers"]
    products = datasets["products"]
    payments = datasets["payments"]
    reviews = datasets["reviews"]

    # Aggregate payments to order level to avoid row explosion
    payments_agg = (
        payments.groupby("order_id")
        .agg(
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "max"),
            payment_value=("payment_value", "sum"),
            payment_methods_count=("payment_sequential", "nunique"),
        )
        .reset_index()
    )

    # Deduplicate reviews: keep latest review per order
    reviews_dedup = (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates(subset="order_id", keep="last")
        [["order_id", "review_score", "review_comment_title", "review_comment_message",
          "review_creation_date", "review_answer_timestamp"]]
    )

    # Sequential merges
    print("\n" + "=" * 70)
    print("STEP 5 — BUILDING MASTER DATAFRAME")
    print("=" * 70)

    master = orders.merge(customers, on="customer_id", how="left")
    print(f"  1. orders + customers     → {master.shape[0]:,} rows")

    master = master.merge(order_items, on="order_id", how="left")
    print(f"  2. + order_items          → {master.shape[0]:,} rows")

    master = master.merge(
        products[["product_id", "product_category_name_english",
                  "product_weight_g", "product_length_cm",
                  "product_height_cm", "product_width_cm",
                  "product_photos_qty"]],
        on="product_id",
        how="left",
    )
    print(f"  3. + products             → {master.shape[0]:,} rows")

    master = master.merge(
        sellers, on="seller_id", how="left", suffixes=("", "_seller")
    )
    print(f"  4. + sellers              → {master.shape[0]:,} rows")

    master = master.merge(payments_agg, on="order_id", how="left")
    print(f"  5. + payments (agg)       → {master.shape[0]:,} rows")

    master = master.merge(reviews_dedup, on="order_id", how="left")
    print(f"  6. + reviews (dedup)      → {master.shape[0]:,} rows")

    return master


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — ENGINEER COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_columns(master):
    """
    Add delivery timing, delay flags, and calendar columns.
    These are the foundation for every downstream analysis.
    """

    print("\n" + "=" * 70)
    print("STEP 6 — FEATURE ENGINEERING")
    print("=" * 70)

    # Delivery timing columns
    master["delivery_days"] = (
        master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    master["estimated_days"] = (
        master["order_estimated_delivery_date"] - master["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    master["delay_days"] = master["delivery_days"] - master["estimated_days"]

    master["is_late"] = (master["delay_days"] > 0).astype("Int64")

    # Calendar columns
    master["order_month"] = master["order_purchase_timestamp"].dt.to_period("M")
    master["order_quarter"] = master["order_purchase_timestamp"].dt.to_period("Q")
    master["order_year"] = master["order_purchase_timestamp"].dt.year
    master["order_dayofweek"] = master["order_purchase_timestamp"].dt.dayofweek

    # Revenue per item
    master["revenue_per_item"] = np.where(
        master["order_item_id"] > 0,
        master["payment_value"] / master["order_item_id"],
        master["payment_value"],
    )

    engineered_cols = [
        "delivery_days", "estimated_days", "delay_days", "is_late",
        "order_month", "order_quarter", "order_year", "order_dayofweek",
        "revenue_per_item",
    ]

    print("  New columns added:")
    for col in engineered_cols:
        non_null = master[col].notna().sum()
        print(f"    {col}: {non_null:,} non-null values ({non_null/len(master)*100:.1f}%)")

    # Summary of delivery timing
    delivered = master[master["delivery_days"].notna()]
    print(f"\n  Delivery stats (n={len(delivered):,} delivered orders):")
    print(f"    Mean delivery: {delivered['delivery_days'].mean():.1f} days")
    print(f"    Median delivery: {delivered['delivery_days'].median():.1f} days")
    print(f"    Late delivery rate: {delivered['is_late'].mean()*100:.1f}%")
    print(f"    Mean delay (when late): {delivered.loc[delivered['is_late']==1, 'delay_days'].mean():.1f} days")

    return master


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE & FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_and_report(master):
    """Save master dataframe and print final validation report."""

    output_path = os.path.join(OUTPUT_DIR, "master_df.csv")

    # Convert Period columns to strings for CSV compatibility
    master_save = master.copy()
    master_save["order_month"] = master_save["order_month"].astype(str)
    master_save["order_quarter"] = master_save["order_quarter"].astype(str)

    master_save.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("STEP 7 — MASTER DATAFRAME SAVED")
    print("=" * 70)
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Shape: {master.shape[0]:,} rows × {master.shape[1]} columns")
    print(f"\n  Column list ({master.shape[1]} total):")
    for i, col in enumerate(master.columns, 1):
        dtype = master[col].dtype
        print(f"    {i:2d}. {col} ({dtype})")

    # Order status distribution
    print(f"\n  Order status distribution:")
    status_counts = master.drop_duplicates("order_id")["order_status"].value_counts()
    for status, count in status_counts.items():
        print(f"    {status}: {count:,}")

    # Date range
    min_date = master["order_purchase_timestamp"].min()
    max_date = master["order_purchase_timestamp"].max()
    print(f"\n  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    # Revenue summary
    total_revenue = master.drop_duplicates("order_id")["payment_value"].sum()
    print(f"  Total GMV: R$ {total_revenue:,.2f}")

    print("\n" + "=" * 70)
    print("✓ PHASE 1 COMPLETE — Master dataframe ready for analysis")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the full Phase 1 pipeline end-to-end."""

    print("\n" + "█" * 70)
    print("  OLIST BUSINESS ANALYTICS — PHASE 1: DATA INGESTION & VALIDATION")
    print("█" * 70)

    datasets = load_datasets()
    validate_tables(datasets)
    datasets = fix_known_issues(datasets)
    check_referential_integrity(datasets)
    master = build_master_df(datasets)
    master = engineer_columns(master)
    save_and_report(master)

    return master


if __name__ == "__main__":
    master = main()
