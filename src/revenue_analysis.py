"""
Phase 2 — Revenue & Business Performance Analysis
==================================================
Produces 4 publication-quality charts:
  1. Monthly GMV trend with Black Friday annotation
  2. Revenue by product category (top 15) with dual axis
  3. Payment method analysis (pie + bar combo)
  4. Seller concentration Lorenz curve with Gini coefficient

All charts use the project palette and are saved to /outputs at 150+ dpi.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import os
import sys
import io
import warnings

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── project palette ──────────────────────────────────────────────────────────
BLUE = "#378ADD"
TEAL = "#1D9E75"
RED = "#E24B4A"
AMBER = "#EF9F27"
DARK_BG = "#1a1a2e"
LIGHT_TEXT = "#e0e0e0"
GRID_COLOR = "#2a2a4a"

PALETTE = [BLUE, TEAL, RED, AMBER]

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_chart_style():
    """Configure matplotlib for dark-themed publication-quality charts."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": LIGHT_TEXT,
        "text.color": LIGHT_TEXT,
        "xtick.color": LIGHT_TEXT,
        "ytick.color": LIGHT_TEXT,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": DARK_BG,
    })


def load_master():
    """Load master dataframe with proper date parsing."""
    df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "master_df.csv"),
        parse_dates=[
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        low_memory=False,
    )
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Monthly GMV Trend
# ═══════════════════════════════════════════════════════════════════════════════

def chart_monthly_gmv(df):
    """
    Line chart of monthly GMV from Jan 2017 to Aug 2018.
    Annotates Black Friday spike (Nov 2017) and Jan 2018 growth inflection.
    Includes a 3-month rolling average overlay.
    """
    # Aggregate to order level first to avoid double-counting multi-item orders
    order_revenue = (
        df.drop_duplicates("order_id")[["order_id", "order_month", "payment_value"]]
    )

    monthly = (
        order_revenue
        .groupby("order_month")["payment_value"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["month", "gmv"]
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    # Filter to main analysis window
    mask = (monthly["month_dt"] >= "2017-01-01") & (monthly["month_dt"] <= "2018-08-31")
    monthly = monthly[mask].reset_index(drop=True)

    monthly["rolling_3m"] = monthly["gmv"].rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(monthly["month_dt"], monthly["gmv"], color=BLUE, linewidth=2.5,
            marker="o", markersize=6, label="Monthly GMV", zorder=3)
    ax.plot(monthly["month_dt"], monthly["rolling_3m"], color=AMBER, linewidth=2,
            linestyle="--", alpha=0.7, label="3-month rolling avg", zorder=2)

    ax.fill_between(monthly["month_dt"], monthly["gmv"], alpha=0.1, color=BLUE)

    # Annotate Black Friday (Nov 2017)
    bf_row = monthly[monthly["month_dt"].dt.strftime("%Y-%m") == "2017-11"]
    if not bf_row.empty:
        bf_x, bf_y = bf_row["month_dt"].values[0], bf_row["gmv"].values[0]
        ax.annotate(
            f"Black Friday\nR$ {bf_y/1e6:.2f}M",
            xy=(bf_x, bf_y),
            xytext=(bf_x - np.timedelta64(90, "D"), bf_y * 1.15),
            fontsize=10, fontweight="bold", color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
            ha="center",
        )

    # Annotate Jan 2018 inflection
    jan_row = monthly[monthly["month_dt"].dt.strftime("%Y-%m") == "2018-01"]
    if not jan_row.empty:
        j_x, j_y = jan_row["month_dt"].values[0], jan_row["gmv"].values[0]
        ax.annotate(
            f"Growth inflection\nR$ {j_y/1e6:.2f}M",
            xy=(j_x, j_y),
            xytext=(j_x + np.timedelta64(60, "D"), j_y * 1.20),
            fontsize=10, fontweight="bold", color=TEAL,
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
            ha="center",
        )

    ax.set_title("Olist GMV Trend — 2017 to 2018", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Gross Merchandise Value (R$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:.1f}M"))
    ax.legend(loc="upper left", framealpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart01_monthly_gmv.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Business insights
    total_2017 = monthly[monthly["month_dt"].dt.year == 2017]["gmv"].sum()
    q4_2017 = monthly[
        (monthly["month_dt"].dt.year == 2017) & (monthly["month_dt"].dt.month >= 10)
    ]["gmv"].sum()
    q4_pct = q4_2017 / total_2017 * 100

    first_gmv = monthly.iloc[0]["gmv"]
    last_gmv = monthly.iloc[-1]["gmv"]
    growth = (last_gmv / first_gmv - 1) * 100

    peak_idx = monthly["gmv"].idxmax()
    peak_month = monthly.loc[peak_idx, "month_dt"].strftime("%B %Y")
    peak_val = monthly.loc[peak_idx, "gmv"]

    print(f"\n  BUSINESS INSIGHT — GMV Trend:")
    print(f"    Overall growth: {growth:.0f}% from Jan 2017 to Aug 2018")
    print(f"    Peak month: {peak_month} at R$ {peak_val/1e6:.2f}M")
    print(f"    Q4 2017 contributed {q4_pct:.1f}% of full-year 2017 revenue")
    print(f"    Recommendation: Allocate additional logistics capacity for Q4 —")
    print(f"    {q4_pct:.0f}% of annual revenue in 3 months strains delivery SLAs")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Revenue by Product Category (Top 15)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_revenue_by_category(df):
    """
    Horizontal bar chart: top 15 categories by total revenue with order count overlay.
    Identifies the category with highest revenue-per-order as the non-obvious insight.
    """
    # Aggregate at order-item level for revenue, order level for count
    cat_revenue = (
        df.groupby("product_category_name_english")
        .agg(total_revenue=("price", "sum"), order_count=("order_id", "nunique"))
        .reset_index()
    )
    cat_revenue["revenue_per_order"] = cat_revenue["total_revenue"] / cat_revenue["order_count"]
    cat_revenue = cat_revenue.sort_values("total_revenue", ascending=False)

    top15 = cat_revenue.head(15).sort_values("total_revenue", ascending=True)

    fig, ax1 = plt.subplots(figsize=(14, 8))

    colors = [RED if i >= 12 else BLUE for i in range(len(top15))]

    bars = ax1.barh(top15["product_category_name_english"], top15["total_revenue"],
                    color=colors, alpha=0.85, zorder=3, height=0.6)

    ax1.set_xlabel("Total Revenue (R$)", color=BLUE)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:.1f}M"))

    ax2 = ax1.twiny()
    ax2.plot(top15["order_count"], top15["product_category_name_english"],
             color=AMBER, marker="D", markersize=7, linewidth=1.5, zorder=4)
    ax2.set_xlabel("Order Count", color=AMBER)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K"))

    ax1.set_title("Top 15 Product Categories by Revenue", fontsize=16, fontweight="bold", pad=30)
    ax1.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart02_revenue_by_category.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Non-obvious insight: highest revenue-per-order category
    # Filter to categories with meaningful volume (>50 orders)
    significant = cat_revenue[cat_revenue["order_count"] >= 50]
    top_rpo = significant.sort_values("revenue_per_order", ascending=False).iloc[0]
    top_rev = cat_revenue.iloc[0]

    ratio = top_rpo["revenue_per_order"] / top_rev["revenue_per_order"]

    print(f"\n  BUSINESS INSIGHT — Category Revenue:")
    print(f"    Top revenue category: {top_rev['product_category_name_english']}")
    print(f"      R$ {top_rev['total_revenue']/1e6:.2f}M from {top_rev['order_count']:,} orders")
    print(f"      Revenue per order: R$ {top_rev['revenue_per_order']:.2f}")
    print(f"    Highest revenue-per-order category: {top_rpo['product_category_name_english']}")
    print(f"      R$ {top_rpo['revenue_per_order']:.2f} per order ({ratio:.1f}x the top category)")
    print(f"    Recommendation: '{top_rpo['product_category_name_english']}' generates")
    print(f"    {ratio:.1f}x more revenue per order — a margin opportunity for targeted promotion")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Payment Method Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def chart_payment_analysis(df):
    """
    Dual visualization: pie chart by volume + bar chart by value.
    Analyzes whether high-value orders use different payment methods.
    """
    orders = df.drop_duplicates("order_id")[["order_id", "payment_type",
                                              "payment_installments", "payment_value"]]

    # Volume and value by payment type
    pay_summary = (
        orders.groupby("payment_type")
        .agg(count=("order_id", "count"), total_value=("payment_value", "sum"),
             avg_value=("payment_value", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    pay_summary["pct_volume"] = pay_summary["count"] / pay_summary["count"].sum() * 100
    pay_summary["pct_value"] = pay_summary["total_value"] / pay_summary["total_value"].sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Pie chart — by volume
    colors_pie = [BLUE, TEAL, AMBER, RED] + ["#666666"] * 10
    wedges, texts, autotexts = ax1.pie(
        pay_summary["pct_volume"], labels=pay_summary["payment_type"],
        autopct="%1.1f%%", colors=colors_pie[:len(pay_summary)],
        textprops={"color": LIGHT_TEXT, "fontsize": 11},
        startangle=90, pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax1.set_title("Payment Split by Volume", fontsize=14, fontweight="bold")

    # Bar chart — by value
    bars = ax2.bar(pay_summary["payment_type"], pay_summary["pct_value"],
                   color=colors_pie[:len(pay_summary)], alpha=0.85, zorder=3)
    ax2.set_title("Payment Split by Value (%)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("% of Total GMV")
    ax2.grid(axis="y", alpha=0.3)
    for bar, pct in zip(bars, pay_summary["pct_value"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{pct:.1f}%", ha="center", fontsize=11, fontweight="bold",
                 color=LIGHT_TEXT)

    fig.suptitle("Payment Method Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart03_payment_analysis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # High-value vs low-value payment method split
    median_val = orders["payment_value"].median()
    high_val = orders[orders["payment_value"] > orders["payment_value"].quantile(0.75)]
    low_val = orders[orders["payment_value"] <= orders["payment_value"].quantile(0.25)]

    high_cc = (high_val["payment_type"] == "credit_card").mean() * 100
    low_cc = (low_val["payment_type"] == "credit_card").mean() * 100
    high_boleto = (high_val["payment_type"] == "boleto").mean() * 100
    low_boleto = (low_val["payment_type"] == "boleto").mean() * 100

    # Installment analysis for credit card orders
    cc_orders = orders[orders["payment_type"] == "credit_card"]
    avg_installments = cc_orders["payment_installments"].mean()
    high_installment = cc_orders[cc_orders["payment_installments"] >= 6]
    high_inst_revenue = high_installment["payment_value"].sum()
    total_cc_revenue = cc_orders["payment_value"].sum()
    high_inst_pct = high_inst_revenue / total_cc_revenue * 100

    print(f"\n  BUSINESS INSIGHT — Payment Methods:")
    print(f"    Credit card: {pay_summary.iloc[0]['pct_volume']:.1f}% of orders, {pay_summary.iloc[0]['pct_value']:.1f}% of value")
    print(f"    High-value orders (top 25%): {high_cc:.1f}% credit card vs {high_boleto:.1f}% boleto")
    print(f"    Low-value orders (bottom 25%): {low_cc:.1f}% credit card vs {low_boleto:.1f}% boleto")
    print(f"    Avg credit card installments: {avg_installments:.1f}")
    print(f"    Orders with 6+ installments contribute {high_inst_pct:.1f}% of credit card revenue")
    print(f"    Recommendation: Installment plans drive high-value purchases —")
    print(f"    promoting 6+ installment options could increase AOV for mid-tier products")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Seller Concentration (Lorenz Curve)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_seller_lorenz(df):
    """
    Lorenz curve showing cumulative revenue vs cumulative seller share.
    Calculates Gini coefficient and identifies the 80/20 concentration point.
    """
    seller_rev = (
        df.groupby("seller_id")["price"]
        .sum()
        .sort_values()
        .reset_index()
    )
    seller_rev.columns = ["seller_id", "revenue"]

    total_rev = seller_rev["revenue"].sum()
    n = len(seller_rev)

    seller_rev["cum_rev"] = seller_rev["revenue"].cumsum() / total_rev
    seller_rev["cum_sellers"] = np.arange(1, n + 1) / n

    # Gini coefficient: area between Lorenz curve and equality line
    gini = 1 - 2 * np.trapezoid(seller_rev["cum_rev"], seller_rev["cum_sellers"])

    # Find what % of sellers generate 80% of revenue
    pct_80 = seller_rev[seller_rev["cum_rev"] >= 0.20].iloc[0]["cum_sellers"]
    top_seller_pct = (1 - pct_80) * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot([0, 1], [0, 1], color=LIGHT_TEXT, linestyle="--", alpha=0.5,
            label="Perfect equality", linewidth=1)
    ax.plot(seller_rev["cum_sellers"], seller_rev["cum_rev"],
            color=BLUE, linewidth=2.5, label="Lorenz curve", zorder=3)
    ax.fill_between(seller_rev["cum_sellers"], seller_rev["cum_rev"],
                    seller_rev["cum_sellers"], alpha=0.15, color=RED)

    # Annotate Gini
    ax.text(0.15, 0.75, f"Gini = {gini:.3f}", fontsize=18, fontweight="bold",
            color=RED, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=RED, alpha=0.8))

    # Annotate 80/20 point
    ax.axhline(y=0.80, color=AMBER, linestyle=":", alpha=0.6)
    ax.axvline(x=pct_80, color=AMBER, linestyle=":", alpha=0.6)
    ax.plot(pct_80, 0.80, "o", color=AMBER, markersize=10, zorder=4)
    ax.annotate(
        f"Top {top_seller_pct:.0f}% of sellers\ngenerate 80% of revenue",
        xy=(pct_80, 0.80),
        xytext=(0.35, 0.55),
        fontsize=11, fontweight="bold", color=AMBER,
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
        transform=ax.transAxes,
    )

    ax.set_title("Seller Revenue Concentration (Lorenz Curve)", fontsize=16,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Cumulative Share of Sellers (sorted by revenue)")
    ax.set_ylabel("Cumulative Share of Revenue")
    ax.legend(loc="upper left", framealpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart04_seller_lorenz.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Top seller stats
    top10_rev = seller_rev.tail(int(n * 0.10))["revenue"].sum()
    top10_pct = top10_rev / total_rev * 100

    print(f"\n  BUSINESS INSIGHT — Seller Concentration:")
    print(f"    Total active sellers: {n:,}")
    print(f"    Gini coefficient: {gini:.3f} (high concentration)")
    print(f"    Top {top_seller_pct:.0f}% of sellers generate 80% of revenue")
    print(f"    Top 10% of sellers generate {top10_pct:.1f}% of revenue")
    print(f"    Recommendation: High seller concentration (Gini {gini:.2f}) creates")
    print(f"    platform risk — losing a few top sellers could slash revenue.")
    print(f"    Invest in seller development programs for the mid-tier to diversify")
    print(f"    the revenue base and reduce dependency on the top {top_seller_pct:.0f}%.")


# ═══════════════════════════════════════════════════════════════════════════════
# REVENUE SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_revenue_summary(df):
    """Print a comprehensive revenue summary report."""

    orders = df.drop_duplicates("order_id")
    delivered = orders[orders["order_status"] == "delivered"]

    total_gmv = orders["payment_value"].sum()
    total_orders = len(orders)
    avg_order_value = orders["payment_value"].mean()
    median_order_value = orders["payment_value"].median()
    unique_customers = df["customer_unique_id"].nunique()
    unique_sellers = df["seller_id"].nunique()

    print("\n" + "=" * 70)
    print("  REVENUE SUMMARY REPORT")
    print("=" * 70)
    print(f"    Total GMV:            R$ {total_gmv/1e6:.2f}M")
    print(f"    Total Orders:         {total_orders:,}")
    print(f"    Avg Order Value:      R$ {avg_order_value:.2f}")
    print(f"    Median Order Value:   R$ {median_order_value:.2f}")
    print(f"    Unique Customers:     {unique_customers:,}")
    print(f"    Unique Sellers:       {unique_sellers:,}")
    print(f"    Avg Items per Order:  {len(df)/total_orders:.2f}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  PHASE 2 — REVENUE & BUSINESS PERFORMANCE ANALYSIS")
    print("█" * 70)

    set_chart_style()
    df = load_master()
    print(f"\n  Master dataframe loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    chart_monthly_gmv(df)
    chart_revenue_by_category(df)
    chart_payment_analysis(df)
    chart_seller_lorenz(df)
    print_revenue_summary(df)

    print("\n" + "=" * 70)
    print("✓ PHASE 2 COMPLETE — All 4 revenue charts saved to /outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
