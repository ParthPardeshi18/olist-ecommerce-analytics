"""
Phase 3 — Delivery & Operations Analysis
=========================================
Produces 4 publication-quality charts:
  5. Delivery performance by state (all 27 states)
  6. Delivery delay distribution (histogram + KDE)
  7. Review score vs delivery delay (scatter + regression)
  8. On-time delivery rate over time (monthly trend)

All charts use the project dark palette and save at 150+ dpi.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


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
# CHART 5 — Delivery Performance by State
# ═══════════════════════════════════════════════════════════════════════════════

def chart_delivery_by_state(df):
    """
    Horizontal bar chart of average delivery days per Brazilian state (all 27).
    Color-coded: green (<=10), amber (10-20), red (>20).
    Overlays late delivery rate as dots on secondary axis.
    """
    delivered = df[df["delivery_days"].notna()].copy()

    state_perf = (
        delivered.groupby("customer_state")
        .agg(
            avg_delivery=("delivery_days", "mean"),
            late_rate=("is_late", "mean"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values("avg_delivery", ascending=True)
    )

    def bar_color(days):
        """Green for fast, amber for moderate, red for slow delivery."""
        if days <= 10:
            return TEAL
        elif days <= 20:
            return AMBER
        return RED

    colors = [bar_color(d) for d in state_perf["avg_delivery"]]

    fig, ax1 = plt.subplots(figsize=(14, 10))

    bars = ax1.barh(state_perf["customer_state"], state_perf["avg_delivery"],
                    color=colors, alpha=0.85, zorder=3, height=0.65)
    ax1.set_xlabel("Average Delivery Days", color=BLUE)
    ax1.set_title("Delivery Performance by State", fontsize=16, fontweight="bold", pad=15)

    # Annotate best and worst
    best = state_perf.iloc[0]
    worst = state_perf.iloc[-1]
    ax1.annotate(
        f"Best: {best['customer_state']} ({best['avg_delivery']:.1f}d)",
        xy=(best["avg_delivery"], 0),
        xytext=(best["avg_delivery"] + 5, 2),
        fontsize=10, fontweight="bold", color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
    )
    ax1.annotate(
        f"Worst: {worst['customer_state']} ({worst['avg_delivery']:.1f}d)",
        xy=(worst["avg_delivery"], len(state_perf) - 1),
        xytext=(worst["avg_delivery"] - 8, len(state_perf) - 4),
        fontsize=10, fontweight="bold", color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )

    # Secondary axis: late delivery rate
    ax2 = ax1.twiny()
    ax2.scatter(state_perf["late_rate"] * 100, state_perf["customer_state"],
                color=AMBER, s=60, zorder=4, marker="D", edgecolors="white", linewidths=0.5)
    ax2.set_xlabel("Late Delivery Rate (%)", color=AMBER)

    ax1.grid(axis="x", alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TEAL, markersize=10, label="<= 10 days"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=AMBER, markersize=10, label="10-20 days"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, markersize=10, label="> 20 days"),
        Line2D([0], [0], marker="D", color=AMBER, markersize=8, linestyle="None", label="Late rate (%)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart05_delivery_by_state.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Business insight
    ratio = worst["avg_delivery"] / best["avg_delivery"]
    high_late = state_perf[state_perf["late_rate"] > 0.10]

    print(f"\n  BUSINESS INSIGHT — Delivery by State:")
    print(f"    Fastest: {best['customer_state']} at {best['avg_delivery']:.1f} days ({best['late_rate']*100:.1f}% late)")
    print(f"    Slowest: {worst['customer_state']} at {worst['avg_delivery']:.1f} days ({worst['late_rate']*100:.1f}% late)")
    print(f"    Gap: {ratio:.1f}x difference between fastest and slowest state")
    print(f"    States with >10% late rate: {', '.join(high_late['customer_state'].tolist())}")
    print(f"    Recommendation: {worst['customer_state']} and remote Northern states need")
    print(f"    dedicated logistics partnerships or regional fulfillment centers")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Delivery Delay Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def chart_delay_distribution(df):
    """
    Histogram + KDE of delay_days (negative = early, positive = late).
    Marks mean, median, 90th percentile. Separate distributions for on-time vs late.
    """
    delivered = df[df["delay_days"].notna()].copy()
    # Clip extreme outliers for visualization
    delays = delivered["delay_days"].clip(-30, 60)

    on_time = delays[delivered["is_late"] == 0]
    late = delays[delivered["is_late"] == 1]

    mean_delay = delivered["delay_days"].mean()
    median_delay = delivered["delay_days"].median()
    p90 = delivered["delay_days"].quantile(0.90)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.hist(on_time, bins=80, alpha=0.6, color=TEAL, label="On-time / Early",
            density=True, zorder=2)
    ax.hist(late, bins=40, alpha=0.6, color=RED, label="Late",
            density=True, zorder=2)

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde_all = gaussian_kde(delays.dropna())
    x_range = np.linspace(-30, 60, 300)
    ax.plot(x_range, kde_all(x_range), color=BLUE, linewidth=2, label="Overall KDE", zorder=3)

    # Vertical reference lines
    ax.axvline(mean_delay, color=AMBER, linewidth=2, linestyle="--",
               label=f"Mean: {mean_delay:.1f} days", zorder=4)
    ax.axvline(median_delay, color=BLUE, linewidth=2, linestyle="-.",
               label=f"Median: {median_delay:.1f} days", zorder=4)
    ax.axvline(p90, color=RED, linewidth=2, linestyle=":",
               label=f"90th pctl: {p90:.1f} days", zorder=4)
    ax.axvline(0, color=LIGHT_TEXT, linewidth=1, linestyle="-", alpha=0.5, zorder=1)

    ax.set_title("Delivery Delay Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Delay (days) — negative = early, positive = late")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", framealpha=0.3, fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart06_delay_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    early_pct = (delivered["delay_days"] < 0).mean() * 100
    late_pct = (delivered["is_late"] == 1).mean() * 100
    very_late = (delivered["delay_days"] > 14).mean() * 100

    print(f"\n  BUSINESS INSIGHT — Delay Distribution:")
    print(f"    {early_pct:.1f}% of orders arrive EARLY (before estimated date)")
    print(f"    {late_pct:.1f}% arrive LATE; {very_late:.1f}% are >14 days late")
    print(f"    Mean delay: {mean_delay:.1f} days | Median: {median_delay:.1f} days")
    print(f"    90th percentile: {p90:.1f} days")
    print(f"    Recommendation: {early_pct:.0f}% early delivery suggests estimates")
    print(f"    are deliberately conservative — this buffers satisfaction but may")
    print(f"    reduce urgency. The {very_late:.1f}% severely late orders (>14d) should")
    print(f"    trigger proactive customer communication to mitigate review damage.")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Review Score vs Delivery Delay
# ═══════════════════════════════════════════════════════════════════════════════

def chart_review_vs_delay(df):
    """
    Scatter plot of delay_days vs review_score with regression line.
    Also shows binned average review score per delay bucket for clarity.
    Reports R-squared and p-value.
    """
    analysis = df[df["delay_days"].notna() & df["review_score"].notna()].copy()
    analysis = analysis[(analysis["delay_days"] >= -20) & (analysis["delay_days"] <= 40)]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        analysis["delay_days"], analysis["review_score"]
    )
    r_sq = r_value ** 2

    # Bin delays for average scores
    bins = list(range(-20, 45, 5))
    analysis["delay_bin"] = pd.cut(analysis["delay_days"], bins=bins)
    binned = analysis.groupby("delay_bin", observed=True).agg(
        avg_score=("review_score", "mean"),
        count=("order_id", "count"),
        bin_mid=("delay_days", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Scatter (sample for performance)
    sample = analysis.sample(min(5000, len(analysis)), random_state=42)
    ax.scatter(sample["delay_days"], sample["review_score"],
               alpha=0.08, s=15, color=BLUE, zorder=1)

    # Binned averages
    ax.scatter(binned["bin_mid"], binned["avg_score"],
               s=binned["count"] / binned["count"].max() * 300,
               color=AMBER, edgecolors="white", linewidths=1, zorder=3,
               label="Avg score per 5-day bucket")

    # Regression line
    x_line = np.linspace(-20, 40, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=RED, linewidth=2.5, linestyle="--",
            label=f"Regression (R²={r_sq:.3f}, p<0.001)", zorder=4)

    # Zero delay line
    ax.axvline(0, color=LIGHT_TEXT, linewidth=1, linestyle=":", alpha=0.4)

    # Stats annotation
    ax.text(0.02, 0.02,
            f"slope = {slope:.4f}\nR² = {r_sq:.3f}\np = {p_value:.2e}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=AMBER, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_BG, edgecolor=AMBER, alpha=0.8))

    ax.set_title("Review Score vs Delivery Delay", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Delay (days) — negative = early, positive = late")
    ax.set_ylabel("Review Score (1–5)")
    ax.set_ylim(0.5, 5.5)
    ax.legend(loc="upper right", framealpha=0.3)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart07_review_vs_delay.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Calculate business impact
    avg_on_time = analysis[analysis["delay_days"] <= 0]["review_score"].mean()
    avg_late = analysis[analysis["delay_days"] > 0]["review_score"].mean()
    avg_very_late = analysis[analysis["delay_days"] > 14]["review_score"].mean()
    score_drop = avg_on_time - avg_late

    print(f"\n  BUSINESS INSIGHT — Review Score vs Delay:")
    print(f"    Linear regression: each additional day of delay reduces")
    print(f"    review score by {abs(slope):.4f} points (R²={r_sq:.3f}, p={p_value:.2e})")
    print(f"    Avg score — on-time/early: {avg_on_time:.2f} | late: {avg_late:.2f} | >14d late: {avg_very_late:.2f}")
    print(f"    Score drop from on-time to late: {score_drop:.2f} stars")
    print(f"    Recommendation: Late delivery costs ~{score_drop:.1f} review stars on average.")
    print(f"    With {analysis[analysis['delay_days']>0].shape[0]:,} late orders, this represents")
    print(f"    significant reputational damage and likely reduces repeat purchase rates.")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8 — On-Time Delivery Rate Over Time
# ═══════════════════════════════════════════════════════════════════════════════

def chart_ontime_trend(df):
    """
    Monthly on-time delivery rate from 2017-2018.
    Assesses whether operational performance scales with growth.
    """
    delivered = df[
        (df["delivery_days"].notna()) &
        (df["order_purchase_timestamp"] >= "2017-01-01") &
        (df["order_purchase_timestamp"] <= "2018-08-31")
    ].copy()

    # Deduplicate to order level to avoid multi-item inflation
    order_level = (
        delivered.drop_duplicates("order_id")[["order_id", "order_month", "is_late"]]
    )

    monthly = (
        order_level.groupby("order_month")
        .agg(
            total_orders=("order_id", "count"),
            on_time=("is_late", lambda x: (x == 0).sum()),
            late=("is_late", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    monthly["on_time_rate"] = monthly["on_time"] / monthly["total_orders"] * 100
    monthly["month_dt"] = monthly["order_month"].dt.to_timestamp()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(monthly["month_dt"], monthly["on_time_rate"], color=TEAL,
             linewidth=2.5, marker="o", markersize=7, zorder=3, label="On-time rate (%)")

    # Fill below 90% threshold
    ax1.axhline(y=90, color=AMBER, linewidth=1.5, linestyle="--", alpha=0.6,
                label="90% target")
    ax1.fill_between(monthly["month_dt"], monthly["on_time_rate"], 90,
                     where=monthly["on_time_rate"] < 90,
                     alpha=0.2, color=RED, interpolate=True)

    # Volume overlay
    ax2 = ax1.twinx()
    ax2.bar(monthly["month_dt"], monthly["total_orders"], alpha=0.2,
            color=BLUE, width=20, label="Order volume", zorder=1)
    ax2.set_ylabel("Order Volume", color=BLUE)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K"))

    ax1.set_title("On-Time Delivery Rate vs Order Volume", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("On-Time Delivery Rate (%)", color=TEAL)
    ax1.set_ylim(80, 100)
    ax1.grid(axis="y", alpha=0.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart08_ontime_trend.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Business insight
    first_half = monthly[monthly["month_dt"] < "2017-07-01"]["on_time_rate"].mean()
    second_half = monthly[monthly["month_dt"] >= "2018-01-01"]["on_time_rate"].mean()
    trend = "improving" if second_half > first_half else "deteriorating"

    # Correlation between volume and on-time rate
    corr, corr_p = stats.pearsonr(monthly["total_orders"], monthly["on_time_rate"])

    worst_month = monthly.loc[monthly["on_time_rate"].idxmin()]
    best_month = monthly.loc[monthly["on_time_rate"].idxmax()]

    print(f"\n  BUSINESS INSIGHT — On-Time Delivery Trend:")
    print(f"    H1 2017 avg on-time rate: {first_half:.1f}%")
    print(f"    2018 avg on-time rate: {second_half:.1f}%")
    print(f"    Trend: {trend} as platform scales")
    print(f"    Volume-vs-OTR correlation: r={corr:.3f} (p={corr_p:.3f})")
    print(f"    Best month: {best_month['month_dt'].strftime('%B %Y')} ({best_month['on_time_rate']:.1f}%)")
    print(f"    Worst month: {worst_month['month_dt'].strftime('%B %Y')} ({worst_month['on_time_rate']:.1f}%)")
    if corr < -0.3:
        print(f"    WARNING: Negative correlation suggests operations struggle to scale —")
        print(f"    as order volume grows, delivery reliability drops. This is a")
        print(f"    critical scalability bottleneck requiring logistics investment.")
    else:
        print(f"    Operations are scaling reasonably well with volume growth,")
        print(f"    but maintaining >90% on-time rate should remain a priority.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  PHASE 3 — DELIVERY & OPERATIONS ANALYSIS")
    print("█" * 70)

    set_chart_style()
    df = load_master()
    print(f"\n  Master dataframe loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    chart_delivery_by_state(df)
    chart_delay_distribution(df)
    chart_review_vs_delay(df)
    chart_ontime_trend(df)

    print("\n" + "=" * 70)
    print("✓ PHASE 3 COMPLETE — All 4 delivery charts saved to /outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
