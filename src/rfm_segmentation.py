"""
Phase 5 — RFM Customer Segmentation
=====================================
Calculates Recency, Frequency, Monetary for each customer_unique_id.
Runs K-Means (k=4) with elbow plot justification.
Labels segments meaningfully, handles the one-time buyer reality.

Produces:
  12. RFM segment bubble chart
  13. Segment revenue waterfall / summary
  - Elbow plot saved to outputs
  - rfm_segments.csv saved to outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
PALETTE = [BLUE, TEAL, AMBER, RED, "#9B59B6"]

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
    """Load master dataframe with date parsing."""
    df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "master_df.csv"),
        parse_dates=["order_purchase_timestamp"],
        low_memory=False,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# RFM CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_rfm(df):
    """
    Calculate Recency, Frequency, Monetary per customer_unique_id.
    Snapshot date = max order date + 1 day.
    Only includes delivered orders to ensure valid revenue attribution.
    """
    delivered = df[df["order_status"] == "delivered"].copy()

    snapshot_date = delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    print(f"  Snapshot date: {snapshot_date.strftime('%Y-%m-%d')}")

    # Order-level aggregation first (avoid double-counting multi-item orders)
    order_level = (
        delivered.drop_duplicates("order_id")
        [["order_id", "customer_unique_id", "order_purchase_timestamp", "payment_value"]]
    )

    rfm = (
        order_level.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("payment_value", "sum"),
        )
        .reset_index()
    )

    print(f"\n  RFM stats:")
    print(f"    Total unique customers: {len(rfm):,}")
    print(f"    Recency — mean: {rfm['recency'].mean():.0f}d, median: {rfm['recency'].median():.0f}d")
    print(f"    Frequency — mean: {rfm['frequency'].mean():.2f}, median: {rfm['frequency'].median():.0f}")
    print(f"    Monetary — mean: R${rfm['monetary'].mean():.2f}, median: R${rfm['monetary'].median():.2f}")

    one_time = (rfm["frequency"] == 1).mean() * 100
    print(f"\n    One-time buyers: {one_time:.1f}% of customers")
    print(f"    NOTE: Olist's marketplace model means most customers order once.")
    print(f"    This is typical for marketplace platforms where brand loyalty")
    print(f"    attaches to the product seller, not the platform itself.")

    return rfm


# ═══════════════════════════════════════════════════════════════════════════════
# ELBOW PLOT & K-MEANS
# ═══════════════════════════════════════════════════════════════════════════════

def run_clustering(rfm):
    """
    Run K-Means with k=4 after standardization.
    Save elbow plot to justify k choice.
    Label segments based on RFM centroid profiles.
    """
    # Log-transform monetary to handle right skew
    rfm_features = rfm[["recency", "frequency", "monetary"]].copy()
    rfm_features["monetary"] = np.log1p(rfm_features["monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_features)

    # Elbow plot
    inertias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(K_range), inertias, color=BLUE, linewidth=2.5, marker="o", markersize=8)
    ax.axvline(x=4, color=AMBER, linewidth=2, linestyle="--", alpha=0.7, label="k=4 chosen")
    ax.set_title("Elbow Plot — K-Means on RFM Features", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.legend(framealpha=0.3)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "elbow_plot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Final clustering with k=4
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(X_scaled)

    # Label segments based on centroid characteristics
    centroids = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    print(f"\n  Cluster centroids:")
    print(centroids.to_string())

    # Assign meaningful labels: sort by monetary desc to find champions
    centroid_df = centroids.reset_index()
    centroid_df["label"] = ""

    # Champions: lowest recency + highest monetary
    # Lost: highest recency + lowest monetary
    # Strategy: rank each metric and combine
    centroid_df["r_rank"] = centroid_df["recency"].rank()  # low recency = good = low rank
    centroid_df["f_rank"] = centroid_df["frequency"].rank(ascending=False)  # high freq = good = low rank
    centroid_df["m_rank"] = centroid_df["monetary"].rank(ascending=False)  # high monetary = good = low rank
    centroid_df["score"] = centroid_df["r_rank"] + centroid_df["f_rank"] + centroid_df["m_rank"]

    sorted_clusters = centroid_df.sort_values("score")
    labels = ["Champions", "Loyal Customers", "At Risk", "Lost / Dormant"]

    label_map = {}
    for i, (_, row) in enumerate(sorted_clusters.iterrows()):
        label_map[row["cluster"]] = labels[i]

    rfm["segment"] = rfm["cluster"].map(label_map)

    print(f"\n  Segment assignments:")
    for cluster_id, label in sorted(label_map.items()):
        seg = rfm[rfm["cluster"] == cluster_id]
        print(f"    {label}: {len(seg):,} customers "
              f"(R={seg['recency'].mean():.0f}d, F={seg['frequency'].mean():.2f}, "
              f"M=R${seg['monetary'].mean():.2f})")

    return rfm


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 12 — RFM Segment Bubble Chart
# ═══════════════════════════════════════════════════════════════════════════════

def chart_rfm_bubble(rfm):
    """
    Bubble chart: X=Recency, Y=Frequency, size=Monetary, color=segment.
    Labels each cluster centroid.
    """
    segment_order = ["Champions", "Loyal Customers", "At Risk", "Lost / Dormant"]
    color_map = dict(zip(segment_order, PALETTE[:4]))

    fig, ax = plt.subplots(figsize=(14, 8))

    for seg in segment_order:
        subset = rfm[rfm["segment"] == seg]
        # Sample for performance
        sample = subset.sample(min(2000, len(subset)), random_state=42)

        sizes = np.clip(sample["monetary"] / sample["monetary"].quantile(0.99) * 100, 5, 300)

        ax.scatter(
            sample["recency"], sample["frequency"],
            s=sizes, alpha=0.35, color=color_map[seg],
            label=f"{seg} (n={len(subset):,})", edgecolors="white",
            linewidths=0.3, zorder=2,
        )

    # Plot centroids
    centroids = rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean()
    for seg in segment_order:
        if seg in centroids.index:
            c = centroids.loc[seg]
            ax.scatter(c["recency"], c["frequency"], s=200, color=color_map[seg],
                       edgecolors="white", linewidths=2, zorder=5, marker="*")
            ax.annotate(
                seg, xy=(c["recency"], c["frequency"]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=10, fontweight="bold", color=color_map[seg],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG,
                          edgecolor=color_map[seg], alpha=0.8),
            )

    ax.set_title("RFM Customer Segmentation", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Recency (days since last order)")
    ax.set_ylabel("Frequency (number of orders)")
    ax.legend(loc="upper right", framealpha=0.3, fontsize=10)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart12_rfm_bubble.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 13 — Segment Revenue Waterfall / Summary
# ═══════════════════════════════════════════════════════════════════════════════

def chart_segment_summary(rfm):
    """
    Combined visualization showing revenue contribution, customer count,
    and average order value per segment — presented as grouped bar chart.
    """
    segment_order = ["Champions", "Loyal Customers", "At Risk", "Lost / Dormant"]
    color_map = dict(zip(segment_order, PALETTE[:4]))

    summary = (
        rfm.groupby("segment")
        .agg(
            customer_count=("customer_unique_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_monetary=("monetary", "mean"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
        )
        .reindex(segment_order)
        .reset_index()
    )
    total_rev = summary["total_revenue"].sum()
    total_cust = summary["customer_count"].sum()
    summary["pct_revenue"] = summary["total_revenue"] / total_rev * 100
    summary["pct_customers"] = summary["customer_count"] / total_cust * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: Revenue contribution
    bars1 = axes[0].bar(summary["segment"], summary["pct_revenue"],
                        color=[color_map[s] for s in summary["segment"]], alpha=0.85, zorder=3)
    axes[0].set_title("Revenue Contribution (%)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("%")
    for bar, val in zip(bars1, summary["pct_revenue"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold", color=LIGHT_TEXT)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.2)

    # Panel 2: Customer count
    bars2 = axes[1].bar(summary["segment"], summary["pct_customers"],
                        color=[color_map[s] for s in summary["segment"]], alpha=0.85, zorder=3)
    axes[1].set_title("Customer Share (%)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("%")
    for bar, val in zip(bars2, summary["pct_customers"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold", color=LIGHT_TEXT)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.2)

    # Panel 3: Avg monetary value
    bars3 = axes[2].bar(summary["segment"], summary["avg_monetary"],
                        color=[color_map[s] for s in summary["segment"]], alpha=0.85, zorder=3)
    axes[2].set_title("Avg Customer Value (R$)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("R$")
    for bar, val in zip(bars3, summary["avg_monetary"]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"R${val:.0f}", ha="center", fontsize=10, fontweight="bold", color=LIGHT_TEXT)
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle("Customer Segment Performance", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart13_segment_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Print summary table
    champs = summary[summary["segment"] == "Champions"].iloc[0]
    print(f"\n  BUSINESS INSIGHT — Customer Segmentation:")
    print(f"    Segment Summary:")
    print(f"    {'Segment':<20} {'Customers':>10} {'% Cust':>8} {'Revenue':>12} {'% Rev':>8} {'Avg Value':>10}")
    print(f"    {'─'*68}")
    for _, row in summary.iterrows():
        print(f"    {row['segment']:<20} {row['customer_count']:>10,} {row['pct_customers']:>7.1f}% "
              f"R${row['total_revenue']:>10,.0f} {row['pct_revenue']:>7.1f}% R${row['avg_monetary']:>9,.2f}")

    print(f"\n    Champions represent {champs['pct_customers']:.1f}% of customers")
    print(f"    but generate {champs['pct_revenue']:.1f}% of revenue")
    print(f"    Their avg value (R${champs['avg_monetary']:.0f}) is "
          f"{champs['avg_monetary']/summary['avg_monetary'].mean():.1f}x the overall average")
    print(f"    Recommendation: Retention investment in Champions is high-ROI —")
    print(f"    losing even 10% of this segment would cost R${champs['total_revenue']*0.1:,.0f}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  PHASE 5 — RFM CUSTOMER SEGMENTATION")
    print("█" * 70)

    set_chart_style()
    df = load_master()
    print(f"\n  Master dataframe loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    rfm = calculate_rfm(df)
    rfm = run_clustering(rfm)
    chart_rfm_bubble(rfm)
    summary = chart_segment_summary(rfm)

    # Save RFM segments
    rfm_path = os.path.join(OUTPUT_DIR, "rfm_segments.csv")
    rfm.to_csv(rfm_path, index=False)
    print(f"\n  Saved RFM data: {rfm_path} ({len(rfm):,} customers)")

    print("\n" + "=" * 70)
    print("✓ PHASE 5 COMPLETE — RFM segmentation saved to /outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
