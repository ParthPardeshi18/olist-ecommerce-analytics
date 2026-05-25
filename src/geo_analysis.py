"""
Phase 4 — Geospatial Analysis
==============================
Produces 3 visualizations:
  9. Brazil choropleth: Revenue by state (Folium HTML + PNG)
  10. Brazil choropleth: Late delivery rate by state (Folium HTML + PNG)
  11. Seller-to-customer state flow chart (top 10 routes)

Downloads Brazil GeoJSON automatically and matches state codes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import folium
import json
import requests
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
DATA_DIR = os.path.join(BASE_DIR, "data")


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
    return df


def get_brazil_geojson():
    """Download Brazil state-level GeoJSON, cache locally for reuse."""
    cache_path = os.path.join(DATA_DIR, "brazil_states.geojson")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    print(f"  Downloading Brazil GeoJSON from GitHub...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    geojson = response.json()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    print(f"  Cached to {cache_path}")

    return geojson


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Revenue Choropleth
# ═══════════════════════════════════════════════════════════════════════════════

def chart_revenue_choropleth(df, geojson):
    """
    Interactive Folium choropleth of total revenue by Brazilian state.
    Saves as HTML (interactive) and PNG (static matplotlib fallback).
    """
    # Aggregate order-level data by state
    orders = df.drop_duplicates("order_id")[["order_id", "customer_state", "payment_value",
                                              "delivery_days"]]
    state_data = (
        orders.groupby("customer_state")
        .agg(
            total_revenue=("payment_value", "sum"),
            order_count=("order_id", "count"),
            avg_delivery=("delivery_days", "mean"),
        )
        .reset_index()
    )

    # Map sigla property in GeoJSON to our state codes
    # The GeoJSON uses "sigla" for 2-letter state codes
    state_name_map = {}
    for feature in geojson["features"]:
        sigla = feature["properties"].get("sigla", "")
        name = feature["properties"].get("name", "")
        state_name_map[sigla] = name

    # Build Folium map
    m = folium.Map(location=[-14.2, -51.9], zoom_start=4,
                   tiles="CartoDB dark_matter")

    choropleth = folium.Choropleth(
        geo_data=geojson,
        data=state_data,
        columns=["customer_state", "total_revenue"],
        key_on="feature.properties.sigla",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="Total Revenue (R$)",
        nan_fill_color="#333333",
    ).add_to(m)

    # Add tooltips
    tooltip = folium.GeoJsonTooltip(
        fields=["sigla", "name"],
        aliases=["State Code:", "State:"],
        localize=True,
    )
    choropleth.geojson.add_child(tooltip)

    # Custom tooltip with revenue data
    state_lookup = state_data.set_index("customer_state").to_dict("index")
    for feature in geojson["features"]:
        sigla = feature["properties"].get("sigla", "")
        if sigla in state_lookup:
            info = state_lookup[sigla]
            feature["properties"]["revenue"] = f"R$ {info['total_revenue']:,.0f}"
            feature["properties"]["orders"] = f"{info['order_count']:,}"
            feature["properties"]["avg_del"] = f"{info['avg_delivery']:.1f} days"
        else:
            feature["properties"]["revenue"] = "N/A"
            feature["properties"]["orders"] = "N/A"
            feature["properties"]["avg_del"] = "N/A"

    folium.GeoJson(
        geojson,
        style_function=lambda x: {"fillOpacity": 0, "color": "transparent", "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["sigla", "name", "revenue", "orders", "avg_del"],
            aliases=["State:", "Name:", "Revenue:", "Orders:", "Avg Delivery:"],
        ),
    ).add_to(m)

    html_path = os.path.join(OUTPUT_DIR, "chart09_revenue_choropleth.html")
    m.save(html_path)
    print(f"  Saved HTML: {html_path}")

    # Static PNG fallback using matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    top5 = state_data.nlargest(5, "total_revenue")
    all_states = state_data.sort_values("total_revenue", ascending=True)

    bars = ax.barh(all_states["customer_state"], all_states["total_revenue"],
                   color=[RED if s in top5["customer_state"].values else BLUE
                          for s in all_states["customer_state"]],
                   alpha=0.85, zorder=3)
    ax.set_title("Revenue by State — Brazil", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Total Revenue (R$)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:.1f}M"))
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "chart09_revenue_by_state.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved PNG: {png_path}")

    # Insight
    sp = state_data[state_data["customer_state"] == "SP"]
    sp_pct = sp["total_revenue"].values[0] / state_data["total_revenue"].sum() * 100
    top3 = state_data.nlargest(3, "total_revenue")
    top3_pct = top3["total_revenue"].sum() / state_data["total_revenue"].sum() * 100

    print(f"\n  BUSINESS INSIGHT — Revenue Geography:")
    print(f"    SP alone generates {sp_pct:.1f}% of total revenue")
    print(f"    Top 3 states (SP, RJ, MG) generate {top3_pct:.1f}% of total revenue")
    print(f"    Revenue is heavily concentrated in the Southeast region")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 10 — Late Delivery Rate Choropleth
# ═══════════════════════════════════════════════════════════════════════════════

def chart_late_rate_choropleth(df, geojson):
    """
    Interactive Folium choropleth of late delivery rate by state.
    Reveals whether high-revenue states also have delivery problems.
    """
    delivered = df[df["delivery_days"].notna()].drop_duplicates("order_id")
    state_late = (
        delivered.groupby("customer_state")
        .agg(
            late_rate=("is_late", "mean"),
            total_orders=("order_id", "count"),
            avg_delay=("delay_days", "mean"),
        )
        .reset_index()
    )
    state_late["late_rate_pct"] = state_late["late_rate"] * 100

    # Folium map
    m = folium.Map(location=[-14.2, -51.9], zoom_start=4,
                   tiles="CartoDB dark_matter")

    folium.Choropleth(
        geo_data=geojson,
        data=state_late,
        columns=["customer_state", "late_rate_pct"],
        key_on="feature.properties.sigla",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="Late Delivery Rate (%)",
        nan_fill_color="#333333",
    ).add_to(m)

    # Tooltips
    state_lookup = state_late.set_index("customer_state").to_dict("index")
    for feature in geojson["features"]:
        sigla = feature["properties"].get("sigla", "")
        if sigla in state_lookup:
            info = state_lookup[sigla]
            feature["properties"]["late_pct"] = f"{info['late_rate_pct']:.1f}%"
            feature["properties"]["orders"] = f"{info['total_orders']:,}"
            feature["properties"]["avg_delay"] = f"{info['avg_delay']:.1f} days"
        else:
            feature["properties"]["late_pct"] = "N/A"
            feature["properties"]["orders"] = "N/A"
            feature["properties"]["avg_delay"] = "N/A"

    folium.GeoJson(
        geojson,
        style_function=lambda x: {"fillOpacity": 0, "color": "transparent", "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["sigla", "name", "late_pct", "orders", "avg_delay"],
            aliases=["State:", "Name:", "Late Rate:", "Orders:", "Avg Delay:"],
        ),
    ).add_to(m)

    html_path = os.path.join(OUTPUT_DIR, "chart10_late_rate_choropleth.html")
    m.save(html_path)
    print(f"\n  Saved HTML: {html_path}")

    # Static PNG
    fig, ax = plt.subplots(figsize=(12, 10))
    state_sorted = state_late.sort_values("late_rate_pct", ascending=True)

    def bar_color(rate):
        if rate <= 5:
            return TEAL
        elif rate <= 10:
            return AMBER
        return RED

    colors = [bar_color(r) for r in state_sorted["late_rate_pct"]]
    ax.barh(state_sorted["customer_state"], state_sorted["late_rate_pct"],
            color=colors, alpha=0.85, zorder=3)
    ax.set_title("Late Delivery Rate by State", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Late Delivery Rate (%)")
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "chart10_late_rate_by_state.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved PNG: {png_path}")

    # Cross-reference: high revenue + high late rate
    orders_by_state = df.drop_duplicates("order_id").groupby("customer_state")["payment_value"].sum()
    top_rev_states = orders_by_state.nlargest(5).index.tolist()
    high_rev_late = state_late[
        state_late["customer_state"].isin(top_rev_states)
    ].sort_values("late_rate_pct", ascending=False)

    print(f"\n  BUSINESS INSIGHT — Late Rate Geography:")
    print(f"    Top 5 revenue states and their late delivery rates:")
    for _, row in high_rev_late.iterrows():
        print(f"      {row['customer_state']}: {row['late_rate_pct']:.1f}% late ({row['total_orders']:,} orders)")
    worst_rev_state = high_rev_late.iloc[0]
    print(f"    RISK: {worst_rev_state['customer_state']} — a top revenue state — has")
    print(f"    {worst_rev_state['late_rate_pct']:.1f}% late delivery rate. This combination")
    print(f"    of high revenue and poor delivery creates a retention risk.")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 11 — Seller-Customer State Flow (Top 10 Routes)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_seller_customer_flow(df):
    """
    Horizontal bar chart showing top 10 seller_state → customer_state routes
    by order volume, revealing geographic supply chain concentration.
    """
    routes = (
        df.dropna(subset=["seller_state", "customer_state"])
        .groupby(["seller_state", "customer_state"])
        .agg(order_count=("order_id", "nunique"), revenue=("price", "sum"))
        .reset_index()
        .sort_values("order_count", ascending=False)
    )

    top10 = routes.head(10).sort_values("order_count", ascending=True)
    top10["route"] = top10["seller_state"] + " → " + top10["customer_state"]

    # Color: same-state routes in teal, cross-state in blue
    colors = [TEAL if row["seller_state"] == row["customer_state"] else BLUE
              for _, row in top10.iterrows()]

    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.barh(top10["route"], top10["order_count"],
                   color=colors, alpha=0.85, zorder=3, height=0.6)

    for bar, rev in zip(bars, top10["revenue"]):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                f"R$ {rev/1e6:.1f}M", ha="left", va="center",
                fontsize=10, color=AMBER, fontweight="bold")

    ax.set_title("Top 10 Seller → Customer State Routes", fontsize=16,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Number of Orders")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K"))
    ax.grid(axis="x", alpha=0.2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TEAL, markersize=10,
               label="Intra-state"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=BLUE, markersize=10,
               label="Inter-state"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart11_seller_customer_flow.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Insights
    total_orders = routes["order_count"].sum()
    intra_state = routes[routes["seller_state"] == routes["customer_state"]]["order_count"].sum()
    intra_pct = intra_state / total_orders * 100

    sp_sp = routes[(routes["seller_state"] == "SP") & (routes["customer_state"] == "SP")]
    sp_sp_pct = sp_sp["order_count"].sum() / total_orders * 100 if not sp_sp.empty else 0

    sp_sellers = routes[routes["seller_state"] == "SP"]["order_count"].sum()
    sp_seller_pct = sp_sellers / total_orders * 100

    print(f"\n  BUSINESS INSIGHT — Supply Chain Geography:")
    print(f"    Intra-state orders: {intra_pct:.1f}% of all orders")
    print(f"    SP→SP route alone: {sp_sp_pct:.1f}% of total orders")
    print(f"    SP sellers fulfill {sp_seller_pct:.1f}% of all orders nationally")
    print(f"    Top 10 routes account for {top10['order_count'].sum()/total_orders*100:.1f}% of order volume")
    print(f"    Recommendation: SP dominates both supply and demand — this geographic")
    print(f"    concentration creates efficiency but also single-point-of-failure risk.")
    print(f"    Developing seller bases in MG and PR could reduce delivery times")
    print(f"    and diversify the supply chain for non-SE customers.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  PHASE 4 — GEOSPATIAL ANALYSIS")
    print("█" * 70)

    set_chart_style()
    df = load_master()
    print(f"\n  Master dataframe loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    geojson = get_brazil_geojson()
    print(f"  GeoJSON loaded: {len(geojson['features'])} state features")

    chart_revenue_choropleth(df, geojson)
    chart_late_rate_choropleth(df, geojson)
    chart_seller_customer_flow(df)

    print("\n" + "=" * 70)
    print("✓ PHASE 4 COMPLETE — All geospatial outputs saved to /outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
