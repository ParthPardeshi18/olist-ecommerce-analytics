"""
Phase 7 — Streamlit Multi-Tab Dashboard
========================================
5 tabs: Executive Overview, Geography, Customer Segments,
Delivery Intelligence, Delay Risk Predictor.

Launch:  streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import json

# ─── config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Business Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

BLUE = "#378ADD"
TEAL = "#1D9E75"
RED = "#E24B4A"
AMBER = "#EF9F27"


# ─── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── global page ── */
    .stApp { background-color: #0f0f1a; color: #e0e0e0; }

    /* ── metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 18px 20px;
    }
    [data-testid="stMetricLabel"]  { color: #a0a8c0 !important; font-size: 13px; }
    [data-testid="stMetricValue"]  { color: #e0e0e0 !important; font-size: 26px; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 13px; }

    /* ── tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #1a1a2e;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 18px;
        color: #a0a8c0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #378ADD !important;
        color: #ffffff !important;
    }

    /* ── dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .dvn-scroller { background: #1a1a2e !important; }

    /* ── section headers ── */
    h1, h2, h3 { color: #e0e0e0 !important; }

    /* ── divider ── */
    hr { border-color: #2a2a4a; }

    /* ── selectbox / input labels ── */
    label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #a0a8c0 !important;
    }

    /* ── sidebar (if open) ── */
    [data-testid="stSidebar"] { background: #1a1a2e; }

    /* ── buttons ── */
    .stButton > button {
        background: #378ADD;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
    }
    .stButton > button:hover { background: #2a6bb5; }
</style>
""", unsafe_allow_html=True)


# ─── data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_master():
    """Load master dataframe with caching."""
    df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "master_df.csv"),
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                      "order_estimated_delivery_date"],
        low_memory=False,
    )
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    return df


@st.cache_data
def load_rfm():
    """Load RFM segments."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, "rfm_segments.csv"))


@st.cache_resource
def load_model():
    """Load trained XGBoost model and scaler."""
    model = joblib.load(os.path.join(OUTPUT_DIR, "delay_model.pkl"))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    return model, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Overview
# ═══════════════════════════════════════════════════════════════════════════════

def tab_executive(df):
    """KPI cards, monthly GMV trend, top categories."""
    st.header("Executive Overview")

    orders = df.drop_duplicates("order_id")
    delivered = orders[orders["order_status"] == "delivered"]

    total_gmv = orders["payment_value"].sum()
    total_orders = len(orders)
    avg_order_value = orders["payment_value"].mean()
    on_time_rate = (delivered["is_late"] == 0).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total GMV", f"R$ {total_gmv/1e6:.2f}M")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Avg Order Value", f"R$ {avg_order_value:.2f}")
    col4.metric("On-Time Delivery", f"{on_time_rate:.1f}%")

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Monthly GMV Trend")
        monthly = (
            orders[orders["order_purchase_timestamp"] >= "2017-01-01"]
            .groupby(orders["order_purchase_timestamp"].dt.to_period("M"))["payment_value"]
            .sum()
            .reset_index()
        )
        monthly.columns = ["month", "gmv"]
        monthly["month_dt"] = monthly["month"].dt.to_timestamp()
        monthly["rolling_3m"] = monthly["gmv"].rolling(3, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month_dt"], y=monthly["gmv"],
            mode="lines+markers", name="Monthly GMV",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=monthly["month_dt"], y=monthly["rolling_3m"],
            mode="lines", name="3M Rolling Avg",
            line=dict(color=AMBER, width=2, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            yaxis_title="Revenue (R$)",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top 5 Categories")
        cat_rev = (
            df.groupby("product_category_name_english")["price"]
            .sum()
            .nlargest(5)
            .reset_index()
        )
        cat_rev.columns = ["category", "revenue"]

        fig = px.bar(
            cat_rev, y="category", x="revenue", orientation="h",
            color_discrete_sequence=[BLUE],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            xaxis_title="Revenue (R$)",
            yaxis_title="",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Geography
# ═══════════════════════════════════════════════════════════════════════════════

def tab_geography(df):
    """Choropleth map, state selector, seller vs customer comparison."""
    st.header("Geographic Analysis")

    orders = df.drop_duplicates("order_id")

    col_map, col_detail = st.columns([2, 1])

    with col_map:
        state_data = (
            orders.groupby("customer_state")
            .agg(
                revenue=("payment_value", "sum"),
                order_count=("order_id", "count"),
                avg_delivery=("delivery_days", "mean"),
                late_rate=("is_late", "mean"),
            )
            .reset_index()
        )
        state_data["late_rate_pct"] = state_data["late_rate"] * 100

        # Brazilian state coordinates for plotting
        br_state_coords = {
            "AC": (-8.77, -70.55), "AL": (-9.71, -35.73), "AM": (-3.07, -61.66),
            "AP": (1.41, -51.77), "BA": (-12.96, -38.51), "CE": (-3.71, -38.54),
            "DF": (-15.83, -47.86), "ES": (-19.19, -40.34), "GO": (-16.64, -49.31),
            "MA": (-2.55, -44.28), "MG": (-18.10, -44.38), "MS": (-20.51, -54.54),
            "MT": (-12.64, -55.42), "PA": (-5.53, -52.29), "PB": (-7.06, -35.55),
            "PE": (-8.28, -35.07), "PI": (-8.28, -43.68), "PR": (-24.89, -51.55),
            "RJ": (-22.84, -43.15), "RN": (-5.22, -36.52), "RO": (-11.22, -62.80),
            "RR": (1.89, -61.22), "RS": (-30.01, -51.22), "SC": (-27.33, -49.44),
            "SE": (-10.90, -37.07), "SP": (-23.55, -46.64), "TO": (-10.25, -48.25),
        }

        state_data["lat"] = state_data["customer_state"].map(lambda x: br_state_coords.get(x, (0, 0))[0])
        state_data["lon"] = state_data["customer_state"].map(lambda x: br_state_coords.get(x, (0, 0))[1])

        metric = st.radio("Color by:", ["Revenue", "Late Rate (%)"], horizontal=True)
        color_col = "revenue" if metric == "Revenue" else "late_rate_pct"
        color_scale = "Blues" if metric == "Revenue" else "OrRd"

        fig = px.scatter_geo(
            state_data,
            lat="lat", lon="lon",
            size="order_count",
            color=color_col,
            hover_name="customer_state",
            hover_data={"revenue": ":,.0f", "order_count": ":,", "late_rate_pct": ":.1f"},
            color_continuous_scale=color_scale,
            scope="south america",
            size_max=40,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            geo=dict(
                bgcolor="#1a1a2e",
                showland=True, landcolor="#2a2a4a",
                showocean=True, oceancolor="#1a1a2e",
                fitbounds="locations",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_detail:
        st.subheader("State Details")
        states = sorted(state_data["customer_state"].unique())
        selected = st.selectbox("Select state:", states, index=states.index("SP"))

        s = state_data[state_data["customer_state"] == selected].iloc[0]
        st.metric("Revenue", f"R$ {s['revenue']:,.0f}")
        st.metric("Orders", f"{s['order_count']:,}")
        st.metric("Avg Delivery", f"{s['avg_delivery']:.1f} days")
        st.metric("Late Rate", f"{s['late_rate_pct']:.1f}%")

    st.divider()
    st.subheader("Seller vs Customer State Comparison")

    seller_share = df.groupby("seller_state")["order_id"].nunique().reset_index()
    seller_share.columns = ["state", "seller_orders"]
    cust_share = orders.groupby("customer_state")["order_id"].count().reset_index()
    cust_share.columns = ["state", "customer_orders"]
    comparison = seller_share.merge(cust_share, on="state", how="outer").fillna(0)
    comparison = comparison.sort_values("customer_orders", ascending=False).head(10)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=comparison["state"], y=comparison["seller_orders"],
                         name="Fulfilled by sellers in state", marker_color=TEAL))
    fig.add_trace(go.Bar(x=comparison["state"], y=comparison["customer_orders"],
                         name="Ordered by customers in state", marker_color=BLUE))
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Customer Segments
# ═══════════════════════════════════════════════════════════════════════════════

def tab_segments(df, rfm):
    """RFM bubble chart, segment summary table, segment filter."""
    st.header("Customer Segmentation (RFM)")

    color_map = {
        "Champions": BLUE, "Loyal Customers": TEAL,
        "At Risk": AMBER, "Lost / Dormant": RED,
    }

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.subheader("Recency vs Spend by Segment")
        st.caption("97% of Olist customers order once — Recency × Monetary reveals segment separation better than frequency.")

        sample = rfm.sample(min(4000, len(rfm)), random_state=42).copy()
        # Log-scale monetary for better spread; clip extreme outliers
        sample["monetary_log"] = np.log1p(sample["monetary"])
        cap = sample["monetary"].quantile(0.98)
        bubble_size = np.clip(sample["monetary"], 1, cap)

        fig = px.scatter(
            sample,
            x="recency",
            y="monetary_log",
            color="segment",
            color_discrete_map=color_map,
            size=bubble_size,
            size_max=18,
            opacity=0.55,
            hover_data={"monetary": ":.2f", "recency": True, "frequency": True, "monetary_log": False},
            labels={
                "recency": "Recency (days since last order)",
                "monetary_log": "Log(Spend + 1)  →  higher = more valuable",
                "segment": "Segment",
            },
        )
        # Segment centroid labels
        for seg, color in color_map.items():
            sub = sample[sample["segment"] == seg]
            if len(sub) == 0:
                continue
            cx, cy = sub["recency"].mean(), sub["monetary_log"].mean()
            fig.add_annotation(
                x=cx, y=cy, text=f"<b>{seg}</b>",
                showarrow=False, font=dict(size=12, color=color),
                bgcolor="rgba(15,15,26,0.75)", bordercolor=color,
                borderwidth=1, borderpad=4,
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            height=480,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0,
                font=dict(size=12), bgcolor="rgba(26,26,46,0.8)",
                bordercolor="#2a2a4a", borderwidth=1,
            ),
        )
        fig.update_xaxes(gridcolor="#2a2a4a", zerolinecolor="#2a2a4a")
        fig.update_yaxes(gridcolor="#2a2a4a", zerolinecolor="#2a2a4a")
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.subheader("Segment Summary")
        summary = (
            rfm.groupby("segment")
            .agg(
                customers=("customer_unique_id", "count"),
                avg_monetary=("monetary", "mean"),
                avg_recency=("recency", "mean"),
                total_revenue=("monetary", "sum"),
            )
            .reset_index()
        )
        summary["pct_revenue"] = summary["total_revenue"] / summary["total_revenue"].sum() * 100
        summary = summary.sort_values("total_revenue", ascending=False)

        seg_color_map = {"Champions": BLUE, "Loyal Customers": TEAL,
                         "At Risk": AMBER, "Lost / Dormant": RED}

        rows = ""
        for _, r in summary.iterrows():
            dot_color = seg_color_map.get(r["segment"], "#aaa")
            rows += (
                f"<tr>"
                f"<td><span style='display:inline-block;width:10px;height:10px;"
                f"border-radius:50%;background:{dot_color};margin-right:7px'></span>"
                f"<b>{r['segment']}</b></td>"
                f"<td>{r['customers']:,}</td>"
                f"<td>R$ {r['avg_monetary']:.0f}</td>"
                f"<td>{r['avg_recency']:.0f}d</td>"
                f"<td>R$ {r['total_revenue']/1e6:.2f}M</td>"
                f"<td><b>{r['pct_revenue']:.1f}%</b></td>"
                f"</tr>"
            )

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px;color:#e0e0e0;">
          <thead>
            <tr style="border-bottom:2px solid #2a2a4a;color:#a0a8c0;text-align:left;">
              <th style="padding:8px 6px">Segment</th>
              <th style="padding:8px 6px">Customers</th>
              <th style="padding:8px 6px">Avg Spend</th>
              <th style="padding:8px 6px">Recency</th>
              <th style="padding:8px 6px">Revenue</th>
              <th style="padding:8px 6px">% Rev</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        <style>
          table tr:hover {{ background: #1e2a3a; }}
          table td {{ padding: 9px 6px; border-bottom: 1px solid #2a2a4a; }}
        </style>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Explore Segment")
    seg_order = ["Champions", "Loyal Customers", "At Risk", "Lost / Dormant"]
    selected_seg = st.selectbox("Filter by segment:", seg_order)
    seg_data = rfm[rfm["segment"] == selected_seg].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", f"{len(seg_data):,}")
    col2.metric("Avg Value", f"R$ {seg_data['monetary'].mean():.2f}")
    col3.metric("Avg Recency", f"{seg_data['recency'].mean():.0f} days")

    # Render top 20 as HTML table — avoids dark-mode invisibility of st.dataframe
    top20 = seg_data[["customer_unique_id", "recency", "frequency", "monetary"]].head(20)
    table_rows = "".join(
        f"<tr><td style='font-family:monospace;font-size:11px'>{r['customer_unique_id'][:16]}…</td>"
        f"<td>{r['recency']:.0f}d</td>"
        f"<td>{r['frequency']:.0f}</td>"
        f"<td>R$ {r['monetary']:.2f}</td></tr>"
        for _, r in top20.iterrows()
    )
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;color:#e0e0e0;margin-top:12px">
      <thead>
        <tr style="border-bottom:2px solid #2a2a4a;color:#a0a8c0;text-align:left;">
          <th style="padding:8px 6px">Customer ID</th>
          <th style="padding:8px 6px">Recency</th>
          <th style="padding:8px 6px">Orders</th>
          <th style="padding:8px 6px">Total Spend</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Delivery Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

def tab_delivery(df):
    """Late delivery KPIs, state performance, review vs delay scatter."""
    st.header("Delivery Intelligence")

    delivered = df[df["delivery_days"].notna()].drop_duplicates("order_id")

    late_rate = (delivered["is_late"] == 1).mean() * 100
    avg_delivery = delivered["delivery_days"].mean()
    avg_delay_late = delivered[delivered["is_late"] == 1]["delay_days"].mean()

    # Trend arrow
    recent = delivered[delivered["order_purchase_timestamp"] >= "2018-06-01"]
    older = delivered[
        (delivered["order_purchase_timestamp"] >= "2018-01-01") &
        (delivered["order_purchase_timestamp"] < "2018-06-01")
    ]
    delta = (recent["is_late"] == 1).mean() * 100 - (older["is_late"] == 1).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Late Delivery Rate", f"{late_rate:.1f}%", delta=f"{delta:+.1f}pp",
                delta_color="inverse")
    col2.metric("Avg Delivery Time", f"{avg_delivery:.1f} days")
    col3.metric("Avg Delay (when late)", f"{avg_delay_late:.1f} days")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Delivery by State")
        state_perf = (
            delivered.groupby("customer_state")
            .agg(avg_days=("delivery_days", "mean"), late_pct=("is_late", "mean"))
            .reset_index()
            .sort_values("avg_days")
        )
        state_perf["late_pct"] *= 100
        state_perf["color"] = state_perf["avg_days"].apply(
            lambda d: TEAL if d <= 10 else (AMBER if d <= 20 else RED)
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=state_perf["customer_state"], x=state_perf["avg_days"],
            orientation="h",
            marker_color=state_perf["color"],
            name="Avg Delivery Days",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Average Delivery Days",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Review Score vs Delay")
        sample = delivered.dropna(subset=["delay_days", "review_score"])
        sample = sample[
            (sample["delay_days"] >= -20) & (sample["delay_days"] <= 40)
        ].sample(min(3000, len(sample)), random_state=42)

        fig = px.scatter(
            sample, x="delay_days", y="review_score",
            opacity=0.2, color_discrete_sequence=[BLUE],
            trendline="ols",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=600,
            xaxis_title="Delay (days)",
            yaxis_title="Review Score",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Delay Risk Predictor
# ═══════════════════════════════════════════════════════════════════════════════

def tab_predictor(df):
    """Interactive form for predicting late delivery risk."""
    st.header("Delay Risk Predictor")
    st.write("Enter order details to estimate the probability of late delivery.")

    model, scaler = load_model()

    states = sorted(df["customer_state"].dropna().unique())
    categories = sorted(df["product_category_name_english"].dropna().unique())

    col1, col2 = st.columns(2)

    with col1:
        seller_state = st.selectbox("Seller State", states, index=states.index("SP"))
        customer_state = st.selectbox("Customer State", states, index=states.index("RJ"))
        category = st.selectbox("Product Category", categories)
        weight = st.number_input("Product Weight (g)", min_value=50, max_value=40000,
                                 value=1000, step=100)
        volume = st.number_input("Product Volume (cm³)", min_value=100, max_value=500000,
                                 value=5000, step=500)

    with col2:
        price = st.number_input("Price (R$)", min_value=10.0, max_value=5000.0,
                                value=150.0, step=10.0)
        freight = st.number_input("Freight Value (R$)", min_value=5.0, max_value=500.0,
                                  value=25.0, step=5.0)
        installments = st.slider("Payment Installments", 1, 12, 3)
        day_of_week = st.selectbox("Order Day", ["Monday", "Tuesday", "Wednesday",
                                                   "Thursday", "Friday", "Saturday", "Sunday"])
        photos = st.slider("Product Photos", 1, 10, 3)

    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    same_state = 1 if seller_state == customer_state else 0
    payment_value = price + freight
    freight_ratio = freight / payment_value if payment_value > 0 else 0

    # Get historical averages for the seller state and category
    cat_late = df[df["product_category_name_english"] == category]["is_late"].mean()
    cat_late = cat_late if not np.isnan(cat_late) else 0.08

    # Estimated days: use median for the customer state
    state_median_est = df[df["customer_state"] == customer_state]["estimated_days"].median()
    est_days = state_median_est if not np.isnan(state_median_est) else 25

    features = np.array([[
        est_days,               # estimated_days
        weight,                 # product_weight_g
        volume,                 # product_volume_cm3
        freight,                # freight_value
        price,                  # price
        payment_value,          # payment_value
        same_state,             # seller_customer_same_state
        is_weekend,             # is_weekend_order
        freight_ratio,          # freight_to_value_ratio
        0,                      # seller_avg_delay (unknown for new order)
        cat_late,               # category_late_rate
        installments,           # payment_installments
        photos,                 # product_photos_qty
    ]])

    if st.button("Predict Delay Risk", type="primary"):
        prob = model.predict_proba(features)[0][1]

        st.divider()

        if prob < 0.3:
            risk_level = "LOW RISK"
            risk_color = TEAL
            emoji = "✅"
        elif prob < 0.6:
            risk_level = "MEDIUM RISK"
            risk_color = AMBER
            emoji = "⚠️"
        else:
            risk_level = "HIGH RISK"
            risk_color = RED
            emoji = "🚨"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    border: 2px solid {risk_color}; border-radius: 15px;
                    padding: 30px; text-align: center; margin: 20px 0;">
            <h1 style="color: {risk_color}; margin: 0;">{emoji} {risk_level}</h1>
            <h2 style="color: {risk_color}; margin: 10px 0;">
                {prob*100:.1f}% probability of late delivery
            </h2>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Risk Drivers")
        drivers = {
            "Estimated delivery time": f"{est_days:.0f} days",
            "Same state seller": "Yes" if same_state else "No (cross-state)",
            "Freight-to-value ratio": f"{freight_ratio:.2%}",
            "Category late rate": f"{cat_late:.1%}",
            "Weekend order": "Yes" if is_weekend else "No",
            "Product weight": f"{weight:,}g",
        }
        for driver, value in drivers.items():
            st.write(f"- **{driver}**: {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("📦 Olist Business Analytics Dashboard")
    st.caption("Brazilian E-Commerce Insights | 2016–2018 | 99K+ Orders")

    df = load_master()
    rfm = load_rfm()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Overview",
        "🗺️ Geography",
        "👥 Customer Segments",
        "🚚 Delivery Intelligence",
        "🔮 Delay Risk Predictor",
    ])

    with tab1:
        tab_executive(df)
    with tab2:
        tab_geography(df)
    with tab3:
        tab_segments(df, rfm)
    with tab4:
        tab_delivery(df)
    with tab5:
        tab_predictor(df)


if __name__ == "__main__":
    main()
