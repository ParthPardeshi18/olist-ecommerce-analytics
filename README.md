# Olist Business Analytics

End-to-end business analytics project on the **Olist Brazilian E-Commerce dataset** (99K+ orders, 9 tables, Sep 2016 – Oct 2018). Covers revenue analysis, delivery operations, geospatial mapping, customer segmentation, and ML-based late delivery prediction — all with publication-quality visualizations and actionable business insights.

## Key Findings

1. **Seller Concentration Risk** — Gini coefficient 0.79: the top 18% of sellers generate 80% of platform revenue. A seller development program for mid-tier sellers is critical for diversification.

2. **Delivery Scalability Bottleneck** — On-time rate dropped from 95.5% to 90.8% as order volume grew (r = -0.54 correlation). Logistics capacity is not scaling with demand.

3. **Late Delivery Costs 1.7 Review Stars** — On-time orders average 4.22 stars vs 2.55 for late orders. Orders delayed >14 days average just 1.61 stars, representing significant reputational damage.

4. **Rio de Janeiro Retention Risk** — 2nd-highest revenue state but worst late delivery rate (13.5%) among the top 5 states. Highest-priority market for logistics investment.

5. **SP Supply Chain Dominance** — SP sellers fulfill 70.8% of all orders nationally. The SP→SP route alone is 31.8% of volume, creating efficiency but also single-point-of-failure risk.

## Project Structure

```
olist-analytics/
├── data/                          # Raw CSV files (9 Olist tables + GeoJSON)
├── src/
│   ├── data_ingestion.py          # Phase 1: Load, validate, merge, engineer features
│   ├── revenue_analysis.py        # Phase 2: GMV trend, categories, payments, Lorenz
│   ├── delivery_analysis.py       # Phase 3: State performance, delays, reviews, OTR
│   ├── geo_analysis.py            # Phase 4: Choropleths, seller-customer flows
│   ├── rfm_segmentation.py        # Phase 5: RFM calculation, K-Means, segment analysis
│   ├── delay_model.py             # Phase 6: XGBoost late delivery predictor + SHAP
│   └── app.py                     # Phase 7: Streamlit multi-tab dashboard
├── outputs/                       # All charts (PNG), maps (HTML), model files
│   ├── chart01–16_*.png           # 16 publication-quality charts
│   ├── chart09/10_*choropleth.html # Interactive Folium maps
│   ├── delay_model.pkl            # Trained XGBoost model
│   ├── scaler.pkl                 # Feature scaler
│   ├── rfm_segments.csv           # Customer segment assignments
│   └── project_summary.txt        # Quantified findings and recommendations
└── README.md
```

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap plotly folium streamlit scipy requests joblib
```

### Run Analysis Scripts

Each script is self-contained and runs end-to-end:

```bash
python src/data_ingestion.py       # Phase 1: Creates master_df.csv
python src/revenue_analysis.py     # Phase 2: Charts 1-4
python src/delivery_analysis.py    # Phase 3: Charts 5-8
python src/geo_analysis.py         # Phase 4: Charts 9-11 + HTML maps
python src/rfm_segmentation.py     # Phase 5: Charts 12-13 + rfm_segments.csv
python src/delay_model.py          # Phase 6: Charts 14-16 + model files
```

### Launch Dashboard

```bash
streamlit run src/app.py
```

Opens a 5-tab interactive dashboard at `http://localhost:8501`:
- **Executive Overview** — KPI cards, GMV trend, top categories
- **Geography** — Interactive Brazil map, state selector, seller-customer comparison
- **Customer Segments** — RFM bubble chart, segment explorer
- **Delivery Intelligence** — State performance, review-delay scatter
- **Delay Risk Predictor** — Input form with real-time probability prediction

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.14 | Core language |
| pandas | 3.0 | Data manipulation |
| NumPy | 2.4 | Numerical computing |
| matplotlib | 3.10 | Static visualizations |
| seaborn | 0.13 | Statistical plots |
| scikit-learn | 1.8 | ML pipeline, Logistic Regression, K-Means |
| XGBoost | 3.2 | Gradient boosted classifier |
| SHAP | 0.51 | Model interpretability |
| Plotly | 6.7 | Interactive dashboard charts |
| Folium | 0.20 | Interactive choropleth maps |
| Streamlit | 1.57 | Dashboard framework |
| SciPy | 1.17 | Statistical tests |

## Dataset

[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 9 CSV files containing orders, items, payments, reviews, customers, sellers, products, geolocation, and category translations from a Brazilian marketplace (2016–2018).
