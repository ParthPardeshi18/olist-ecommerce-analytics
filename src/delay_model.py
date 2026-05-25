"""
Phase 6 — Late Delivery Prediction Model
=========================================
Builds a binary classifier predicting is_late (1 = delivered after estimate).

Feature engineering:
  - seller_customer_same_state, product_volume_cm3, is_weekend_order
  - seller_avg_delay (expanding window to prevent leakage)
  - category_late_rate (expanding window)
  - freight_to_value_ratio

Models:
  - Baseline: Logistic Regression
  - Main: XGBoost with scale_pos_weight

Produces:
  14. SHAP feature importance (top 15)
  15. ROC curve comparison
  16. Confusion matrix at optimal threshold
  - Saves model + scaler to /outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
import shap
import joblib
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
    """Load master dataframe with date parsing."""
    df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "master_df.csv"),
        parse_dates=["order_purchase_timestamp"],
        low_memory=False,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """
    Build prediction features from the master dataframe.
    Uses expanding-window aggregates to prevent data leakage
    for seller_avg_delay and category_late_rate.
    """
    # Filter to delivered orders with valid delivery data
    model_df = df[
        (df["order_status"] == "delivered") &
        (df["is_late"].notna()) &
        (df["delivery_days"].notna())
    ].copy()

    # Drop duplicates to order level for model training
    model_df = model_df.drop_duplicates("order_id").reset_index(drop=True)

    print(f"  Modelling dataset: {len(model_df):,} delivered orders")
    print(f"  Late rate: {model_df['is_late'].mean()*100:.1f}%")

    # Feature: same state
    model_df["seller_customer_same_state"] = (
        model_df["seller_state"] == model_df["customer_state"]
    ).astype(int)

    # Feature: product volume
    model_df["product_volume_cm3"] = (
        model_df["product_length_cm"].fillna(0) *
        model_df["product_height_cm"].fillna(0) *
        model_df["product_width_cm"].fillna(0)
    )

    # Feature: weekend order
    model_df["is_weekend_order"] = model_df["order_dayofweek"].isin([5, 6]).astype(int)

    # Feature: freight to value ratio
    model_df["freight_to_value_ratio"] = np.where(
        model_df["payment_value"] > 0,
        model_df["freight_value"] / model_df["payment_value"],
        0,
    )

    # Feature: seller historical avg delay (expanding window — sort by time first)
    model_df = model_df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    # For seller_avg_delay, compute cumulative mean excluding current row
    # to avoid data leakage. Use transform to keep alignment with index.
    model_df["seller_avg_delay"] = (
        model_df.groupby("seller_id")["delay_days"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    model_df["seller_avg_delay"] = model_df["seller_avg_delay"].fillna(0)

    # Feature: category historical late rate (expanding window)
    model_df["product_category_name_english"] = (
        model_df["product_category_name_english"].fillna("unknown")
    )
    model_df["category_late_rate"] = (
        model_df.groupby("product_category_name_english")["is_late"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    model_df["category_late_rate"] = model_df["category_late_rate"].fillna(
        model_df["is_late"].mean()
    )

    # Feature: estimated_days (the delivery promise itself is very predictive)
    # Feature: product_weight_g
    # Feature: freight_value
    # Feature: price

    feature_cols = [
        "estimated_days",
        "product_weight_g",
        "product_volume_cm3",
        "freight_value",
        "price",
        "payment_value",
        "seller_customer_same_state",
        "is_weekend_order",
        "freight_to_value_ratio",
        "seller_avg_delay",
        "category_late_rate",
        "payment_installments",
        "product_photos_qty",
    ]

    # Fill remaining nulls with median
    for col in feature_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        if model_df[col].isnull().sum() > 0:
            model_df[col] = model_df[col].fillna(model_df[col].median())

    target = "is_late"
    model_df[target] = model_df[target].astype(int)

    X = model_df[feature_cols].values
    y = model_df[target].values

    print(f"\n  Features ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"    {col}")
    print(f"  Target: is_late (positive rate: {y.mean()*100:.1f}%)")

    return X, y, feature_cols, model_df


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_models(X, y, feature_cols):
    """
    Train Logistic Regression (baseline) and XGBoost (main).
    5-fold stratified CV for evaluation. Tune XGBoost hyperparameters.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ─── Logistic Regression ──────────────────────────────────────────────
    print("\n  Training Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_probs = cross_val_predict(lr, X_scaled, y, cv=skf, method="predict_proba")[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)

    lr_auc = roc_auc_score(y, lr_probs)
    lr_f1 = f1_score(y, lr_preds)
    lr_prec = precision_score(y, lr_preds)
    lr_rec = recall_score(y, lr_preds)

    print(f"    AUC: {lr_auc:.4f}")
    print(f"    F1:  {lr_f1:.4f}")
    print(f"    Precision: {lr_prec:.4f} | Recall: {lr_rec:.4f}")

    # ─── XGBoost Tuning ──────────────────────────────────────────────────
    print("\n  Tuning XGBoost (3 configurations)...")
    pos_weight = (y == 0).sum() / (y == 1).sum()

    configs = [
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 200},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
        {"max_depth": 5, "learning_rate": 0.08, "n_estimators": 250},
    ]

    best_auc = 0
    best_config = None
    best_xgb_probs = None

    for i, cfg in enumerate(configs):
        xgb = XGBClassifier(
            **cfg,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        probs = cross_val_predict(xgb, X, y, cv=skf, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, probs)
        print(f"    Config {i+1} (depth={cfg['max_depth']}, lr={cfg['learning_rate']}, "
              f"n={cfg['n_estimators']}): AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_config = cfg
            best_xgb_probs = probs

    print(f"    Best config: depth={best_config['max_depth']}, "
          f"lr={best_config['learning_rate']}, n={best_config['n_estimators']}")

    # Train final XGBoost on full data
    xgb_final = XGBClassifier(
        **best_config,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    xgb_final.fit(X, y)

    xgb_preds = (best_xgb_probs >= 0.5).astype(int)
    xgb_f1 = f1_score(y, xgb_preds)
    xgb_prec = precision_score(y, xgb_preds)
    xgb_rec = recall_score(y, xgb_preds)

    print(f"\n  XGBoost (best) CV results:")
    print(f"    AUC: {best_auc:.4f}")
    print(f"    F1:  {xgb_f1:.4f}")
    print(f"    Precision: {xgb_prec:.4f} | Recall: {xgb_rec:.4f}")

    # Fit logistic regression on full data too (for saving)
    lr.fit(X_scaled, y)

    # Save model and scaler
    joblib.dump(xgb_final, os.path.join(OUTPUT_DIR, "delay_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print(f"\n  Saved: delay_model.pkl, scaler.pkl")

    return lr, xgb_final, scaler, lr_probs, best_xgb_probs, y


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 14 — SHAP Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def chart_shap_importance(xgb_model, X, feature_cols):
    """
    SHAP values for XGBoost showing direction of impact per feature.
    Top 15 features as horizontal bar chart.
    """
    print("\n  Computing SHAP values (this may take a moment)...")

    # Use a sample for SHAP to keep computation time reasonable
    sample_size = min(5000, len(X))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), sample_size, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP for ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": mean_abs_shap,
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("importance", ascending=True)

    top15 = feature_importance.tail(15)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [RED if v > 0 else TEAL for v in top15["mean_shap"]]
    ax.barh(top15["feature"], top15["importance"], color=colors, alpha=0.85, zorder=3)

    ax.set_title("SHAP Feature Importance — Late Delivery Predictor",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Mean |SHAP value| (impact on prediction)")
    ax.grid(axis="x", alpha=0.2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=RED, markersize=10,
               label="Increases late probability"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TEAL, markersize=10,
               label="Decreases late probability"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart14_shap_importance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    top_feature = feature_importance.iloc[-1]
    print(f"\n  BUSINESS INSIGHT — Prediction Drivers:")
    print(f"    The strongest predictor of late delivery is '{top_feature['feature']}'")
    print(f"    (mean |SHAP| = {top_feature['importance']:.4f})")
    print(f"    Top 3 features by impact:")
    for _, row in feature_importance.tail(3).iloc[::-1].iterrows():
        direction = "increases" if row["mean_shap"] > 0 else "decreases"
        print(f"      {row['feature']}: {direction} late probability")

    return shap_values, X_sample


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 15 — ROC Curve Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def chart_roc_comparison(y, lr_probs, xgb_probs):
    """
    Plot ROC curves for both models on the same axes.
    Annotate AUC for each. Identify optimal threshold for XGBoost.
    """
    lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
    xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y, xgb_probs)

    lr_auc = roc_auc_score(y, lr_probs)
    xgb_auc = roc_auc_score(y, xgb_probs)

    # Optimal threshold: maximize Youden's J statistic (TPR - FPR)
    j_scores = xgb_tpr - xgb_fpr
    best_idx = np.argmax(j_scores)
    best_threshold = xgb_thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(lr_fpr, lr_tpr, color=AMBER, linewidth=2,
            label=f"Logistic Regression (AUC = {lr_auc:.4f})")
    ax.plot(xgb_fpr, xgb_tpr, color=BLUE, linewidth=2.5,
            label=f"XGBoost (AUC = {xgb_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=LIGHT_TEXT, linewidth=1, linestyle="--",
            alpha=0.5, label="Random baseline")

    # Mark optimal threshold
    ax.scatter(xgb_fpr[best_idx], xgb_tpr[best_idx], color=RED, s=100, zorder=5,
               edgecolors="white", linewidths=2)
    ax.annotate(
        f"Optimal threshold: {best_threshold:.3f}\nTPR={xgb_tpr[best_idx]:.2f}, "
        f"FPR={xgb_fpr[best_idx]:.2f}",
        xy=(xgb_fpr[best_idx], xgb_tpr[best_idx]),
        xytext=(xgb_fpr[best_idx] + 0.15, xgb_tpr[best_idx] - 0.15),
        fontsize=10, fontweight="bold", color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=RED, alpha=0.8),
    )

    ax.set_title("ROC Curve — Model Comparison", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", framealpha=0.3, fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart15_roc_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    print(f"\n  Model Comparison:")
    print(f"    Logistic Regression AUC: {lr_auc:.4f}")
    print(f"    XGBoost AUC: {xgb_auc:.4f} ({(xgb_auc/lr_auc-1)*100:.1f}% improvement)")
    print(f"    Optimal threshold (Youden's J): {best_threshold:.3f}")

    return best_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 16 — Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def chart_confusion_matrix(y, xgb_probs, threshold):
    """
    Confusion matrix at the optimal threshold, translated to business terms.
    """
    preds = (xgb_probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds)

    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(9, 7))

    # Custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom", [DARK_BG, BLUE])

    im = ax.imshow(cm, interpolation="nearest", cmap=custom_cmap)

    labels = [
        [f"True Negative\n{tn:,}\n({tn/(tn+fp)*100:.1f}% of actual on-time)",
         f"False Positive\n{fp:,}\n({fp/(tn+fp)*100:.1f}% false alarms)"],
        [f"False Negative\n{fn:,}\n({fn/(fn+tp)*100:.1f}% missed late orders)",
         f"True Positive\n{tp:,}\n({tp/(fn+tp)*100:.1f}% caught late orders)"],
    ]

    for i in range(2):
        for j in range(2):
            color = LIGHT_TEXT if cm[i, j] < cm.max() * 0.7 else DARK_BG
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nOn-Time", "Predicted\nLate"])
    ax.set_yticklabels(["Actual\nOn-Time", "Actual\nLate"])
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.3f})",
                 fontsize=16, fontweight="bold", pad=15)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart16_confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Business translation
    monthly_orders = len(y) / 20  # ~20 months of data
    monthly_flagged = (fp + tp) / 20
    monthly_true_late = tp / 20
    precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\n  BUSINESS INSIGHT — Model in Production:")
    print(f"    At threshold {threshold:.3f}:")
    print(f"      Flagged as at-risk: {tp+fp:,} orders ({(tp+fp)/len(y)*100:.1f}% of all)")
    print(f"      True late caught: {tp:,} of {tp+fn:,} ({tp/(tp+fn)*100:.1f}% recall)")
    print(f"      False alarms: {fp:,} ({fp/(tp+fp)*100:.1f}% of flagged)")
    print(f"    Monthly projection (~{monthly_orders:.0f} orders/month):")
    print(f"      ~{monthly_flagged:.0f} orders flagged as at-risk per month")
    print(f"      ~{monthly_true_late:.0f} are genuinely late ({precision_at_threshold*100:.1f}% precision)")
    print(f"    Recommendation: Use this model to proactively notify customers")
    print(f"    of potential delays and prioritize logistics routing for high-risk orders.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  PHASE 6 — LATE DELIVERY PREDICTION MODEL")
    print("█" * 70)

    set_chart_style()
    df = load_master()
    print(f"\n  Master dataframe loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    X, y, feature_cols, model_df = engineer_features(df)
    lr, xgb_final, scaler, lr_probs, xgb_probs, y = train_models(X, y, feature_cols)
    chart_shap_importance(xgb_final, X, feature_cols)
    threshold = chart_roc_comparison(y, lr_probs, xgb_probs)
    chart_confusion_matrix(y, xgb_probs, threshold)

    print("\n" + "=" * 70)
    print("✓ PHASE 6 COMPLETE — Model and charts saved to /outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
