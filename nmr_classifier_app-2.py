"""
NMR Relaxation Time Classifier — Streamlit App
Features : T1_Relaxation_Time(s), T2_Relaxation_Time(s), Correlation_Time(ns)
Target   : Class
"""

import io, time, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier,
                               BaggingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                            QuadraticDiscriminantAnalysis)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#DB2777", "#0891B2", "#65A30D", "#EA580C", "#4338CA",
]

CSCALE_DIV = "RdBu_r"
CSCALE_HOT = [[0, "#EFF6FF"], [0.5, "#3B82F6"], [1, "#1E3A8A"]]

# Typography
FS_TITLE  = 22
FS_AXIS   = 17
FS_TICK   = 14
FS_LEGEND = 15
FS_ANNOT  = 14

# Layout
MARGIN      = dict(l=80, r=40, t=95, b=80)
HEIGHT_STD  = 530
HEIGHT_TALL = 680
HEIGHT_3D   = 700

PAPER_CLR = "white"
PLOT_CLR  = "#F8FAFC"
GRID_CLR  = "rgba(200,210,220,0.45)"

# PNG export scale — 96 dpi × 3.15 ≈ 302 dpi
EXPORT_SCALE = 3

REQUIRED_COLS = ["T1_Relaxation_Time(s)", "T2_Relaxation_Time(s)",
                 "Correlation_Time(ns)", "Class"]
FEATURE_COLS  = REQUIRED_COLS[:3]
FEAT_LABELS   = ["T₁ Relaxation Time (s)", "T₂ Relaxation Time (s)",
                 "Correlation Time (ns)"]
TARGET_COL    = "Class"


# ── Custom Plotly template ────────────────────────────────────────────────────
def _register_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=PAPER_CLR,
        plot_bgcolor=PLOT_CLR,
        font=dict(family="Inter, Arial, sans-serif", size=FS_TICK, color="#1E293B"),
        title=dict(font=dict(size=FS_TITLE, color="#0F172A"),
                   x=0.5, xanchor="center", pad=dict(b=14)),
        legend=dict(font=dict(size=FS_LEGEND),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#CBD5E1", borderwidth=1.5,
                    title_font_size=FS_LEGEND),
        xaxis=dict(title_font=dict(size=FS_AXIS, color="#334155"),
                   tickfont=dict(size=FS_TICK),
                   gridcolor=GRID_CLR, showgrid=True,
                   linecolor="#94A3B8", linewidth=1.5,
                   ticks="outside", ticklen=6, mirror=False),
        yaxis=dict(title_font=dict(size=FS_AXIS, color="#334155"),
                   tickfont=dict(size=FS_TICK),
                   gridcolor=GRID_CLR, showgrid=True,
                   linecolor="#94A3B8", linewidth=1.5,
                   ticks="outside", ticklen=6, mirror=False),
        margin=MARGIN,
        hoverlabel=dict(font_size=FS_TICK + 1, bgcolor="white",
                        bordercolor="#94A3B8"),
    )
    pio.templates["nmr"] = t
    pio.templates.default = "nmr"

_register_template()


def _finish(fig, title="", height=HEIGHT_STD, margin=None):
    """Uniform finishing pass for every figure."""
    fig.update_layout(title_text=title, height=height,
                      margin=margin or MARGIN)
    for ax in ("xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"):
        if hasattr(fig.layout, ax):
            getattr(fig.layout, ax).update(
                title_font_size=FS_AXIS, tickfont_size=FS_TICK)
    return fig


def _dl_button(fig, label, filename):
    """High-DPI PNG download button (requires kaleido)."""
    try:
        img = fig.to_image(format="png", scale=EXPORT_SCALE,
                           width=1200, height=fig.layout.height or HEIGHT_STD)
        st.download_button(f"⬇️ {label}", img,
                           file_name=filename, mime="image/png")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  DATA & MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_data(file):
    df = pd.read_csv(file)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    return df[REQUIRED_COLS].dropna()


def build_models():
    base = [
        ("Logistic Regression",  LogisticRegression(max_iter=1000)),
        ("K-Nearest Neighbours", KNeighborsClassifier()),
        ("SVM (RBF)",            SVC(probability=True)),
        ("Decision Tree",        DecisionTreeClassifier(random_state=42)),
        ("Random Forest",        RandomForestClassifier(n_estimators=200, random_state=42)),
        ("Extra Trees",          ExtraTreesClassifier(n_estimators=200, random_state=42)),
        ("Gradient Boosting",    GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ("AdaBoost",             AdaBoostClassifier(n_estimators=200, random_state=42)),
        ("Bagging",              BaggingClassifier(n_estimators=100, random_state=42)),
        ("Gaussian Naïve Bayes", GaussianNB()),
        ("LDA",                  LinearDiscriminantAnalysis()),
        ("QDA",                  QuadraticDiscriminantAnalysis()),
    ]
    if HAS_XGB:
        base.append(("XGBoost",
                      XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                    n_estimators=200, random_state=42, verbosity=0)))
    if HAS_LGB:
        base.append(("LightGBM",
                      LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1)))

    return {n: Pipeline([("scaler", StandardScaler()), ("clf", c)])
            for n, c in base}


def multiclass_roc_auc(y_true, y_prob, classes):
    try:
        if len(classes) == 2:
            return roc_auc_score(y_true, y_prob[:, 1])
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Data Overview ─────────────────────────────────────────────────────────
def page_data_overview(df):
    st.header("📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip(
        [c1, c2, c3, c4],
        ["Total Rows", "Features", "Classes", "Missing Values"],
        [len(df), len(FEATURE_COLS), df[TARGET_COL].nunique(),
         int(df.isnull().sum().sum())],
    ):
        col.metric(lbl, f"{val:,}")

    st.subheader("Sample rows")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Descriptive statistics")
    st.dataframe(df.describe().T.style.format("{:.5f}"), width="stretch")

    # Class distribution bar
    vc = df[TARGET_COL].value_counts().reset_index()
    vc.columns = ["Class", "Count"]
    vc["Pct"] = (vc["Count"] / vc["Count"].sum() * 100).round(1)

    fig = px.bar(vc, x="Class", y="Count", color="Class",
                 color_discrete_sequence=PALETTE, text="Count",
                 custom_data=["Pct"])
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        textfont=dict(size=FS_ANNOT + 2, color="#0F172A"),
        marker_line_width=1.8, marker_line_color="white",
        hovertemplate=(
            "<b>%{x}</b><br>Count: %{y:,}<br>"
            "Share: %{customdata[0]:.1f}%<extra></extra>"))
    fig.update_layout(showlegend=False,
                      xaxis_title="Enzyme State",
                      yaxis_title="Sample Count")
    _finish(fig, "Class Distribution", HEIGHT_STD)
    st.plotly_chart(fig, width="stretch")
    _dl_button(fig, "Download chart (300 DPI)", "class_distribution.png")


# ── 2. EDA ───────────────────────────────────────────────────────────────────
def page_eda(df):
    st.header("🔍 Exploratory Data Analysis")

    # Violin plots
    st.subheader("Feature distributions by class")
    for feat, label in zip(FEATURE_COLS, FEAT_LABELS):
        fig = px.violin(df, x=TARGET_COL, y=feat, color=TARGET_COL,
                        box=True, points="outliers",
                        color_discrete_sequence=PALETTE,
                        labels={feat: label, TARGET_COL: "Enzyme State"})
        fig.update_traces(meanline_visible=True, jitter=0.05,
                          marker=dict(size=4, opacity=0.5),
                          box_line_width=2.2)
        fig.update_layout(
            xaxis_title="Enzyme State",
            yaxis_title=label,
            legend=dict(
                title_text="Enzyme State",
                title_font_size=FS_LEGEND,
                font_size=FS_LEGEND,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#CBD5E1",
                borderwidth=1.5,
            ),
        )
        _finish(fig, f"Distribution of {label}", HEIGHT_STD)
        st.plotly_chart(fig, width="stretch")
        _dl_button(fig, f"Download {label[:6]}… (300 DPI)", f"violin_{feat[:4]}.png")

    st.markdown("---")

    # 3-D scatter
    st.subheader("3-D Feature Space")
    fig3d = px.scatter_3d(
        df, x=FEATURE_COLS[0], y=FEATURE_COLS[1], z=FEATURE_COLS[2],
        color=TARGET_COL, color_discrete_sequence=PALETTE, opacity=0.75,
        labels={c: l for c, l in zip(FEATURE_COLS, FEAT_LABELS)})
    fig3d.update_traces(
        marker=dict(size=4, line=dict(width=0.6, color="white")))
    fig3d.update_layout(
        height=HEIGHT_3D,
        margin=dict(l=0, r=0, t=90, b=0),
        title_text="3-D Feature Space — All Classes",
        title_x=0.5, title_font_size=FS_TITLE,
        legend=dict(font_size=FS_LEGEND, title_text="Enzyme State",
                    title_font_size=FS_LEGEND),
        scene=dict(
            bgcolor="#F1F5F9",
            xaxis=dict(title_text=FEAT_LABELS[0],
                       title_font=dict(size=FS_AXIS - 2),
                       tickfont=dict(size=FS_TICK - 1)),
            yaxis=dict(title_text=FEAT_LABELS[1],
                       title_font=dict(size=FS_AXIS - 2),
                       tickfont=dict(size=FS_TICK - 1)),
            zaxis=dict(title_text=FEAT_LABELS[2],
                       title_font=dict(size=FS_AXIS - 2),
                       tickfont=dict(size=FS_TICK - 1))))
    st.plotly_chart(fig3d, width="stretch")

    st.markdown("---")

    # Pair-plot scatter matrix
    st.subheader("Pair-plot scatter matrix")
    fig_pm = px.scatter_matrix(
        df, dimensions=FEATURE_COLS, color=TARGET_COL,
        color_discrete_sequence=PALETTE, opacity=0.55,
        labels={c: l for c, l in zip(FEATURE_COLS, FEAT_LABELS)})
    fig_pm.update_traces(
        diagonal_visible=False,
        marker=dict(size=3.5, line=dict(width=0.4, color="white")),
        showupperhalf=True)
    # Enlarge axis labels inside the matrix
    fig_pm.update_layout(
        height=HEIGHT_TALL,
        font_size=FS_TICK,
        legend=dict(font_size=FS_LEGEND, title_text="Enzyme State",
                    title_font_size=FS_LEGEND))
    for ax in fig_pm.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            getattr(fig_pm.layout, ax).update(
                title_font_size=FS_AXIS - 2, tickfont_size=FS_TICK - 1)
    _finish(fig_pm, "Pairwise Feature Scatter Matrix", HEIGHT_TALL)
    st.plotly_chart(fig_pm, width="stretch")
    _dl_button(fig_pm, "Download pair plot (300 DPI)", "pairplot.png")

    st.markdown("---")

    # Correlation heatmap
    st.subheader("Feature correlation heatmap")
    corr = df[FEATURE_COLS].corr()
    fig_corr = px.imshow(corr, text_auto=".3f",
                         color_continuous_scale=CSCALE_DIV,
                         zmin=-1, zmax=1,
                         x=FEAT_LABELS, y=FEAT_LABELS,
                         aspect="auto")
    fig_corr.update_traces(
        textfont=dict(size=FS_ANNOT + 3, color="#0F172A"),
        hoverongaps=False)
    fig_corr.update_coloraxes(
        colorbar=dict(title_text="Pearson r",
                      title_font_size=FS_AXIS,
                      tickfont_size=FS_TICK,
                      len=0.85))
    _finish(fig_corr, "Feature Correlation Matrix (Pearson)", 460,
            margin=dict(l=210, r=80, t=90, b=150))
    st.plotly_chart(fig_corr, width="stretch")
    _dl_button(fig_corr, "Download heatmap (300 DPI)", "correlation_heatmap.png")

    st.markdown("---")

    # Grouped box plots
    st.subheader("Box plots — all features")
    fig_box = make_subplots(rows=1, cols=3, subplot_titles=FEAT_LABELS,
                             horizontal_spacing=0.10)
    shown, cls_list = set(), df[TARGET_COL].unique()
    for i, feat in enumerate(FEATURE_COLS, 1):
        for cls, clr in zip(cls_list, PALETTE):
            sub  = df[df[TARGET_COL] == cls]
            show = cls not in shown
            fig_box.add_trace(
                go.Box(y=sub[feat], name=cls, marker_color=clr,
                       marker=dict(size=4, opacity=0.55),
                       line_width=2.2, boxmean="sd",
                       showlegend=show, legendgroup=cls),
                row=1, col=i)
            shown.add(cls)
        fig_box.update_yaxes(title_text=FEAT_LABELS[i - 1],
                              title_font_size=FS_AXIS,
                              tickfont_size=FS_TICK,
                              gridcolor=GRID_CLR,
                              linecolor="#94A3B8", row=1, col=i)
        fig_box.update_xaxes(tickfont_size=FS_TICK, row=1, col=i)

    for ann in fig_box.layout.annotations:
        ann.font.size = FS_AXIS

    fig_box.update_layout(
        boxmode="group", height=HEIGHT_STD,
        paper_bgcolor=PAPER_CLR, plot_bgcolor=PLOT_CLR,
        title_text="Feature Box Plots by Enzyme State",
        title_x=0.5, title_font_size=FS_TITLE,
        legend=dict(font_size=FS_LEGEND, title_text="Enzyme State",
                    title_font_size=FS_LEGEND),
        margin=dict(l=80, r=40, t=100, b=60))
    st.plotly_chart(fig_box, width="stretch")
    _dl_button(fig_box, "Download box plots (300 DPI)", "boxplots.png")


# ── 3. Model Training ────────────────────────────────────────────────────────
def page_train(df):
    st.header("🤖 Model Training & Evaluation")

    with st.sidebar:
        st.subheader("⚙️ Training settings")
        test_size    = st.slider("Test size (%)", 10, 40, 20) / 100
        cv_folds     = st.slider("CV folds", 3, 10, 5)
        random_seed  = st.number_input("Random seed", value=42, step=1)

    X     = df[FEATURE_COLS].values
    y     = df[TARGET_COL].values
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size,
        random_state=int(random_seed), stratify=y_enc)

    st.info(f"Train: **{len(X_train):,}** samples  ·  "
            f"Test: **{len(X_test):,}** samples  ·  "
            f"Classes: {list(classes)}")

    if st.button("🚀  Train all models", type="primary"):
        models  = build_models()
        results = []
        prog    = st.progress(0, text="Starting…")
        status  = st.empty()

        for idx, (name, pipe) in enumerate(models.items()):
            status.markdown(f"⏳ Training **{name}**…")
            t0 = time.time()
            cv_sc = cross_val_score(
                pipe, X_train, y_train,
                cv=StratifiedKFold(cv_folds, shuffle=True,
                                   random_state=int(random_seed)),
                scoring="accuracy", n_jobs=1)
            pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test)
            y_prob  = pipe.predict_proba(X_test)
            elapsed = time.time() - t0

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
            roc  = multiclass_roc_auc(y_test, y_prob, classes)

            results.append({
                "Model":             name,
                "CV Acc (mean±std)": f"{cv_sc.mean():.4f} ± {cv_sc.std():.4f}",
                "Test Acc":          round(acc,  4),
                "Precision":         round(prec, 4),
                "Recall":            round(rec,  4),
                "F1 (macro)":        round(f1,   4),
                "ROC-AUC":           round(roc,  4) if not np.isnan(roc) else "N/A",
                "Time (s)":          round(elapsed, 2),
                "_acc_raw":          acc,
                "_cv_mean":          cv_sc.mean(),
                "_pipe":             pipe,
                "_y_pred":           y_pred,
                "_y_prob":           y_prob,
            })
            prog.progress((idx + 1) / len(models),
                          text=f"Trained {idx+1}/{len(models)}: {name}")

        status.empty(); prog.empty()
        results.sort(key=lambda r: r["_acc_raw"], reverse=True)

        st.session_state.update({
            "results": results, "le": le, "classes": classes,
            "X_test": X_test, "y_test": y_test,
            "X_train": X_train, "y_train": y_train,
            "X": X, "y_enc": y_enc,
            "best": results[0], "cv_folds": cv_folds,
        })
        st.success("✅ All models trained!")

    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    classes = st.session_state["classes"]
    X_test  = st.session_state["X_test"]
    y_test  = st.session_state["y_test"]
    best    = st.session_state["best"]

    # Summary table
    st.subheader("📊 Performance summary")
    DISP = ["Model", "CV Acc (mean±std)", "Test Acc", "Precision",
            "Recall", "F1 (macro)", "ROC-AUC", "Time (s)"]
    df_res = pd.DataFrame([{k: r[k] for k in DISP} for r in results])
    st.dataframe(
        df_res.style
              .highlight_max(subset=["Test Acc", "Precision", "Recall", "F1 (macro)"],
                             color="#BBF7D0")
              .highlight_min(subset=["Time (s)"], color="#FEF9C3"),
        width="stretch")

    # Grouped bar — multiple metrics
    st.subheader("Model metric comparison")
    metric_cols = ["Test Acc", "Precision", "Recall", "F1 (macro)"]
    df_bar = df_res.copy()
    for mc in metric_cols:
        df_bar[mc] = pd.to_numeric(df_bar[mc], errors="coerce")
    df_bar = df_bar.dropna(subset=metric_cols)
    order   = df_bar.sort_values("Test Acc", ascending=False)["Model"].tolist()
    df_long = df_bar.melt(id_vars="Model", value_vars=metric_cols,
                           var_name="Metric", value_name="Score")

    fig_cmp = px.bar(
        df_long, x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=["#2563EB", "#16A34A", "#D97706", "#DC2626"],
        category_orders={"Model": order}, text="Score")
    fig_cmp.update_traces(
        texttemplate="%{text:.3f}", textposition="outside",
        textfont=dict(size=FS_ANNOT - 1),
        marker_line_width=1.3, marker_line_color="white")
    fig_cmp.update_layout(
        xaxis_tickangle=-38,
        xaxis=dict(title_text="Model", title_font_size=FS_AXIS,
                   tickfont_size=FS_TICK),
        yaxis=dict(title_text="Score", title_font_size=FS_AXIS,
                   tickfont_size=FS_TICK, range=[0, 1.15]),
        legend=dict(title_text="Metric", font_size=FS_LEGEND,
                    title_font_size=FS_LEGEND))
    _finish(fig_cmp, "Classification Metrics — All Models", HEIGHT_TALL)
    st.plotly_chart(fig_cmp, width="stretch")
    _dl_button(fig_cmp, "Download comparison (300 DPI)", "model_comparison.png")

    # Best model banner
    st.markdown("---")
    st.subheader("🏆 Best model")
    st.success(
        f"**{best['Model']}** — "
        f"Test Accuracy: **{best['Test Acc']:.4f}**  ·  "
        f"F1 (macro): **{best['F1 (macro)']:.4f}**  ·  "
        f"ROC-AUC: **{best['ROC-AUC']}**")

    # Per-model detail
    st.markdown("---")
    st.subheader("🔬 Per-model detail")
    chosen = st.selectbox("Select model to inspect",
                          [r["Model"] for r in results], key="model_detail")
    rec    = next(r for r in results if r["Model"] == chosen)
    y_pred = rec["_y_pred"]
    y_prob = rec["_y_prob"]

    col1, col2 = st.columns([1.1, 0.9])

    # Confusion matrix
    with col1:
        cm      = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        annot   = [[
            f"<b>{cm[i, j]:,}</b><br>"
            f"<span style='font-size:11px'>({cm_norm[i,j]*100:.1f}%)</span>"
            for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]

        fig_cm = go.Figure(go.Heatmap(
            z=cm_norm, x=list(classes), y=list(classes),
            colorscale=CSCALE_HOT,
            text=annot, texttemplate="%{text}",
            textfont=dict(size=FS_ANNOT),
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Rate: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(title_text="Recall Rate",
                          title_font_size=FS_AXIS,
                          tickfont_size=FS_TICK)))
        fig_cm.update_layout(
            xaxis=dict(title_text="Predicted Label",
                       title_font_size=FS_AXIS, tickfont_size=FS_TICK,
                       tickangle=-20),
            yaxis=dict(title_text="True Label",
                       title_font_size=FS_AXIS, tickfont_size=FS_TICK,
                       autorange="reversed"))
        _finish(fig_cm, f"Confusion Matrix — {chosen}", HEIGHT_STD,
                margin=dict(l=170, r=80, t=90, b=130))
        st.plotly_chart(fig_cm, width="stretch")
        _dl_button(fig_cm, "Download confusion matrix (300 DPI)", "confusion_matrix.png")

    # Classification report + radar
    with col2:
        report = classification_report(y_test, y_pred,
                                       target_names=classes, output_dict=True)
        df_rep = pd.DataFrame(report).T.iloc[:-3]
        st.markdown("**Per-class metrics**")
        st.dataframe(
            df_rep[["precision", "recall", "f1-score", "support"]]
                .style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                               "f1-score": "{:.4f}", "support": "{:.0f}"})
                .bar(subset=["f1-score"], color="#93c5fd"),
            width="stretch")

        # Radar (spider) chart
        cats     = list(classes) + [classes[0]]
        f1_vals  = [report[c]["f1-score"] for c in classes] + \
                   [report[classes[0]]["f1-score"]]
        fig_rad  = go.Figure(go.Scatterpolar(
            r=f1_vals, theta=cats, fill="toself",
            line=dict(color="#2563EB", width=3),
            fillcolor="rgba(37,99,235,0.15)",
            marker=dict(size=9, color="#2563EB",
                        line=dict(width=2, color="white"))))
        fig_rad.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                tickfont_size=FS_TICK,
                                title_font_size=FS_AXIS),
                angularaxis=dict(tickfont_size=FS_TICK + 2)),
            showlegend=False,
            paper_bgcolor=PAPER_CLR)
        _finish(fig_rad, "Per-class F1 — Radar", 380,
                margin=dict(l=60, r=60, t=80, b=60))
        st.plotly_chart(fig_rad, width="stretch")
        _dl_button(fig_rad, "Download radar (300 DPI)", "radar_f1.png")

    # ROC curves
    st.subheader("ROC Curves (One-vs-Rest)")
    fig_roc = go.Figure()
    y_bin   = label_binarize(y_test, classes=np.arange(len(classes)))
    for i, (cls, clr) in enumerate(zip(classes, PALETTE)):
        col_y = y_bin[:, i] if y_bin.ndim > 1 else y_bin
        col_p = y_prob[:, i] if y_prob.ndim > 1 else y_prob
        fpr, tpr, _ = roc_curve(col_y, col_p)
        auc_val = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{cls}  (AUC = {auc_val:.3f})",
            line=dict(color=clr, width=3.5)))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(dash="dot", color="#94A3B8", width=1.5))
    fig_roc.add_annotation(x=0.6, y=0.38, text="Random classifier",
                            font=dict(size=FS_ANNOT - 1, color="#94A3B8"),
                            showarrow=False, textangle=-34)
    fig_roc.update_layout(
        xaxis=dict(title_text="False Positive Rate (1 - Specificity)",
                   title_font_size=FS_AXIS, tickfont_size=FS_TICK),
        yaxis=dict(title_text="True Positive Rate (Sensitivity)",
                   title_font_size=FS_AXIS, tickfont_size=FS_TICK),
        legend=dict(font_size=FS_LEGEND, title_text="Class  (AUC)",
                    title_font_size=FS_LEGEND))
    _finish(fig_roc, f"ROC Curves — {chosen}", HEIGHT_STD)
    st.plotly_chart(fig_roc, width="stretch")
    _dl_button(fig_roc, "Download ROC curves (300 DPI)", "roc_curves.png")


# ── 4. Learning Curves ───────────────────────────────────────────────────────
def page_learning_curves(df):
    st.header("📈 Learning Curves")
    if "results" not in st.session_state:
        st.warning("Train models first (Model Training tab).")
        return

    results  = st.session_state["results"]
    X        = st.session_state["X"]
    y_enc    = st.session_state["y_enc"]
    cv_folds = st.session_state.get("cv_folds", 5)

    chosen = st.selectbox("Select model", [r["Model"] for r in results],
                          key="lc_model")
    rec    = next(r for r in results if r["Model"] == chosen)

    if st.button("Generate learning curve", type="primary"):
        with st.spinner("Computing…"):
            train_sizes, train_sc, val_sc = learning_curve(
                rec["_pipe"], X, y_enc,
                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42),
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring="accuracy", n_jobs=1)

        fig = go.Figure()
        # Shaded error bands
        x_all = np.concatenate([train_sizes, train_sizes[::-1]])
        fig.add_trace(go.Scatter(
            x=x_all,
            y=np.concatenate([train_sc.mean(1) + train_sc.std(1),
                              (train_sc.mean(1) - train_sc.std(1))[::-1]]),
            fill="toself", fillcolor="rgba(37,99,235,0.12)",
            line_color="rgba(255,255,255,0)", showlegend=False,
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x_all,
            y=np.concatenate([val_sc.mean(1) + val_sc.std(1),
                              (val_sc.mean(1) - val_sc.std(1))[::-1]]),
            fill="toself", fillcolor="rgba(220,38,38,0.10)",
            line_color="rgba(255,255,255,0)", showlegend=False,
            hoverinfo="skip"))
        # Main lines
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_sc.mean(1),
            mode="lines+markers", name="Training accuracy",
            line=dict(color="#2563EB", width=3.5),
            marker=dict(size=10, color="#2563EB",
                        line=dict(width=2.5, color="white"))))
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_sc.mean(1),
            mode="lines+markers", name="Validation accuracy",
            line=dict(color="#DC2626", width=3.5),
            marker=dict(size=10, color="#DC2626",
                        line=dict(width=2.5, color="white"))))
        fig.add_hrect(y0=0.95, y1=1.01, fillcolor="rgba(22,163,74,0.07)",
                      line_width=0, annotation_text="≥ 0.95 zone",
                      annotation_font=dict(size=FS_ANNOT, color="#16A34A"),
                      annotation_position="top right")
        fig.update_layout(
            xaxis=dict(title_text="Number of training samples",
                       title_font_size=FS_AXIS, tickfont_size=FS_TICK),
            yaxis=dict(title_text="Accuracy",
                       title_font_size=FS_AXIS, tickfont_size=FS_TICK,
                       range=[0, 1.04]),
            legend=dict(font_size=FS_LEGEND))
        _finish(fig, f"Learning Curve — {chosen}", HEIGHT_STD)
        st.plotly_chart(fig, width="stretch")
        _dl_button(fig, "Download learning curve (300 DPI)", "learning_curve.png")


# ── 5. Feature Importance ────────────────────────────────────────────────────
def page_feature_importance(df):
    st.header("🎯 Feature Importance")
    if "results" not in st.session_state:
        st.warning("Train models first (Model Training tab).")
        return

    results     = st.session_state["results"]
    tree_models = [r for r in results
                   if hasattr(r["_pipe"]["clf"], "feature_importances_")]
    if not tree_models:
        st.info("No tree-based model found.")
        return

    chosen = st.selectbox("Select tree-based model",
                          [r["Model"] for r in tree_models], key="fi_model")
    rec  = next(r for r in tree_models if r["Model"] == chosen)
    imps = rec["_pipe"]["clf"].feature_importances_

    df_fi = pd.DataFrame({"Feature": FEAT_LABELS, "Importance": imps})
    df_fi = df_fi.sort_values("Importance", ascending=True)

    fig = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="teal",
                 text="Importance")
    fig.update_traces(
        texttemplate="%{text:.4f}", textposition="outside",
        textfont=dict(size=FS_ANNOT + 2),
        marker_line_width=1.8, marker_line_color="white")
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(title_text="Gini Importance",
                   title_font_size=FS_AXIS, tickfont_size=FS_TICK,
                   range=[0, df_fi["Importance"].max() * 1.22]),
        yaxis=dict(title_text="", tickfont_size=FS_TICK + 1))
    _finish(fig, f"Feature Importances — {chosen}", 420,
            margin=dict(l=220, r=70, t=90, b=60))
    st.plotly_chart(fig, width="stretch")
    _dl_button(fig, "Download importance chart (300 DPI)", "feature_importance.png")

    # Cross-model comparison
    if len(tree_models) > 1:
        st.subheader("Cross-model importance comparison")
        rows = []
        for r in tree_models:
            for feat, label, imp in zip(FEATURE_COLS, FEAT_LABELS,
                                         r["_pipe"]["clf"].feature_importances_):
                rows.append({"Model": r["Model"], "Feature": label,
                              "Importance": round(imp, 5)})
        df_cmp = pd.DataFrame(rows)
        fig2 = px.bar(df_cmp, x="Feature", y="Importance", color="Model",
                      barmode="group", color_discrete_sequence=PALETTE,
                      text="Importance")
        fig2.update_traces(
            texttemplate="%{text:.3f}", textposition="outside",
            textfont=dict(size=FS_ANNOT - 1),
            marker_line_width=1.3, marker_line_color="white")
        fig2.update_layout(
            xaxis=dict(title_text="Feature", title_font_size=FS_AXIS,
                       tickfont_size=FS_TICK),
            yaxis=dict(title_text="Importance", title_font_size=FS_AXIS,
                       tickfont_size=FS_TICK),
            legend=dict(font_size=FS_LEGEND, title_text="Model",
                        title_font_size=FS_LEGEND))
        _finish(fig2, "Feature Importance Across Tree Models", HEIGHT_STD)
        st.plotly_chart(fig2, width="stretch")
        _dl_button(fig2, "Download multi-model comparison (300 DPI)",
                   "feature_importance_all.png")


# ── 6. Save Model ────────────────────────────────────────────────────────────
def page_save_model():
    st.header("💾 Save Best Model")
    if "best" not in st.session_state:
        st.warning("Train models first (Model Training tab).")
        return

    best    = st.session_state["best"]
    results = st.session_state["results"]
    le      = st.session_state["le"]
    classes = st.session_state["classes"]

    st.success(
        f"Best model: **{best['Model']}** — "
        f"Test Accuracy {best['Test Acc']:.4f}  ·  "
        f"F1 {best['F1 (macro)']:.4f}")

    chosen_name = st.selectbox("Choose model to export",
                               [r["Model"] for r in results])
    rec  = next(r for r in results if r["Model"] == chosen_name)
    pipe = rec["_pipe"]

    st.markdown("---")
    st.subheader("Export as `.pt`  (PyTorch wrapper — deployment-ready)")
    if not HAS_TORCH:
        st.error("PyTorch not installed. Run: `pip install torch`")
    else:
        if st.button("⬇️ Export as .pt", type="primary"):
            pipe_bytes = pickle.dumps(
                {"pipe": pipe, "label_encoder": le,
                 "classes": list(classes), "model_name": chosen_name,
                 "features": FEATURE_COLS})

            class SklearnWrapper(nn.Module):
                def __init__(self, pb):
                    super().__init__()
                    self.register_buffer(
                        "pipe_bytes",
                        torch.tensor(list(pb), dtype=torch.uint8))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    obj = pickle.loads(bytes(self.pipe_bytes.tolist()))
                    return torch.tensor(obj["pipe"].predict(x.numpy()),
                                        dtype=torch.long)

            wrapper = SklearnWrapper(pipe_bytes)
            payload = {
                "state_dict":    wrapper.state_dict(),
                "pipe_bytes":    pipe_bytes,
                "model_name":    chosen_name,
                "label_encoder": le,
                "classes":       list(classes),
                "features":      FEATURE_COLS,
                "test_accuracy": rec["Test Acc"],
                "f1_macro":      rec["F1 (macro)"],
            }
            buf = io.BytesIO()
            torch.save(payload, buf); buf.seek(0)
            fname = (chosen_name.replace(" ", "_")
                                .replace("(", "").replace(")", "") + ".pt")
            st.download_button(f"📥 Download {fname}", buf,
                               file_name=fname, mime="application/octet-stream")
            st.code(f"""
import torch, pickle, numpy as np

payload = torch.load("{fname}", map_location="cpu")
pipe    = pickle.loads(payload["pipe_bytes"])["pipe"]
le      = payload["label_encoder"]

# Predict — shape [n_samples, 3]
X_new   = np.array([[0.95, 0.048, 12.5]])
y_label = le.inverse_transform(pipe.predict(X_new))
print(y_label)   # e.g. ['Free_Enzyme']
""", language="python")

    st.markdown("---")
    st.subheader("Export as `.pkl`  (scikit-learn — universal)")
    if st.button("⬇️ Export as .pkl"):
        buf = io.BytesIO()
        pickle.dump({"pipe": pipe, "label_encoder": le,
                     "classes": list(classes), "features": FEATURE_COLS}, buf)
        buf.seek(0)
        fname_pkl = chosen_name.replace(" ", "_") + ".pkl"
        st.download_button(f"📥 Download {fname_pkl}", buf,
                           file_name=fname_pkl, mime="application/octet-stream")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="NMR Relaxation Classifier",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global style overrides
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', Arial, sans-serif; }
      /* Tabs */
      .stTabs [data-baseweb="tab"] {
          font-size: 1.05rem; font-weight: 600; padding: 10px 22px;
          letter-spacing: 0.01em;
      }
      /* Sidebar title */
      section[data-testid="stSidebar"] h1 {
          font-size: 1.45rem; font-weight: 700;
      }
      /* Metric */
      [data-testid="stMetricLabel"]  { font-size: 1.0rem  !important; }
      [data-testid="stMetricValue"]  { font-size: 2.1rem  !important; font-weight: 700; }
      /* Section headers */
      h2 { font-size: 1.4rem !important; font-weight: 700; }
      h3 { font-size: 1.2rem !important; font-weight: 600; }
      /* Download buttons */
      [data-testid="stDownloadButton"] > button {
          font-size: 0.95rem; border-radius: 8px; padding: 6px 16px;
      }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("🧪 NMR Classifier for DNA-bounded Proteins")
        st.markdown("---")
        st.subheader("1 · Load data")
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV with columns:\n"
                    "- `T1_Relaxation_Time(s)`\n"
                    "- `T2_Relaxation_Time(s)`\n"
                    "- `Correlation_Time(ns)`\n"
                    "- `Class`")
            st.stop()
        df = load_data(uploaded)
        st.success(f"Loaded **{len(df):,}** rows · "
                   f"{df[TARGET_COL].nunique()} classes")
        st.markdown("---")
        st.markdown("**Classes found**")
        for cls in sorted(df[TARGET_COL].unique()):
            n = (df[TARGET_COL] == cls).sum()
            st.markdown(f"• `{cls}` — {n:,} samples")
        st.markdown("---")
        st.caption("📸 Each chart has a '⬇️ Download (300 DPI)' button "
                   "below it (requires kaleido: `pip install kaleido`).")

    # Tabs
    tabs = st.tabs([
        "📋 Data Overview",
        "🔍 EDA",
        "🤖 Model Training",
        "📈 Learning Curves",
        "🎯 Feature Importance",
        "💾 Save Model",
    ])

    with tabs[0]: page_data_overview(df)
    with tabs[1]: page_eda(df)
    with tabs[2]: page_train(df)
    with tabs[3]: page_learning_curves(df)
    with tabs[4]: page_feature_importance(df)
    with tabs[5]: page_save_model()


if __name__ == "__main__":
    main()
