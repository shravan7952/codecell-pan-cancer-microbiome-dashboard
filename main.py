# app.py
"""
Pan-Cancer Microbiome & Immune Dashboard — All-in-one Streamlit app
Place this file beside merged_data_counts.csv (and optional codecell_logo.png).
Run: streamlit run app.py
"""

import os
import re
import math
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

warnings.filterwarnings("ignore")

# optional libs
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# -------------------------
# Page config & branding
# -------------------------
st.set_page_config(page_title="Codecell.ai — Pan-Cancer Dashboard", layout="wide", initial_sidebar_state="expanded")

if os.path.exists("codecell_logo.png"):
    st.image("codecell_logo.png", width=220)
else:
    st.markdown("### **Codecell.ai**")

st.title("Pan-Cancer Microbiome & Immune Dashboard")
st.markdown("Explore microbiome abundances, immune features, clinical survival and ML in one place. Use the sidebar to control filters and visuals.")

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data(path="merged_data_counts.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path, index_col=0, low_memory=False)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# -------------------------
# Utilities
# -------------------------
def clean_name(col: str) -> str:
    if not isinstance(col, str):
        col = str(col)
    col = re.sub(r"(_score\b|_data\b|\bPCA\b|\bModule\b|\bModule\d+\b)", "", col, flags=re.IGNORECASE)
    col = re.sub(r"\s*\(\d+\)$", "", col)  # trailing (123)
    col = re.sub(r"\s*\d{5,}\b", "", col)  # trailing long ids
    col = col.replace("_", " ").replace(".", " ")
    col = re.sub(r"\s+", " ", col).strip()
    return col

def safe_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("")
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(r"[^\d\.\-eE]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def normalize_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(0, index=series.index)
    return (series - mn) / (mx - mn)

def df_to_csv_bytes(df_in: pd.DataFrame) -> bytes:
    return df_in.to_csv(index=True).encode("utf-8")

def apply_plot_theme(fig: go.Figure, theme_choice="dark", title_size=16):
    font_color = "#E7EEF6"
    paper_bg = "#061225"
    plot_bg = "#041422"
    grid_color = "#123242"

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color, size=14, family="Inter, Arial"),
        title=dict(font=dict(size=title_size, color=font_color)),
        legend=dict(font=dict(color=font_color)),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(color=font_color, gridcolor=grid_color, zerolinecolor=grid_color)
    fig.update_yaxes(color=font_color, gridcolor=grid_color, zerolinecolor=grid_color)
    return fig

# -------------------------
# Detect groups (microbiome / immune)
# -------------------------
meta_cutoff = 113 if df.shape[1] > 120 else min(120, max(10, df.shape[1] // 3))
meta_cols = list(df.columns[:meta_cutoff])
microbiome_cols = [c for c in df.columns if c not in meta_cols]

fungi_keywords = ["candida", "aspergillus", "cryptococcus", "malassezia", "saccharomyces", "penicillium"]
fungi_cols = [c for c in microbiome_cols if any(k in c.lower() for k in fungi_keywords)]
bacteria_cols = [c for c in microbiome_cols if c not in fungi_cols]

# -------------------------
# Detect Immune Groups (fixed)
# -------------------------
def detect_immune_groups(dataframe):
    cols = dataframe.columns.tolist()

    # much broader matching
    scores = [c for c in cols if re.search(r"(immune|score|stromal|microenv|module|pca)", c, re.I)]
    checkpoints = [c for c in cols if re.search(r"(ctla|pd1|pd-1|pdl1|pd-l1|pd l1)", c, re.I)]
    cytokines = [c for c in cols if re.search(r"(^il\d+$|interleukin|ifn|tgf|tnf|cxcl|ccr|chemokine)", c, re.I)]
    celltypes = [c for c in cols if re.search(r"(t.?cell|b.?cell|nk.?cell|macrophage|dendritic|neutrophil|monocyte|mast)", c, re.I)]

    # fallback: if all empty, try any column with "immune" in name
    if not any([scores, checkpoints, cytokines, celltypes]):
        fallback = [c for c in cols if "immune" in c.lower()]
        scores = fallback

    # keep only columns that yield numeric values
    def keep_numeric(lst):
        out = []
        for c in lst:
            if c in dataframe.columns:
                vals = safe_numeric(dataframe[c])
                if vals.notna().sum() > 0:
                    out.append(c)
        return out

    return {
        "Scores": keep_numeric(scores),
        "Checkpoints": keep_numeric(checkpoints),
        "Cytokines": keep_numeric(cytokines),
        "CellTypes": keep_numeric(celltypes),
    }

# -------------------------
# Sidebar (global controls)
# -------------------------
st.sidebar.header("Global controls")
cancers = sorted(df["_primary_disease"].dropna().unique().tolist())
selected_cancer = st.sidebar.selectbox("Filter by cancer type", ["All"] + cancers)
theme_choice = "dark"
normalize = st.sidebar.checkbox("Normalize values (0–1)", value=False)
topN_global = st.sidebar.slider("Top N genera (default lists)", min_value=5, max_value=100, value=20)

# filter dataset
sub = df if selected_cancer == "All" else df[df["_primary_disease"] == selected_cancer]

# -------------------------
# Top metrics
# -------------------------
st.markdown("<style>.big-metric{font-size:16px;font-weight:700}</style>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total samples", f"{df.shape[0]:,}")
m2.metric("Cancer types", f"{df['_primary_disease'].nunique():,}")
m3.metric("Features", f"{df.shape[1]:,}")
if "OS.time" in df.columns:
    os_vals = safe_numeric(df["OS.time"]).dropna()
    m4.metric("Median OS", f"{int(os_vals.median()) if len(os_vals)>0 else 'N/A'}")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Overview", "Bacteria", "Fungi", "Immune",
    "Clinical / Survival", "PCA / UMAP", "Differential (Volcano)",
    "Heatmap Explorer", "ML Playground", "Outlier Detection"
])

# -------------------------
# Overview Tab
# -------------------------
with tabs[0]:
    st.header("Overview")
    counts = sub["_primary_disease"].fillna("Unknown").value_counts()
    if normalize:
        counts = normalize_series(counts)
    sorted_counts = counts.sort_values(ascending=False)
    top10 = sorted_counts.head(10)
    others = sorted_counts.iloc[10:].sum() if sorted_counts.size > 10 else 0
    pie_series = pd.concat([top10, pd.Series({"Other": others})]) if others else top10

    c1, c2 = st.columns(2)
    with c1:
            fig = px.bar(
        counts.sort_values(),
        orientation="h",
        labels={"index": "Cancer", "value": "Samples"},
        color_discrete_sequence=px.colors.sequential.Blues,
        height=600  # taller chart for readability
    )
    apply_plot_theme(fig, theme_choice)
    fig.update_layout(
        title_text="Samples per Cancer (horizontal)",
        margin=dict(l=150, r=40, t=60, b=40),   # bigger left margin for long cancer names
        xaxis_title="Number of Samples",
        yaxis_title="Cancer Type"
    )
    st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(
    names=[clean_name(x) for x in pie_series.index],
    values=pie_series.values,
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set3
)
        # Force labels to be readable
    fig.update_traces(
        textinfo="percent+label",
        textfont=dict(size=14, color="white"),   # bigger & white text
        insidetextorientation="radial"           # rotates labels to fit arcs
    )
    apply_plot_theme(fig, theme_choice)
    fig.update_layout(
        height=450,                              # same height as bar chart
        margin=dict(l=20, r=20, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # stacked relative abundance
    st.subheader("Stacked relative abundance of top genera per cancer")
    stack_topN = st.number_input("Top genera to include in stacked bars", min_value=3, max_value=100, value=10)
    if len(bacteria_cols) > 0:
        mean_by_cancer = df.groupby("_primary_disease")[bacteria_cols].mean().fillna(0)
        top_gen = mean_by_cancer.mean(axis=0).sort_values(ascending=False).head(stack_topN).index.tolist()
        if len(top_gen) > 0:
            rel = mean_by_cancer[top_gen].div(mean_by_cancer[top_gen].sum(axis=1), axis=0).fillna(0) * 100
            rel = rel.reset_index().melt(id_vars="_primary_disease", var_name="genus", value_name="pct")
            fig = px.bar(rel, x="_primary_disease", y="pct", color="genus", title=f"Stacked % abundance — top {stack_topN} genera",
                         color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(xaxis_tickangle=-45)
            apply_plot_theme(fig, theme_choice)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download stacked data CSV", data=df_to_csv_bytes(rel), file_name="stacked_relative_abundance.csv")
        else:
            st.info("No top genera found for stacking.")

# -------------------------
# Bacteria Tab
# -------------------------
with tabs[1]:
    st.header("Bacteria — top genera & correlation")
    if len(bacteria_cols) == 0:
        st.info("No bacterial features detected.")
    else:
        max_bac = len(bacteria_cols)
        n_bac = st.slider("Top N bacterial genera", min_value=5, max_value=max(5, max_bac), value=min(topN_global, max_bac))
        mean_vals = sub[bacteria_cols].apply(safe_numeric).mean().sort_values(ascending=False)
        if normalize:
            mean_vals = normalize_series(mean_vals)
        top_bac = mean_vals.head(n_bac)
        fig = px.bar(x=[clean_name(i) for i in top_bac.index], y=top_bac.values, color=top_bac.values, color_continuous_scale="Viridis")
        fig.update_layout(xaxis_tickangle=-40, title=f"Top {n_bac} bacterial genera (mean)")
        apply_plot_theme(fig, theme_choice)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download top bacteria CSV", data=df_to_csv_bytes(top_bac.reset_index().rename(columns={0: "value", "index": "feature"})), file_name="top_bacteria.csv")

        if st.checkbox("Show correlation heatmap for these top genera", value=False):
            corr_n = min(50, len(top_bac))
            mat = sub[top_bac.head(corr_n).index].apply(safe_numeric).corr()
            if mat.empty:
                st.info("Not enough numeric data for correlation.")
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(mat, ax=ax, cmap="vlag", center=0)
                ax.set_title("Correlation heatmap (top bacteria)")
                st.pyplot(fig)
                st.download_button("Download bacteria correlation CSV", data=df_to_csv_bytes(mat.reset_index()), file_name="bacteria_correlation.csv")

# -------------------------
# Fungi Tab
# -------------------------
with tabs[2]:
    st.header("Fungi — top genera & correlation")
    if len(fungi_cols) == 0:
        st.info("No fungal features detected.")
    else:
        max_fun = len(fungi_cols)
        n_fun = st.slider("Top N fungal genera", min_value=5, max_value=max(5, max_fun), value=min(topN_global, max_fun))
        mean_vals = sub[fungi_cols].apply(safe_numeric).mean().sort_values(ascending=False)
        if normalize:
            mean_vals = normalize_series(mean_vals)
        top_fun = mean_vals.head(n_fun)
        fig = px.bar(x=[clean_name(i) for i in top_fun.index], y=top_fun.values, color=top_fun.values, color_continuous_scale="Magma")
        fig.update_layout(xaxis_tickangle=-40, title=f"Top {n_fun} fungal genera (mean)")
        apply_plot_theme(fig, theme_choice)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download top fungi CSV", data=df_to_csv_bytes(top_fun.reset_index().rename(columns={0: "value", "index": "feature"})), file_name="top_fungi.csv")

        if st.checkbox("Show correlation heatmap for these top fungi", value=False):
            corr_n = min(50, len(top_fun))
            mat = sub[top_fun.head(corr_n).index].apply(safe_numeric).corr()
            if mat.empty:
                st.info("Not enough numeric data for correlation.")
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(mat, ax=ax, cmap="vlag", center=0)
                ax.set_title("Correlation heatmap (top fungi)")
                st.pyplot(fig)
                st.download_button("Download fungi correlation CSV", data=df_to_csv_bytes(mat.reset_index()), file_name="fungi_correlation.csv")

# -------------------------
# Immune Tab
# -------------------------
with tabs[3]:
    st.header("Immune features")
    immune_categories = detect_immune_groups(sub if not sub.empty else df)
    st.write("Detected:", {k: len(v) for k, v in immune_categories.items()})

    plot_style = st.selectbox("Plot style", ["Summary (bar)", "Distribution (box/violin)"])
    stat_choice = st.selectbox("Summary stat", ["Mean", "Median"])
    show_std = st.checkbox("Show error bars in summary", value=True)

    for cat, cols in immune_categories.items():
        with st.expander(f"{cat} ({len(cols)})", expanded=True):
            if not cols:
                st.info("No features detected for this category.")
                continue

            if plot_style.startswith("Summary"):
                rows = []
                for c in cols:
                    vals = safe_numeric(sub[c]).dropna()
                    if normalize:
                        vals = normalize_series(vals)
                    if len(vals) == 0:
                        continue
                    val = vals.mean() if stat_choice == "Mean" else vals.median()
                    rows.append((clean_name(c), val, vals.std()))
                if rows:
                    df_sum = pd.DataFrame(rows, columns=["feature", "value", "std"]).set_index("feature")
                    fig = px.bar(df_sum, x=df_sum.index, y="value", color=df_sum["value"], color_continuous_scale="Plasma")
                    if show_std:
                        fig.data[0].error_y = dict(type="data", array=df_sum["std"].values)
                    apply_plot_theme(fig, theme_choice)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                feat = st.selectbox(f"Pick feature in {cat}", cols, format_func=clean_name, key=f"dist_{cat}")
                vals = safe_numeric(sub[feat]).dropna()
                if normalize:
                    vals = normalize_series(vals)
                fig = px.violin(vals, box=True, points="all", title=clean_name(feat))
                apply_plot_theme(fig, theme_choice)
                st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Clinical / Survival Tab
# -------------------------
with tabs[4]:
    st.header("Clinical & Survival")
    if "OS.time" not in df.columns:
        st.warning("OS.time not present in dataset — survival disabled.")
    else:
        os_series = safe_numeric(sub["OS.time"]).dropna()
        if normalize:
            os_series = normalize_series(os_series)
        st.subheader("OS.time distribution")
        fig = px.histogram(os_series, nbins=40,
                           title="OS.time distribution",
                           color_discrete_sequence=["#FF7F0E"])
        apply_plot_theme(fig, theme_choice)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Kaplan–Meier stratification")
        if HAS_LIFELINES:
            # Build groups of features
            strat_sources = {
                "Top-variance auto": None,
                "All Bacteria": bacteria_cols,
                "All Fungi": fungi_cols,
                "Immune Scores": immune_categories.get("Scores", []),
                "Immune Checkpoints": immune_categories.get("Checkpoints", []),
                "Immune Cytokines": immune_categories.get("Cytokines", []),
                "Immune CellTypes": immune_categories.get("CellTypes", []),
                "Any Column": [c for c in df.columns if safe_numeric(df[c]).notna().sum() > 0],
            }

            strat_group = st.selectbox("Pick feature group", list(strat_sources.keys()))

            # Pick candidate feature
            if strat_group == "Top-variance auto":
                var_all = sub.select_dtypes(include=[np.number]).var().sort_values(ascending=False)
                candidate = var_all.index[0] if not var_all.empty else None
            else:
                options = strat_sources[strat_group]
                if not options:
                    candidate = None
                else:
                    candidate = st.selectbox(f"Pick feature from {strat_group}", options, format_func=clean_name)

            # Run KM if candidate valid
            if candidate is None:
                st.info("No numeric candidate found to stratify.")
            else:
                km_df = sub[["_primary_disease", "OS.time", candidate]].dropna(subset=["OS.time", candidate])
                if km_df.empty:
                    st.info("Not enough data to perform stratification with this feature.")
                else:
                    method = st.radio("Split method", ["Median", "Quartiles", "Custom cutoff"])
                    vals = safe_numeric(km_df[candidate])
                    if method == "Custom cutoff":
                        cutoff = st.number_input("Cutoff value", value=float(vals.median()))
                        km_df["group"] = (vals >= cutoff).astype(int).astype(str)
                    elif method == "Median":
                        km_df["group"] = (vals >= vals.median()).astype(int).astype(str)
                    else:
                        km_df["group"] = pd.qcut(vals, q=4, labels=[f"Q{i}" for i in range(1, 5)])

                    from lifelines import KaplanMeierFitter
                    kmf = KaplanMeierFitter()
                    fig = go.Figure()
                    for g, grp in km_df.groupby("group"):
                        durations = safe_numeric(grp["OS.time"]).astype(float)
                        events = np.ones(len(durations))  # assume all observed
                        kmf.fit(durations, event_observed=events, label=str(g))
                        sf = kmf.survival_function_
                        fig.add_trace(go.Scatter(
                            x=sf.index, y=sf[kmf._label],
                            mode="lines", name=f"{clean_name(candidate)} {g}"
                        ))
                    fig.update_layout(
                        title=f"Kaplan–Meier by {clean_name(candidate)}",
                        xaxis_title="Time",
                        yaxis_title="Survival probability"
                    )
                    apply_plot_theme(fig, theme_choice)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Install `lifelines` to enable Kaplan–Meier plots: pip install lifelines")



# -------------------------
# PCA / UMAP Tab
# -------------------------
with tabs[5]:
    st.header("PCA & UMAP — choose features")
    st.markdown("Pick groups or exact features. PCA/UMAP operate on numeric features only.")
    group_choices = st.multiselect(
        "Feature groups",
        ["Top bacteria", "Top fungi", "Immune Scores", "Immune CellTypes"],
        default=["Top bacteria"]
    )

    feats = []
    if "Top bacteria" in group_choices:
        feats += list(bacteria_cols[:topN_global])
    if "Top fungi" in group_choices:
        feats += list(fungi_cols[:topN_global])
    if "Immune Scores" in group_choices:
        feats += immune_categories.get("Scores", [])[:topN_global]
    if "Immune CellTypes" in group_choices:
        feats += immune_categories.get("CellTypes", [])[:topN_global]

    manual_feats = st.multiselect(
        "Or pick exact features (overrides groups)",
        options=sorted(df.columns),
        default=[]
    )
    if manual_feats:
        feats = manual_feats

    feats = [f for f in feats if f in sub.columns]
    st.write(f"{len(feats)} features selected for dimensionality reduction.")

    if len(feats) < 2:
        st.info("Pick at least 2 numeric features.")
    else:
        X = sub[feats].fillna(0).apply(safe_numeric).fillna(0).astype(float)
        if normalize:
            X = X.apply(normalize_series)

        # PCA
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)
        dfp = pd.DataFrame(proj, columns=["PC1", "PC2"], index=X.index)
        dfp["cancer"] = sub["_primary_disease"].fillna("Unknown").values
        fig = px.scatter(
            dfp, x="PC1", y="PC2", color="cancer",
            title="PCA (2 components)",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=600
        )
        apply_plot_theme(fig, theme_choice)
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        st.plotly_chart(fig, use_container_width=True)
        st.write("Explained variance (PC1, PC2):", list(pca.explained_variance_ratio_.round(4)))

        # UMAP
        if HAS_UMAP:
            st.subheader("UMAP projection")
            with st.spinner("Running UMAP..."):
                try:
                    emb = umap.UMAP(
                        n_neighbors=15,
                        min_dist=0.1,
                        random_state=42,
                        metric="euclidean"
                    ).fit_transform(X)

                    dfe = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"], index=X.index)
                    dfe["cancer"] = sub["_primary_disease"].fillna("Unknown").values

                    fig2 = px.scatter(
                        dfe, x="UMAP1", y="UMAP2", color="cancer",
                        title="UMAP projection",
                        color_discrete_sequence=px.colors.qualitative.Dark24,
                        height=600
                    )
                    apply_plot_theme(fig2, theme_choice)
                    fig2.update_traces(marker=dict(size=7, line=dict(width=0)))
                    st.plotly_chart(fig2, use_container_width=True)

                except Exception as e:
                    st.error(f"UMAP failed: {e}")
        else:
            st.info("UMAP not installed. Run: pip install umap-learn")


# -------------------------
# Differential (Volcano) Tab
# -------------------------
with tabs[6]:
    st.header("Differential abundance (Volcano) — compare two cancer types")
    cancers_list = sorted(df["_primary_disease"].dropna().unique().tolist())
    cancer_a = st.selectbox("Cancer A (group 1)", ["-- select --"] + cancers_list, index=0)
    cancer_b = st.selectbox("Cancer B (group 2)", ["-- select --"] + cancers_list, index=0)
    genus_pool = st.multiselect("Features to test (empty = bacteria + fungi)", options=sorted(list(set(bacteria_cols + fungi_cols))), default=[])
    p_adj_method = st.selectbox("P-value adjustment", ["none", "bonferroni", "fdr_bh"])
    if st.button("Run differential test"):
        if cancer_a == "-- select --" or cancer_b == "-- select --" or cancer_a == cancer_b:
            st.error("Pick two different cancer types.")
        else:
            if not HAS_SCIPY:
                st.error("scipy is required for statistical tests. Install scipy.")
            else:
                if not genus_pool:
                    features_to_test = bacteria_cols + fungi_cols
                else:
                    features_to_test = genus_pool
                a_df = df[df["_primary_disease"] == cancer_a]
                b_df = df[df["_primary_disease"] == cancer_b]
                results = []
                for f in features_to_test:
                    if f not in df.columns:
                        continue
                    a_vals = safe_numeric(a_df[f]).dropna()
                    b_vals = safe_numeric(b_df[f]).dropna()
                    if len(a_vals) < 3 or len(b_vals) < 3:
                        continue
                    mean_a = a_vals.mean() + 1e-9
                    mean_b = b_vals.mean() + 1e-9
                    log2fc = math.log2(mean_a / mean_b)
                    try:
                        tstat, pval = stats.ttest_ind(a_vals, b_vals, equal_var=False, nan_policy="omit")
                    except Exception:
                        pval = np.nan
                    results.append((f, log2fc, pval))
                res_df = pd.DataFrame(results, columns=["feature", "log2FC", "pval"]).dropna()
                if res_df.empty:
                    st.info("No features passed filtering for test.")
                else:
                    if p_adj_method != "none" and HAS_STATSMODELS:
                        adj = multipletests(res_df["pval"].values, method="fdr_bh" if p_adj_method == "fdr_bh" else "bonferroni")[1]
                        res_df["p_adj"] = adj
                    else:
                        res_df["p_adj"] = res_df["pval"]
                    res_df["neglog10p"] = -np.log10(res_df["p_adj"].replace(0, 1e-300))
                    fig = px.scatter(res_df, x="log2FC", y="neglog10p", hover_name=res_df["feature"].map(clean_name),
                                     title=f"Volcano: {cancer_a} vs {cancer_b}", color="neglog10p", color_continuous_scale="Inferno")
                    apply_plot_theme(fig, theme_choice)
                    st.plotly_chart(fig, use_container_width=True)
                    st.download_button("Download differential results CSV", data=df_to_csv_bytes(res_df), file_name="differential_results.csv")

# -------------------------
# Heatmap Explorer Tab
# -------------------------
with tabs[7]:
    st.header("Heatmap Explorer")
    default_feats = (
        list(bacteria_cols[:topN_global]) +
        list(fungi_cols[:topN_global]) +
        immune_categories.get("Scores", [])[:10]
    )
    heat_feats = st.multiselect(
        "Select features for heatmap (choose many)",
        options=list(df.columns),
        default=default_feats[:min(50, len(default_feats))]
    )

    if heat_feats:
        mat = sub[heat_feats].fillna(0).apply(safe_numeric).fillna(0)
        if mat.empty:
            st.info("No numeric values for selected features.")
        else:
            # 🔹 Option to scale values
            scale_method = st.radio("Scaling method", ["None", "Log2(x+1)", "Z-score"], index=2)

            if scale_method == "Log2(x+1)":
                mat_scaled = np.log2(mat + 1)
            elif scale_method == "Z-score":
                mat_scaled = mat.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=0)
            else:
                mat_scaled = mat

            # 🔹 Interactive heatmap (better colormap: inferno)
            fig = px.imshow(
                mat_scaled.T,
                labels=dict(x="Sample", y="Feature", color="Value"),
                aspect="auto",
                title=f"Heatmap ({scale_method} scaled)",
                color_continuous_scale="inferno",   # 🔥 vibrant colormap
                height=700
            )
            apply_plot_theme(fig, theme_choice)
            st.plotly_chart(fig, use_container_width=True)

            # 🔹 Optional clustered heatmap (static, same colormap)
            if st.checkbox("Show clustered heatmap (static)", value=False):
                try:
                    cg = sns.clustermap(
                        mat_scaled,
                        cmap="inferno",
                        metric="correlation",
                        figsize=(12, 10),
                        xticklabels=False,
                        yticklabels=True
                    )
                    st.pyplot(cg.fig)
                except Exception as e:
                    st.error(f"Clustermap failed: {e}")

            st.download_button(
                "Download heatmap matrix CSV",
                data=df_to_csv_bytes(mat_scaled.reset_index()),
                file_name="heatmap_matrix.csv"
            )
    else:
        st.info("Pick features to visualize in heatmap.")


# -------------------------
# ML Playground Tab
# -------------------------
with tabs[8]:
    st.header("ML Playground — classification & explainability")
    st.markdown("Select feature groups or exact features, then run classification. Optional SHAP explanations available if shap is installed.")

    use_bac = st.checkbox("Include bacteria features", value=True)
    use_fun = st.checkbox("Include fungi features", value=False)
    use_imm = st.checkbox("Include immune features (scores + cell types)", value=False)
    manual_feats_ml = st.multiselect("Or pick exact features for ML (overrides groups)", options=list(df.columns), default=[])

    feats = []
    if use_bac:
        feats += bacteria_cols
    if use_fun:
        feats += fungi_cols
    if use_imm:
        feats += immune_categories.get("Scores", []) + immune_categories.get("CellTypes", [])
    if manual_feats_ml:
        feats = manual_feats_ml
    feats = [f for f in feats if f in df.columns]

    st.write(f"Features chosen: {len(feats)}")
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression", "SVM (linear)", "SVM (RBF)", "Gradient Boosting", "KNN", "XGBoost (if installed)"])
    cv_folds = st.slider("CV folds", 2, 10, 5)
    run_shap = st.checkbox("Compute SHAP explanations (may be slow)", value=False)

    if len(feats) < 2:
        st.info("Select at least 2 features to run ML.")
    else:
        if st.button("Run ML (train/test)"):
            data = df[feats + ["_primary_disease"]].dropna(subset=["_primary_disease"])
            if data.shape[0] < 10:
                st.error("Not enough samples to run ML.")
            else:
                X = data[feats].fillna(0).apply(safe_numeric).fillna(0).astype(float).values
                y = data["_primary_disease"].astype(str).values

                unique_classes = np.unique(y)
                if len(unique_classes) > 12:
                    top_labels = pd.Series(y).value_counts().head(12).index.tolist()
                    mask = pd.Series(y).isin(top_labels)
                    X = X[mask.values]; y = y[mask.values]
                    st.warning("Too many target classes — using top 12 for demo.")

                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr); Xte = scaler.transform(Xte)

                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=2000)
                elif model_choice == "SVM (linear)":
                    model = SVC(kernel="linear", probability=True)
                elif model_choice == "SVM (RBF)":
                    model = SVC(kernel="rbf", probability=True)
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                elif model_choice == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)
                elif model_choice == "XGBoost (if installed)":
                    if HAS_XGBOOST:
                        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
                    else:
                        st.warning("XGBoost not installed; using GradientBoosting instead.")
                        model = GradientBoostingClassifier()
                else:
                    model = RandomForestClassifier(n_estimators=200, random_state=42)

                t0 = time.time()
                model.fit(Xtr, ytr)
                preds = model.predict(Xte)
                t1 = time.time()
                acc = (preds == yte).mean()
                st.success(f"Accuracy (test): {acc:.3f} — training time {t1 - t0:.1f}s")

                st.text("Classification report (test):")
                st.text(classification_report(yte, preds))

                labels = np.unique(yte)
                cm = confusion_matrix(yte, preds, labels=labels)
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(Xte)
                        if probs.shape[1] == 2:
                            fpr, tpr, _ = roc_curve((yte == labels[1]).astype(int), probs[:, 1])
                            aucv = auc(fpr, tpr)
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={aucv:.3f}"))
                            fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")))
                            fig2.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                            apply_plot_theme(fig2, theme_choice)
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            lb = LabelBinarizer()
                            yte_bin = lb.fit_transform(yte)
                            aucv = roc_auc_score(yte_bin, probs, average="macro", multi_class="ovr")
                            st.info(f"Multi-class ROC AUC (macro): {aucv:.3f}")
                    except Exception:
                        st.info("Could not compute ROC/probabilities.")

                if hasattr(model, "feature_importances_"):
                    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).head(30)
                    fig3 = px.bar(x=[clean_name(i) for i in imp.index], y=imp.values, color=imp.values, color_continuous_scale="Cividis", title="Top feature importances")
                    fig3.update_layout(xaxis_tickangle=-45)
                    apply_plot_theme(fig3, theme_choice)
                    st.plotly_chart(fig3, use_container_width=True)

                if run_shap:
                    if not HAS_SHAP:
                        st.warning("SHAP not installed. pip install shap to enable.")
                    else:
                        with st.spinner("Computing SHAP values (may take long)..."):
                            try:
                                explainer = shap.Explainer(model, Xtr) if not HAS_XGBOOST else shap.TreeExplainer(model)
                                shap_vals = explainer(Xte)
                                st.subheader("SHAP summary (bar)")
                                try:
                                    shap.plots.bar(shap_vals, max_display=30)
                                except Exception:
                                    st.info("SHAP plotting fallback: show numeric values table.")
                                    sv = pd.DataFrame(shap_vals.values, columns=[clean_name(f) for f in feats])
                                    st.dataframe(sv.head())
                            except Exception as e:
                                st.error(f"SHAP computation failed: {e}")

                try:
                    cv_scores = cross_val_score(model, Xtr, ytr, cv=min(5, max(2, len(ytr)//5)))
                    st.write(f"Cross-validation mean (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                except Exception:
                    st.info("Cross-validation not run (too few samples?).")

# -------------------------
# Outlier Detection Tab
# -------------------------
with tabs[9]:
    st.header("Outlier detection")
    method = st.selectbox("Method", ["Isolation Forest", "Z-score per feature"])
    choose_feats = st.multiselect("Features to include for outlier detection", options=list(df.columns), default=list(bacteria_cols[:20]) + list(immune_categories.get("Scores", [])[:10]))
    if st.button("Run outlier detection"):
        if len(choose_feats) < 2:
            st.error("Pick at least 2 features.")
        else:
            X = df[choose_feats].fillna(0).apply(safe_numeric).fillna(0).astype(float)
            if method == "Isolation Forest":
                iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                preds = iso.fit_predict(X)
                df_out = df.copy()
                df_out["_outlier_flag"] = (preds == -1)
                st.write(f"Detected outliers: {df_out['_outlier_flag'].sum()}")
                st.dataframe(df_out[df_out["_outlier_flag"]].head(50))
                st.download_button("Download outliers CSV", data=df_to_csv_bytes(df_out[df_out["_outlier_flag"]].reset_index()), file_name="outliers.csv")
            else:
                z = (X - X.mean()) / X.std()
                outlier_mask = (z.abs() > 3).any(axis=1)
                st.write(f"Detected outliers by Z>3 in any selected feature: {outlier_mask.sum()}")
                st.dataframe(df[outlier_mask].head(50))
                st.download_button("Download outliers CSV (zscore)", data=df_to_csv_bytes(df[outlier_mask].reset_index()), file_name="outliers_zscore.csv")

# -------------------------
# Footer / tips
# -------------------------
st.markdown("---")
st.caption(
    "###Data Sources\n"
    "- **Knight Lab Pan-Cancer Mycobiome dataset**: Narunsky-Haziza et al., Cell 2022. "
    "[DOI: 10.1016/j.cell.2022.09.005](https://doi.org/10.1016/j.cell.2022.09.005)\n"
    "- **TCGA Pan-Cancer Atlas**: Data downloaded via the UCSC Xena Browser "
    "(Goldman et al., Nucleic Acids Research 2020)."
)
