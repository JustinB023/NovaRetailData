"""
NovaRetail — Customer Intelligence Dashboard (Streamlit)

Run locally (repo root must contain NR_dataset.xlsx):
  pip install -r requirements.txt
  streamlit run app.py

Deployment: push to GitHub and deploy on Streamlit Community Cloud (see README.md).

Key insights this app is designed to surface (reproducible with NR_dataset.xlsx):
1) Revenue concentration: revenue from the top 10% customers (by lifetime revenue in the selected window).
2) Segment performance: which segment (Promising/Growth/Stable/Decline) leads revenue vs satisfaction (growth vs risk).
3) Early warning: customers with both declining purchase frequency and falling satisfaction (risk score).

Constraints:
- Offline-only, no external APIs, no secrets.
- Single-file Streamlit app with modular functions.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from streamlit_plotly_events import plotly_events


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "NovaRetail — Customer Intelligence Dashboard"
DATA_PATH = "NR_dataset.xlsx"

REQUIRED_COLS = [
    "CustomerID",
    "TransactionID",
    "TransactionDate",
    "ProductCategory",
    "PurchaseAmount",
    "CustomerAgeGroup",
    "CustomerGender",
    "CustomerRegion",
    "CustomerSatisfaction",
    "RetailChannel",
    "label",
]

SEGMENT_ORDER = ["Promising", "Growth", "Stable", "Decline"]
AGG_MAP = {"Daily": "D", "Monthly": "M", "Quarterly": "Q"}


# -----------------------------
# Helpers
# -----------------------------
def fmt_currency(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def minmax_01(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. If constant, return zeros."""
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def to_png_bytes(fig) -> Tuple[bytes, str]:
    """
    Export Plotly figure to PNG bytes (requires kaleido).
    If PNG export fails, returns HTML bytes as a fallback.
    """
    try:
        return pio.to_image(fig, format="png", scale=2), "image/png"
    except Exception:
        html = pio.to_html(fig, include_plotlyjs="cdn")
        return html.encode("utf-8"), "text/html"


# -----------------------------
# ETL
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load Excel file from repo root and standardize schema/types."""
    df = pd.read_excel(path, engine="openpyxl")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")

    df = df[REQUIRED_COLS].copy()

    # Consistent ID typing
    df["CustomerID"] = df["CustomerID"].apply(safe_str)
    df["TransactionID"] = df["TransactionID"].apply(safe_str)

    # Date parsing
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df = df.dropna(subset=["TransactionDate"]).copy()

    # Numeric parsing/sanity
    df["PurchaseAmount"] = pd.to_numeric(df["PurchaseAmount"], errors="coerce").fillna(0.0)
    df["PurchaseAmount"] = df["PurchaseAmount"].clip(lower=0)

    df["CustomerSatisfaction"] = pd.to_numeric(df["CustomerSatisfaction"], errors="coerce")
    df.loc[~df["CustomerSatisfaction"].between(1, 5, inclusive="both"), "CustomerSatisfaction"] = np.nan

    # Categoricals
    cat_cols = [
        "ProductCategory",
        "CustomerAgeGroup",
        "CustomerGender",
        "CustomerRegion",
        "RetailChannel",
        "label",
    ]
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("Unknown")

    df["YearMonth"] = df["TransactionDate"].dt.to_period("M").astype(str)

    return df


def validate_data(df: pd.DataFrame) -> Dict[str, object]:
    """Validation summary shown when the app starts."""
    # Unit-test style asserts (kept here so issues are explicit)
    assert all(c in df.columns for c in REQUIRED_COLS), "Required columns missing after load."
    assert (df["PurchaseAmount"] >= 0).all(), "PurchaseAmount contains negative values."
    assert df["CustomerID"].map(type).isin([str]).all(), "CustomerID should be string after conversion."

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "unique_customers": int(df["CustomerID"].nunique()),
        "date_min": str(df["TransactionDate"].min().date()),
        "date_max": str(df["TransactionDate"].max().date()),
        "missing_by_col": df.isna().sum().to_dict(),
    }


# -----------------------------
# Filtering
# -----------------------------
@dataclass
class Filters:
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    product_categories: List[str]
    regions: List[str]
    channels: List[str]
    segments: List[str]
    age_groups: List[str]
    genders: List[str]
    customer_id_search: str
    agg_level: str


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()
    out = out[(out["TransactionDate"] >= f.date_min) & (out["TransactionDate"] <= f.date_max)]
    if f.product_categories:
        out = out[out["ProductCategory"].isin(f.product_categories)]
    if f.regions:
        out = out[out["CustomerRegion"].isin(f.regions)]
    if f.channels:
        out = out[out["RetailChannel"].isin(f.channels)]
    if f.segments:
        out = out[out["label"].isin(f.segments)]
    if f.age_groups:
        out = out[out["CustomerAgeGroup"].isin(f.age_groups)]
    if f.genders:
        out = out[out["CustomerGender"].isin(f.genders)]
    if f.customer_id_search.strip():
        needle = f.customer_id_search.strip()
        out = out[out["CustomerID"].astype(str).str.contains(needle, case=False, na=False)]
    return out


def init_click_state():
    st.session_state.setdefault("click_segment", None)
    st.session_state.setdefault("click_category", None)
    st.session_state.setdefault("click_region", None)


def sidebar_filters(df: pd.DataFrame) -> Filters:
    st.sidebar.header("Filters")

    dmin, dmax = df["TransactionDate"].min(), df["TransactionDate"].max()
    date_range = st.sidebar.slider(
        "TransactionDate range",
        min_value=dmin.to_pydatetime(),
        max_value=dmax.to_pydatetime(),
        value=(dmin.to_pydatetime(), dmax.to_pydatetime()),
    )

    # Defaults are "all", but cross-filtering clicks can set single selections.
    all_cats = sorted(df["ProductCategory"].unique().tolist())
    all_regs = sorted(df["CustomerRegion"].unique().tolist())
    all_ch = sorted(df["RetailChannel"].unique().tolist())
    all_seg = sorted(df["label"].unique().tolist())
    all_age = sorted(df["CustomerAgeGroup"].unique().tolist())
    all_gender = sorted(df["CustomerGender"].unique().tolist())

    default_segments = all_seg
    if st.session_state.get("click_segment"):
        default_segments = [st.session_state["click_segment"]] if st.session_state["click_segment"] in all_seg else all_seg

    default_categories = all_cats
    if st.session_state.get("click_category"):
        default_categories = (
            [st.session_state["click_category"]] if st.session_state["click_category"] in all_cats else all_cats
        )

    default_regions = all_regs
    if st.session_state.get("click_region"):
        default_regions = [st.session_state["click_region"]] if st.session_state["click_region"] in all_regs else all_regs

    if st.sidebar.button("Clear chart selections"):
        st.session_state["click_segment"] = None
        st.session_state["click_category"] = None
        st.session_state["click_region"] = None
        st.rerun()

    product_categories = st.sidebar.multiselect("ProductCategory", options=all_cats, default=default_categories)
    regions = st.sidebar.multiselect("CustomerRegion", options=all_regs, default=default_regions)
    channels = st.sidebar.multiselect("RetailChannel", options=all_ch, default=all_ch)
    segments = st.sidebar.multiselect("Segment (label)", options=all_seg, default=default_segments)
    age_groups = st.sidebar.multiselect("CustomerAgeGroup", options=all_age, default=all_age)
    genders = st.sidebar.multiselect("CustomerGender", options=all_gender, default=all_gender)

    agg_level = st.sidebar.radio("Aggregation level", ["Daily", "Monthly", "Quarterly"], index=1)
    customer_id_search = st.sidebar.text_input("Search CustomerID (contains)", value="")

    return Filters(
        date_min=pd.Timestamp(date_range[0]),
        date_max=pd.Timestamp(date_range[1]),
        product_categories=product_categories,
        regions=regions,
        channels=channels,
        segments=segments,
        age_groups=age_groups,
        genders=genders,
        customer_id_search=customer_id_search,
        agg_level=agg_level,
    )


# -----------------------------
# Metrics / Features
# -----------------------------
def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("CustomerID", as_index=False)
    cust = g.agg(
        CustomerLifetimeRevenue=("PurchaseAmount", "sum"),
        PurchaseFrequency=("TransactionID", "nunique"),
        LastTransactionDate=("TransactionDate", "max"),
        Segment=("label", lambda s: s.iloc[-1]),
        Region=("CustomerRegion", lambda s: s.iloc[-1]),
        AvgSatisfaction=("CustomerSatisfaction", "mean"),
    )
    cust["LastTransactionDate"] = cust["LastTransactionDate"].dt.date.astype(str)
    return cust


def compute_kpis(df_filt: pd.DataFrame, cust: pd.DataFrame) -> Dict[str, object]:
    total_rev = float(df_filt["PurchaseAmount"].sum())
    avg_purchase = float(df_filt["PurchaseAmount"].mean()) if len(df_filt) else 0.0
    n_customers = int(df_filt["CustomerID"].nunique())

    seg_rev = df_filt.groupby("label", as_index=False)["PurchaseAmount"].sum().sort_values("PurchaseAmount", ascending=False)
    top_segment = seg_rev["label"].iloc[0] if len(seg_rev) else "—"

    if len(cust):
        cust_sorted = cust.sort_values("CustomerLifetimeRevenue", ascending=False).copy()
        top_n = max(1, int(math.ceil(0.10 * len(cust_sorted))))
        top10_rev = float(cust_sorted.head(top_n)["CustomerLifetimeRevenue"].sum())
    else:
        top10_rev = 0.0

    avg_sat = float(df_filt["CustomerSatisfaction"].mean()) if df_filt["CustomerSatisfaction"].notna().any() else np.nan

    return {
        "Total Revenue": total_rev,
        "Avg Purchase Amount": avg_purchase,
        "# Unique Customers": n_customers,
        "Top Segment by Revenue": top_segment,
        "Revenue from Top 10% Customers": top10_rev,
        "Avg Customer Satisfaction": avg_sat,
    }


def aggregate_timeseries(df_filt: pd.DataFrame, agg_level: str) -> pd.DataFrame:
    rule = AGG_MAP.get(agg_level, "M")
    ts = df_filt.copy()
    ts["bucket"] = ts["TransactionDate"].dt.to_period(rule).dt.to_timestamp()
    out = ts.groupby(["bucket", "label"], as_index=False).agg(Revenue=("PurchaseAmount", "sum"))
    out = out.sort_values("bucket")
    return out


def segment_health(df_filt: pd.DataFrame) -> pd.DataFrame:
    g = df_filt.groupby("label", as_index=False).agg(
        Revenue=("PurchaseAmount", "sum"),
        AvgPurchase=("PurchaseAmount", "mean"),
        AvgSatisfaction=("CustomerSatisfaction", "mean"),
        Customers=("CustomerID", "nunique"),
    )
    total_rev = df_filt["PurchaseAmount"].sum()
    g["RevenueShare"] = np.where(total_rev > 0, g["Revenue"] / total_rev, 0.0)
    g["label"] = pd.Categorical(g["label"], categories=SEGMENT_ORDER, ordered=True)
    g = g.sort_values("label")
    g["label"] = g["label"].astype(str)
    return g


def pct_customers_declining_mom(df_filt: pd.DataFrame) -> float:
    if df_filt.empty:
        return 0.0
    tmp = df_filt.copy()
    tmp["ym"] = tmp["TransactionDate"].dt.to_period("M")
    months = sorted(tmp["ym"].unique())
    if len(months) < 2:
        return 0.0
    last_m, prev_m = months[-1], months[-2]
    by_c_m = tmp.groupby(["CustomerID", "ym"], as_index=False)["PurchaseAmount"].sum()
    pivot = by_c_m.pivot(index="CustomerID", columns="ym", values="PurchaseAmount").fillna(0.0)
    if prev_m not in pivot.columns or last_m not in pivot.columns:
        return 0.0
    declining = (pivot[last_m] < pivot[prev_m]) & (pivot[prev_m] > 0)
    denom = (pivot[prev_m] > 0).sum()
    return float(declining.sum() / denom) if denom else 0.0


def compute_risk_scores(df_filt: pd.DataFrame) -> pd.DataFrame:
    """
    risk_score = 0.6 * normalized_drop_in_purchase_frequency + 0.4 * normalized_drop_in_satisfaction
    Drops compare last month vs previous month within the selected window.
    """
    if df_filt.empty:
        return pd.DataFrame(columns=["CustomerID", "risk_score"])

    tmp = df_filt.copy()
    tmp["ym"] = tmp["TransactionDate"].dt.to_period("M")
    months = sorted(tmp["ym"].unique())
    if len(months) < 2:
        base = tmp.groupby("CustomerID", as_index=False).agg(
            Segment=("label", lambda s: s.iloc[-1]),
            Region=("CustomerRegion", lambda s: s.iloc[-1]),
            LastTransactionDate=("TransactionDate", "max"),
            AvgSatisfaction=("CustomerSatisfaction", "mean"),
        )
        base["risk_score"] = 0.0
        base["LastTransactionDate"] = base["LastTransactionDate"].dt.date.astype(str)
        return base.sort_values("risk_score", ascending=False)

    last_m, prev_m = months[-1], months[-2]

    freq = tmp.groupby(["CustomerID", "ym"], as_index=False).agg(Tx=("TransactionID", "nunique"))
    freq_p = freq.pivot(index="CustomerID", columns="ym", values="Tx").fillna(0.0)
    prev_tx = freq_p[prev_m] if prev_m in freq_p.columns else pd.Series(0.0, index=freq_p.index)
    last_tx = freq_p[last_m] if last_m in freq_p.columns else pd.Series(0.0, index=freq_p.index)
    drop_freq = np.where(prev_tx > 0, (prev_tx - last_tx).clip(lower=0) / prev_tx, 0.0)

    sat = tmp.groupby(["CustomerID", "ym"], as_index=False).agg(Sat=("CustomerSatisfaction", "mean"))
    sat_p = sat.pivot(index="CustomerID", columns="ym", values="Sat")
    prev_sat = sat_p.get(prev_m, pd.Series(np.nan, index=sat_p.index))
    last_sat = sat_p.get(last_m, pd.Series(np.nan, index=sat_p.index))
    drop_sat = (prev_sat - last_sat).fillna(0.0).clip(lower=0.0)

    drop_freq_n = minmax_01(pd.Series(drop_freq, index=freq_p.index))
    drop_sat_n = minmax_01(pd.Series(drop_sat, index=sat_p.index))

    ix = sorted(set(drop_freq_n.index).union(set(drop_sat_n.index)))
    drop_freq_n = drop_freq_n.reindex(ix).fillna(0.0)
    drop_sat_n = drop_sat_n.reindex(ix).fillna(0.0)

    risk = 0.6 * drop_freq_n + 0.4 * drop_sat_n

    base = tmp.groupby("CustomerID", as_index=False).agg(
        Segment=("label", lambda s: s.iloc[-1]),
        Region=("CustomerRegion", lambda s: s.iloc[-1]),
        LastTransactionDate=("TransactionDate", "max"),
        AvgSatisfaction=("CustomerSatisfaction", "mean"),
    )
    base["risk_score"] = base["CustomerID"].map(risk.to_dict()).fillna(0.0)
    base["LastTransactionDate"] = base["LastTransactionDate"].dt.date.astype(str)
    return base.sort_values("risk_score", ascending=False)


def what_if_lift(df_filt: pd.DataFrame, target_segment: str, retention_uplift_pct: float) -> Tuple[float, str]:
    """
    Conservative projection:
    - Average monthly revenue for target segment over last 3 months in-window
    - Incremental = avg_monthly * 12 * (uplift%/100) * 0.5 (conservative factor)
    """
    if df_filt.empty or retention_uplift_pct <= 0 or not target_segment:
        return 0.0, "No projection available."

    tmp = df_filt[df_filt["label"] == target_segment].copy()
    if tmp.empty:
        return 0.0, f"No data for segment '{target_segment}' in current filters."

    tmp["ym"] = tmp["TransactionDate"].dt.to_period("M").dt.to_timestamp()
    seg_m = tmp.groupby("ym", as_index=False).agg(Revenue=("PurchaseAmount", "sum")).sort_values("ym")
    last3 = seg_m.tail(3)
    if last3.empty:
        return 0.0, "Not enough history for projection."

    avg_monthly = float(last3["Revenue"].mean())
    lift = avg_monthly * 12.0 * (retention_uplift_pct / 100.0) * 0.5
    note = f"Based on avg monthly revenue over last {len(last3)} months in-window (conservative 50% translation)."
    return lift, note


# -----------------------------
# Charts (return figures)
# -----------------------------
def fig_revenue_trend(ts: pd.DataFrame) -> go.Figure:
    # Use graph_objects to ensure trace order maps to curveNumber for click events
    fig = go.Figure()
    present = [s for s in SEGMENT_ORDER if s in ts["label"].unique()]
    for seg in present:
        seg_df = ts[ts["label"] == seg].sort_values("bucket")
        fig.add_trace(
            go.Scatter(
                x=seg_df["bucket"],
                y=seg_df["Revenue"],
                mode="lines+markers",
                name=seg,
                hovertemplate="Segment=%{text}<br>Date=%{x|%Y-%m-%d}<br>Revenue=%{y:$,.0f}<extra></extra>",
                text=[seg] * len(seg_df),
            )
        )
    fig.update_layout(title="Revenue Trend by Segment", xaxis_title="Date", yaxis_title="Revenue", hovermode="x unified")
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def fig_rev_by_category(df_filt: pd.DataFrame) -> go.Figure:
    agg = df_filt.groupby(["ProductCategory", "label"], as_index=False)["PurchaseAmount"].sum()
    fig = px.bar(
        agg,
        x="ProductCategory",
        y="PurchaseAmount",
        color="label",
        barmode="stack",
        category_orders={"label": SEGMENT_ORDER},
        labels={"PurchaseAmount": "Revenue", "label": "Segment"},
        title="Revenue by Product Category (stacked by segment)",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def fig_rev_by_region(df_filt: pd.DataFrame) -> go.Figure:
    agg = df_filt.groupby("CustomerRegion", as_index=False)["PurchaseAmount"].sum().sort_values("PurchaseAmount", ascending=False)
    fig = px.bar(
        agg,
        x="CustomerRegion",
        y="PurchaseAmount",
        labels={"PurchaseAmount": "Revenue", "CustomerRegion": "Region"},
        title="Revenue by Region",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


# -----------------------------
# UI blocks
# -----------------------------
def render_kpis(kpis: Dict[str, object]) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Revenue", fmt_currency(kpis["Total Revenue"]))
    c2.metric("Avg Purchase", fmt_currency(kpis["Avg Purchase Amount"]))
    c3.metric("# Unique Customers", f"{kpis['# Unique Customers']:,}")
    c4.metric("Top Segment (Revenue)", str(kpis["Top Segment by Revenue"]))
    c5.metric("Revenue from Top 10%", fmt_currency(kpis["Revenue from Top 10% Customers"]))
    sat = kpis["Avg Customer Satisfaction"]
    c6.metric("Avg Satisfaction", "—" if pd.isna(sat) else f"{sat:.2f}")


def render_segment_health(health: pd.DataFrame, pct_declining: float) -> None:
    st.subheader("Segment Health")
    st.caption("Tiles update with filters. Use them to spot segments that are high revenue but slipping satisfaction.")
    cols = st.columns(4)
    for i, seg in enumerate(SEGMENT_ORDER):
        seg_row = health[health["label"] == seg]
        if seg_row.empty:
            cols[i].metric(seg, "—")
            continue
        r = seg_row.iloc[0]
        cols[i].metric(
            seg,
            fmt_currency(r["Revenue"]),
            help=f"Revenue share: {r['RevenueShare']*100:.1f}% | Avg purchase: ${r['AvgPurchase']:.0f} | Avg satisfaction: {r['AvgSatisfaction']:.2f}",
        )
    st.caption(f"% customers declining month-over-month (in-window): **{pct_declining*100:.1f}%**")


def render_top_customers(cust: pd.DataFrame) -> None:
    st.subheader("Top Customers by Lifetime Revenue (in-window)")
    top_n = st.slider("Top N customers", min_value=5, max_value=50, value=15, step=5)
    show = cust.sort_values("CustomerLifetimeRevenue", ascending=False).head(top_n).copy()
    show.rename(columns={"CustomerLifetimeRevenue": "LifetimeRevenue", "AvgSatisfaction": "Satisfaction"}, inplace=True)
    show["LifetimeRevenue"] = show["LifetimeRevenue"].round(2)
    show["Satisfaction"] = show["Satisfaction"].round(2)

    cols = ["CustomerID", "LifetimeRevenue", "PurchaseFrequency", "LastTransactionDate", "Segment", "Region", "Satisfaction"]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)

    csv = show[cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (filtered top customers)", csv, "top_customers.csv", "text/csv")


def render_risk_section(risk: pd.DataFrame) -> None:
    st.subheader("Early Warning: At-Risk Customers")
    with st.popover("Risk score formula"):
        st.markdown(
            """
**Risk score** flags customers with **declining purchase frequency** and **falling satisfaction** in the selected time window.

`risk_score = 0.6 * normalized_drop_in_purchase_frequency + 0.4 * normalized_drop_in_satisfaction`

- Purchase frequency drop: last month vs previous month (transaction count)
- Satisfaction drop: last month vs previous month (average satisfaction)
- Components normalized to 0–1 across customers; higher = more at-risk
"""
        )

    top = risk.head(10).copy()
    top["risk_score"] = top["risk_score"].round(3)
    st.dataframe(
        top[["CustomerID", "risk_score", "Segment", "Region", "LastTransactionDate", "AvgSatisfaction"]],
        use_container_width=True,
        hide_index=True,
    )

    if not risk.empty:
        fig = px.box(
            risk,
            x="Segment",
            y="risk_score",
            points="all",
            category_orders={"Segment": SEGMENT_ORDER},
            labels={"risk_score": "Risk score"},
            title="Risk Score Distribution by Segment",
        )
        st.plotly_chart(fig, use_container_width=True)
        b, mime = to_png_bytes(fig)
        st.download_button("Download chart snapshot", b, "risk_chart.png" if mime == "image/png" else "risk_chart.html", mime)


def render_what_if(df_filt: pd.DataFrame) -> None:
    st.subheader("What-If / Investment Lens (Retention)")
    st.caption("Simple, conservative projection for prioritization (not a predictive model).")

    segs = sorted(df_filt["label"].unique().tolist()) if not df_filt.empty else SEGMENT_ORDER
    target = st.selectbox("Target segment", options=segs, index=0 if segs else 0)
    uplift = st.slider("Retention uplift (%)", min_value=0, max_value=20, value=5, step=1)

    lift, note = what_if_lift(df_filt, target, float(uplift))

    c1, c2 = st.columns([1, 2])
    c1.metric("Estimated incremental revenue (12 mo)", fmt_currency(lift))
    c2.info(note)

    base = float(df_filt[df_filt["label"] == target]["PurchaseAmount"].sum()) if not df_filt.empty else 0.0
    fig = go.Figure(
        data=[
            go.Bar(name="Observed revenue (in-window)", x=["Revenue"], y=[base]),
            go.Bar(name=f"Estimated lift (+{uplift}%)", x=["Revenue"], y=[lift]),
        ]
    )
    fig.update_layout(barmode="group", title="Observed vs Estimated Lift", yaxis_title="Revenue")
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    st.plotly_chart(fig, use_container_width=True)
    b, mime = to_png_bytes(fig)
    st.download_button("Download chart snapshot", b, "what_if.png" if mime == "image/png" else "what_if.html", mime)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_click_state()

    st.title(APP_TITLE)
    st.caption("KPIs + interactive visuals for growth, retention, and early warning signals.")

    with st.expander("How to use"):
        st.markdown(
            """
- Use the **sidebar** to filter by date, region, category, channel, segment, and demographics.
- **Click bars/points** in charts to cross-filter segment/category/region (then fine-tune in the sidebar).
- Start at **KPIs**, then use **Trend + Category/Region** to find growth opportunities.
- Use **Segment Health** + **At-Risk Customers** to detect early warning signs.
"""
        )

    # Load data
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Could not load `{DATA_PATH}` from the repo root (same folder as app.py).")
        st.exception(e)
        st.stop()

    # Validation summary at startup
    summary = validate_data(df)
    with st.expander("Dataset validation summary (auto)"):
        st.write({k: summary[k] for k in ["rows", "cols", "unique_customers", "date_min", "date_max"]})
        st.write("Missing values by column:")
        st.dataframe(pd.DataFrame({"missing": summary["missing_by_col"]}).T, use_container_width=True)

    # Sidebar filters (includes click-based default overrides)
    f = sidebar_filters(df)
    df_filt = apply_filters(df, f)

    if df_filt.empty:
        st.warning("No data matches the current filters. Adjust filters (or clear chart selections).")
        st.stop()

    cust = add_customer_features(df_filt)
    kpis = compute_kpis(df_filt, cust)
    render_kpis(kpis)

    # ---------------- Charts + cross-filtering ----------------
    st.subheader("Revenue patterns")
    st.caption("Click a segment point/line, category bar, or region bar to cross-filter.")

    colA, colB = st.columns([2, 1])

    with colA:
        ts = aggregate_timeseries(df_filt, f.agg_level)
        fig_ts = fig_revenue_trend(ts)
        st.plotly_chart(fig_ts, use_container_width=True)
        clicks = plotly_events(fig_ts, click_event=True, hover_event=False, select_event=False, override_height=450)
        if clicks:
            # For go.Scatter, curveNumber maps to trace; take trace name
            cn = clicks[0].get("curveNumber", None)
            if cn is not None and cn < len(fig_ts.data):
                st.session_state["click_segment"] = fig_ts.data[cn].name
                st.toast(f"Selected segment: {st.session_state['click_segment']}", icon="✅")
                st.rerun()

        b, mime = to_png_bytes(fig_ts)
        st.download_button(
            "Download trend snapshot",
            b,
            "revenue_trend.png" if mime == "image/png" else "revenue_trend.html",
            mime,
        )

    with colB:
        fig_cat = fig_rev_by_category(df_filt)
        st.plotly_chart(fig_cat, use_container_width=True)
        clicks = plotly_events(fig_cat, click_event=True, hover_event=False, select_event=False, override_height=450)
        if clicks:
            x = clicks[0].get("x", None)
            if x is not None:
                st.session_state["click_category"] = str(x)
                st.toast(f"Selected category: {st.session_state['click_category']}", icon="✅")
                st.rerun()

        b, mime = to_png_bytes(fig_cat)
        st.download_button(
            "Download category snapshot",
            b,
            "revenue_by_category.png" if mime == "image/png" else "revenue_by_category.html",
            mime,
        )

    fig_reg = fig_rev_by_region(df_filt)
    st.plotly_chart(fig_reg, use_container_width=True)
    clicks = plotly_events(fig_reg, click_event=True, hover_event=False, select_event=False, override_height=420)
    if clicks:
        x = clicks[0].get("x", None)
        if x is not None:
            st.session_state["click_region"] = str(x)
            st.toast(f"Selected region: {st.session_state['click_region']}", icon="✅")
            st.rerun()

    b, mime = to_png_bytes(fig_reg)
    st.download_button(
        "Download region snapshot",
        b,
        "revenue_by_region.png" if mime == "image/png" else "revenue_by_region.html",
        mime,
    )

    # Segment health
    health = segment_health(df_filt)
    render_segment_health(health, pct_customers_declining_mom(df_filt))

    # Top customers
    render_top_customers(cust)

    # Risk
    risk = compute_risk_scores(df_filt)
    render_risk_section(risk)

    # What-if
    render_what_if(df_filt)

    st.caption(
        "Accessibility note: charts include tooltips + labels; use the sidebar and chart selections together. "
        "Avoid relying on color alone—use legends and hover values."
    )


if __name__ == "__main__":
    main()
