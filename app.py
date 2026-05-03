import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest


st.set_page_config(
    page_title="AI Risk Controls Dashboard",
    page_icon="!",
    layout="wide",
)


@st.cache_data
def generate_control_data(days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    business_units = ["Cards", "Retail Banking", "Wealth", "Operations", "Fraud"]
    owners = ["Model Risk", "Operations", "Data Governance", "Compliance", "Product"]
    controls = [
        "Model override review",
        "Data quality threshold",
        "Manual approval sampling",
        "Feature drift check",
        "Customer-impact review",
        "Exception closure SLA",
    ]

    records = []
    for date in dates:
        base_volume = rng.integers(70, 150)
        for _ in range(base_volume):
            unit = rng.choice(business_units, p=[0.24, 0.22, 0.18, 0.2, 0.16])
            control = rng.choice(controls)
            owner = rng.choice(owners)
            severity = rng.choice(["Low", "Medium", "High", "Critical"], p=[0.58, 0.27, 0.11, 0.04])
            exception_probability = {
                "Low": 0.06,
                "Medium": 0.14,
                "High": 0.28,
                "Critical": 0.48,
            }[severity]

            is_exception = rng.random() < exception_probability
            unresolved = bool(is_exception and rng.random() < 0.42)
            age_days = int(rng.integers(1, 45)) if unresolved else 0
            model_score = float(np.clip(rng.normal(0.52, 0.18), 0, 1))
            threshold_breach = bool(model_score > 0.82 or (is_exception and severity in ["High", "Critical"]))

            records.append(
                {
                    "event_date": date,
                    "business_unit": unit,
                    "control_name": control,
                    "owner": owner,
                    "severity": severity,
                    "is_exception": is_exception,
                    "unresolved": unresolved,
                    "age_days": age_days,
                    "model_score": model_score,
                    "threshold_breach": threshold_breach,
                }
            )

    df = pd.DataFrame(records)
    late_window = df["event_date"] >= df["event_date"].max() - pd.Timedelta(days=14)
    stress_units = df["business_unit"].isin(["Cards", "Fraud"])
    stress_rows = late_window & stress_units & (rng.random(len(df)) < 0.16)
    df.loc[stress_rows, "is_exception"] = True
    df.loc[stress_rows, "threshold_breach"] = True
    df.loc[stress_rows, "severity"] = rng.choice(["High", "Critical"], size=stress_rows.sum(), p=[0.72, 0.28])
    df.loc[stress_rows, "unresolved"] = rng.random(stress_rows.sum()) < 0.65
    df.loc[stress_rows & df["unresolved"], "age_days"] = rng.integers(8, 55, size=(stress_rows & df["unresolved"]).sum())
    return df


def add_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["event_date", "business_unit"])
        .agg(
            volume=("control_name", "size"),
            exceptions=("is_exception", "sum"),
            breaches=("threshold_breach", "sum"),
            unresolved=("unresolved", "sum"),
            avg_model_score=("model_score", "mean"),
        )
        .reset_index()
    )
    daily["exception_rate"] = daily["exceptions"] / daily["volume"]

    model = IsolationForest(contamination=0.08, random_state=7)
    features = daily[["volume", "exception_rate", "breaches", "unresolved", "avg_model_score"]]
    daily["anomaly_flag"] = model.fit_predict(features) == -1
    return daily


st.title("AI Risk Controls Monitoring Dashboard")
st.caption("Synthetic financial-services workflow for exception monitoring, control breaches, and executive decision support.")

with st.sidebar:
    st.header("Scenario")
    days = st.slider("History window", 45, 180, 120, step=15)
    seed = st.number_input("Synthetic data seed", min_value=1, max_value=999, value=42)
    exception_target = st.slider("Exception-rate watchlist threshold", 0.05, 0.35, 0.18, step=0.01)
    aging_threshold = st.slider("Aging threshold in days", 7, 45, 21, step=1)

events = generate_control_data(days=days, seed=seed)
daily = add_anomaly_flags(events)

total_events = len(events)
exception_rate = events["is_exception"].mean()
critical_open = int(((events["severity"] == "Critical") & events["unresolved"]).sum())
aged_open = int((events["unresolved"] & (events["age_days"] >= aging_threshold)).sum())
anomaly_count = int(daily["anomaly_flag"].sum())

metric_cols = st.columns(5)
metric_cols[0].metric("Control Events", f"{total_events:,}")
metric_cols[1].metric("Exception Rate", f"{exception_rate:.1%}")
metric_cols[2].metric("Open Critical", f"{critical_open:,}")
metric_cols[3].metric("Aged Open Issues", f"{aged_open:,}")
metric_cols[4].metric("Anomaly Flags", f"{anomaly_count:,}")

recent = events[events["event_date"] >= events["event_date"].max() - pd.Timedelta(days=14)]
prior = events[
    (events["event_date"] < events["event_date"].max() - pd.Timedelta(days=14))
    & (events["event_date"] >= events["event_date"].max() - pd.Timedelta(days=28))
]
recent_rate = recent["is_exception"].mean()
prior_rate = prior["is_exception"].mean() if not prior.empty else 0
rate_delta = recent_rate - prior_rate

watch_units = (
    daily.groupby("business_unit")
    .agg(exception_rate=("exception_rate", "mean"), anomalies=("anomaly_flag", "sum"))
    .query("exception_rate >= @exception_target or anomalies > 0")
    .sort_values(["anomalies", "exception_rate"], ascending=False)
)

st.subheader("Executive Summary")
summary_points = [
    f"Exception rate is {recent_rate:.1%} in the latest 14 days, a {rate_delta:+.1%} change versus the prior 14-day period.",
    f"{critical_open} critical issues remain unresolved and {aged_open} issues are older than {aging_threshold} days.",
    f"{len(watch_units)} business units are on the watchlist based on exception-rate or anomaly criteria.",
]
for point in summary_points:
    st.write(f"- {point}")

if not watch_units.empty:
    lead_unit = watch_units.index[0]
    st.info(
        f"Recommended action: hold an owner review for {lead_unit}, validate whether recent exceptions are process-driven or model-driven, "
        "and agree on closure dates for aged high-severity items."
    )
else:
    st.success("Recommended action: maintain monitoring cadence and review thresholds at the next control forum.")

left, right = st.columns(2)
with left:
    trend = (
        events.groupby(["event_date", "severity"])["is_exception"]
        .sum()
        .reset_index(name="exceptions")
    )
    fig = px.area(trend, x="event_date", y="exceptions", color="severity", title="Exception Trend by Severity")
    st.plotly_chart(fig, use_container_width=True)

with right:
    owner_breakdown = (
        events[events["unresolved"]]
        .groupby(["owner", "severity"])
        .size()
        .reset_index(name="open_issues")
    )
    fig = px.bar(owner_breakdown, x="owner", y="open_issues", color="severity", title="Unresolved Issues by Owner")
    st.plotly_chart(fig, use_container_width=True)

unit_summary = (
    events.groupby("business_unit")
    .agg(
        events=("control_name", "size"),
        exception_rate=("is_exception", "mean"),
        open_issues=("unresolved", "sum"),
        avg_age=("age_days", "mean"),
        breaches=("threshold_breach", "sum"),
    )
    .reset_index()
    .sort_values("exception_rate", ascending=False)
)
unit_summary["exception_rate"] = unit_summary["exception_rate"].map(lambda value: f"{value:.1%}")
unit_summary["avg_age"] = unit_summary["avg_age"].round(1)

st.subheader("Business Unit Control Summary")
st.dataframe(unit_summary, use_container_width=True, hide_index=True)

st.subheader("Daily Anomaly Flags")
flagged = daily[daily["anomaly_flag"]].sort_values("event_date", ascending=False)
st.dataframe(flagged, use_container_width=True, hide_index=True)
