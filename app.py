import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest


st.set_page_config(
    page_title="Operations Exception Analytics",
    page_icon="O",
    layout="wide",
)


@st.cache_data
def generate_operations_data(days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    teams = ["Customer Support", "Fulfillment", "Quality Review", "Intake", "Escalations"]
    owners = ["Queue Ops", "Service Leads", "QA Team", "Vendor Ops", "Platform Ops"]
    stages = [
        "Intake validation",
        "Assignment routing",
        "Quality review",
        "Customer follow-up",
        "Exception closure",
        "Escalation review",
    ]

    records = []
    for date in dates:
        base_volume = rng.integers(70, 150)
        for _ in range(base_volume):
            team = rng.choice(teams, p=[0.24, 0.22, 0.18, 0.2, 0.16])
            stage = rng.choice(stages)
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
            quality_score = float(np.clip(rng.normal(0.52, 0.18), 0, 1))
            sla_exception = bool(quality_score > 0.82 or (is_exception and severity in ["High", "Critical"]))
            escalated = bool(is_exception and rng.random() < {"Low": 0.05, "Medium": 0.14, "High": 0.32, "Critical": 0.58}[severity])

            records.append(
                {
                    "event_date": date,
                    "team": team,
                    "queue_stage": stage,
                    "owner": owner,
                    "severity": severity,
                    "is_exception": is_exception,
                    "unresolved": unresolved,
                    "age_days": age_days,
                    "quality_score": quality_score,
                    "sla_exception": sla_exception,
                    "escalated": escalated,
                }
            )

    df = pd.DataFrame(records)
    late_window = df["event_date"] >= df["event_date"].max() - pd.Timedelta(days=14)
    pressure_teams = df["team"].isin(["Customer Support", "Escalations"])
    pressure_rows = late_window & pressure_teams & (rng.random(len(df)) < 0.16)
    df.loc[pressure_rows, "is_exception"] = True
    df.loc[pressure_rows, "sla_exception"] = True
    df.loc[pressure_rows, "severity"] = rng.choice(["High", "Critical"], size=pressure_rows.sum(), p=[0.72, 0.28])
    df.loc[pressure_rows, "unresolved"] = rng.random(pressure_rows.sum()) < 0.65
    df.loc[pressure_rows, "escalated"] = rng.random(pressure_rows.sum()) < 0.48
    aged_pressure = pressure_rows & df["unresolved"]
    df.loc[aged_pressure, "age_days"] = rng.integers(8, 55, size=aged_pressure.sum())
    return df


def add_bottleneck_flags(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["event_date", "team"])
        .agg(
            volume=("queue_stage", "size"),
            exceptions=("is_exception", "sum"),
            sla_exceptions=("sla_exception", "sum"),
            unresolved=("unresolved", "sum"),
            escalations=("escalated", "sum"),
            avg_quality_score=("quality_score", "mean"),
        )
        .reset_index()
    )
    daily["exception_rate"] = daily["exceptions"] / daily["volume"]
    daily["escalation_rate"] = daily["escalations"] / daily["volume"]

    model = IsolationForest(contamination=0.08, random_state=7)
    features = daily[["volume", "exception_rate", "sla_exceptions", "unresolved", "escalation_rate", "avg_quality_score"]]
    daily["bottleneck_flag"] = model.fit_predict(features) == -1
    return daily


st.title("Operations Exception Analytics Dashboard")
st.caption("Synthetic operations data for workflow quality monitoring, SLA exceptions, queue health, and manager-facing decision support.")

with st.sidebar:
    st.header("Scenario")
    days = st.slider("History window", 45, 180, 120, step=15)
    seed = st.number_input("Synthetic data seed", min_value=1, max_value=999, value=42)
    exception_target = st.slider("Exception-rate watchlist threshold", 0.05, 0.35, 0.18, step=0.01)
    aging_threshold = st.slider("Backlog aging threshold in days", 7, 45, 21, step=1)

events = generate_operations_data(days=days, seed=seed)
daily = add_bottleneck_flags(events)

total_events = len(events)
exception_rate = events["is_exception"].mean()
critical_open = int(((events["severity"] == "Critical") & events["unresolved"]).sum())
aged_open = int((events["unresolved"] & (events["age_days"] >= aging_threshold)).sum())
escalation_rate = events["escalated"].mean()
bottleneck_count = int(daily["bottleneck_flag"].sum())

metric_cols = st.columns(5)
metric_cols[0].metric("Workflow Events", f"{total_events:,}")
metric_cols[1].metric("Exception Rate", f"{exception_rate:.1%}")
metric_cols[2].metric("Open Critical", f"{critical_open:,}")
metric_cols[3].metric("Aged Backlog", f"{aged_open:,}")
metric_cols[4].metric("Bottleneck Flags", f"{bottleneck_count:,}")

recent = events[events["event_date"] >= events["event_date"].max() - pd.Timedelta(days=14)]
prior = events[
    (events["event_date"] < events["event_date"].max() - pd.Timedelta(days=14))
    & (events["event_date"] >= events["event_date"].max() - pd.Timedelta(days=28))
]
recent_rate = recent["is_exception"].mean()
prior_rate = prior["is_exception"].mean() if not prior.empty else 0
rate_delta = recent_rate - prior_rate

watch_teams = (
    daily.groupby("team")
    .agg(exception_rate=("exception_rate", "mean"), bottlenecks=("bottleneck_flag", "sum"), escalation_rate=("escalation_rate", "mean"))
    .query("exception_rate >= @exception_target or bottlenecks > 0")
    .sort_values(["bottlenecks", "exception_rate"], ascending=False)
)

st.subheader("Executive Summary")
summary_points = [
    f"Exception rate is {recent_rate:.1%} in the latest 14 days, a {rate_delta:+.1%} change versus the prior 14-day period.",
    f"{critical_open} critical items remain unresolved and {aged_open} items are older than {aging_threshold} days.",
    f"Escalation rate is {escalation_rate:.1%}, with {len(watch_teams)} teams on the watchlist.",
]
for point in summary_points:
    st.write(f"- {point}")

if not watch_teams.empty:
    lead_team = watch_teams.index[0]
    st.info(
        f"Recommended action: review {lead_team} queue health, confirm whether recent exceptions are volume-driven or process-driven, "
        "and agree on owners for aged high-severity work."
    )
else:
    st.success("Recommended action: maintain monitoring cadence and review queue thresholds at the next operating forum.")

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
        .reset_index(name="open_items")
    )
    fig = px.bar(owner_breakdown, x="owner", y="open_items", color="severity", title="Open Items by Owner")
    st.plotly_chart(fig, use_container_width=True)

team_summary = (
    events.groupby("team")
    .agg(
        events=("queue_stage", "size"),
        exception_rate=("is_exception", "mean"),
        open_items=("unresolved", "sum"),
        avg_age=("age_days", "mean"),
        sla_exceptions=("sla_exception", "sum"),
        escalation_rate=("escalated", "mean"),
    )
    .reset_index()
    .sort_values("exception_rate", ascending=False)
)
team_summary["exception_rate"] = team_summary["exception_rate"].map(lambda value: f"{value:.1%}")
team_summary["escalation_rate"] = team_summary["escalation_rate"].map(lambda value: f"{value:.1%}")
team_summary["avg_age"] = team_summary["avg_age"].round(1)

st.subheader("Team Queue Health Summary")
st.dataframe(team_summary, use_container_width=True, hide_index=True)

st.subheader("Daily Bottleneck Flags")
flagged = daily[daily["bottleneck_flag"]].sort_values("event_date", ascending=False)
st.dataframe(flagged, use_container_width=True, hide_index=True)
