# AI Risk Controls Monitoring Dashboard

A business-facing Streamlit dashboard that monitors operational and model-risk control indicators for a fictional financial-services workflow. The project is designed to show how analytics leaders can turn control data into practical executive decision support.

## Problem

Risk and operations leaders often receive exception data after issues have already escalated. This dashboard simulates a control-monitoring workflow that helps teams see exception trends, unresolved issues, ownership concentration, and emerging anomaly patterns earlier.

## Solution

The app generates synthetic control events, model outputs, business-unit metadata, and owner assignments. It then summarizes KPI movement, flags threshold breaches, highlights aging exceptions, and produces an executive summary with recommended actions.

## Core Features

- Synthetic dataset generator for control events, exceptions, model outputs, and business-unit metadata
- KPI dashboard for exception rate, critical issues, unresolved breaches, aging, and owner breakdown
- Threshold and isolation-forest anomaly detection for unusual control activity
- Severity trends and business-unit views for prioritization
- Executive summary that explains what changed, why it matters, and recommended actions

## Business Value

- Helps leaders identify deteriorating controls before they become audit or operational issues
- Makes risk ownership visible across business units and control owners
- Connects analytics output to management actions instead of stopping at charts
- Demonstrates AI/data product thinking, governance awareness, and practical ML familiarity

## Tech Stack

Python, Streamlit, pandas, NumPy, scikit-learn, Plotly

## How To Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- data/
|   `-- README.md
`-- screenshots/
    `-- README.md
```

## Notes

All data is synthetic and fictional. The dashboard is intended as a portfolio project, not a production risk-management system.
