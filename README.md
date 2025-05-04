# EcomAI Plus: AI-Enhanced Sales Dashboard (v4.0)

## Overview

EcomAI Plus is a Streamlit dashboard designed for E-commerce Sales Managers analyzing performance in the Spanish market (across all 17 Autonomous Communities). It provides actionable insights by integrating **advanced data visualization techniques** based on **cognitive principles** and **bias mitigation strategies**, enhanced with **AI-powered analysis**. The goal is to facilitate clearer understanding and data-driven decision-making.

## Key Features

*   **Context-Aware KPIs:** View core metrics (Net Sales, Orders, Return Rate) with automated comparison deltas vs. the previous period, color-coded to reflect business impact.
*   **Interactive Visualizations:** Explore trends with moving averages, performance vs. targets, regional comparisons (user-selectable), category breakdowns (including return rates), sales heatmaps, and sales distribution by quantity using Plotly.
*   **AI Analysis:** Generate on-demand summaries and actionable suggestions based on currently filtered data using the OpenAI API.
*   **Data Flexibility:** Analyze built-in synthetic data or upload your own CSV/Excel sales data (requires specific columns).
*   **Downloadable Summary:** Export key KPIs and the AI analysis as a Markdown executive report.

## Setup & Usage

1.  **Dependencies:** Install required libraries:
    ```bash
    pip install streamlit pandas plotly numpy openai python-dateutil openpyxl
    ```
2.  **API Key:** **IMPORTANT:** Replace the placeholder OpenAI API key string within the `app.py` script with your actual key. (Best practice: Use Streamlit Secrets or environment variables).
3.  **Run:** Navigate to the project directory in your terminal and run:
    ```bash
    streamlit run app.py
    ```
4.  **Interact:** Use the sidebar filters to select data sources, date ranges, regions, and categories. Click the "Generate AI Summary" button for insights.
