import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from openai import OpenAI
import os

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="EcomAI Plus Dashboard",
    page_icon="✅"
)

# --- Initialize Session State ---
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None

# --- Data Generation Function ---
@st.cache_data
def generate_spanish_data_v3(num_rows=2000):
    """Generates synthetic e-commerce sales data focused on ALL Spain regions (from 2024-2025)."""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1); end_date = datetime(2025, 12, 31)
    date_range_delta = (end_date - start_date).days
    dates = [start_date + timedelta(days=np.random.randint(0, date_range_delta + 1)) for _ in range(num_rows)]
    regions_spain = ['Andalucía', 'Aragón', 'Principado de Asturias', 'Illes Balears', 'Canarias', 'Cantabria', 'Castilla y León', 'Castilla-La Mancha', 'Cataluña', 'Comunitat Valenciana', 'Extremadura', 'Galicia', 'La Rioja', 'Comunidad de Madrid', 'Región de Murcia', 'Comunidad Foral de Navarra', 'País Vasco']
    regions_prob = [0.18, 0.03, 0.025, 0.03, 0.05, 0.015, 0.06, 0.05, 0.17, 0.11, 0.025, 0.07, 0.01, 0.15, 0.035, 0.015, 0.05]
    regions_prob = np.array(regions_prob); regions_prob /= regions_prob.sum()
    categories = ['Electronics', 'Clothing', 'Home Goods', 'Groceries', 'Toys', 'Books & Media']
    categories_prob = [0.28, 0.24, 0.18, 0.13, 0.09, 0.08]
    managers_spain = ['Sofía García', 'Mateo Fernández', 'Lucía Martínez', 'Hugo López']
    data = {'OrderID': range(1, num_rows + 1), 'OrderDate': dates, 'Region': np.random.choice(regions_spain, num_rows, p=regions_prob), 'Category': np.random.choice(categories, num_rows, p=categories_prob), 'SalesAmount': np.random.lognormal(mean=4, sigma=0.8, size=num_rows).round(2), 'Quantity': np.random.randint(1, 6, num_rows), 'RegionalManager': np.random.choice(managers_spain, num_rows)}
    df = pd.DataFrame(data)
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['Returned'] = False
    return_rate_clothing = 0.19; return_rate_electronics = 0.13; return_rate_other = 0.06
    for index, row in df.iterrows():
        rate = return_rate_other
        if row['Category'] == 'Clothing': rate = return_rate_clothing
        elif row['Category'] == 'Electronics': rate = return_rate_electronics
        if np.random.rand() < rate: df.loc[index, 'Returned'] = True
    df['Month'] = df['OrderDate'].dt.month
    category_target_multiplier = {'Electronics': 1.1, 'Clothing': 0.9, 'Home Goods': 1.0, 'Groceries': 0.8, 'Toys': 0.95, 'Books & Media': 0.85}
    month_target_multiplier = {m: 1 + 0.1 * np.sin((m - 1) * np.pi / 6) for m in range(1, 13)}
    df['SalesTarget'] = df.apply(lambda row: (row['SalesAmount'] / (month_target_multiplier[row['Month']] * category_target_multiplier[row['Category']])) * np.random.uniform(0.88, 1.12), axis=1).round(2)
    df['YearMonth'] = df['OrderDate'].dt.to_period('M').astype(str)
    df['Year'] = df['OrderDate'].dt.year
    df['MonthName'] = df['OrderDate'].dt.strftime('%B')
    df['DayOfWeek'] = df['OrderDate'].dt.day_name()
    df['NetSalesAmount'] = df.apply(lambda row: 0 if row['Returned'] else row['SalesAmount'], axis=1)
    return df

# --- Load Data ---
df_original = generate_spanish_data_v3()

# --- Color Palette Definition ---
category_color_map = {cat: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, cat in enumerate(sorted(df_original['Category'].unique()))}
region_color_map = {reg: px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)] for i, reg in enumerate(sorted(df_original['Region'].unique()))}
all_categories = sorted(df_original['Category'].unique())
all_regions_spain = sorted(df_original['Region'].unique())

# --- Dashboard Title & Description ---
st.title("✅ EcomAI Plus: AI-Enhanced Sales Dashboard")
st.markdown(""" Welcome to EcomAI Plus! Analyze Spanish e-commerce performance across all 17 Autonomous Communities. Leveraging cognitive principles and AI for insights. Generate an **AI summary** or **download an executive report** below. """)
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("📊 Filters")
min_date_data = df_original['OrderDate'].min().date(); max_date_data = df_original['OrderDate'].max().date()
default_start = min_date_data; default_end = max_date_data
date_range_selection = st.sidebar.date_input("Select Date Range", [default_start, default_end], min_value=min_date_data, max_value=max_date_data)
if len(date_range_selection) == 2: start_date_filter, end_date_filter = date_range_selection
else: st.sidebar.warning("Please select both start and end dates."); start_date_filter = default_start; end_date_filter = default_end
selected_regions_global = st.sidebar.multiselect("Select Region(s) (Global Filter)", all_regions_spain, default=all_regions_spain)
selected_categories_global = st.sidebar.multiselect("Select Category(s) (Global Filter)", all_categories, default=all_categories)

# --- Filter Dataframe (Global Filters) ---
try: start_datetime = pd.to_datetime(start_date_filter); end_datetime = pd.to_datetime(end_date_filter)
except ValueError: st.error("Invalid date selected."); st.stop()
df_filtered = df_original[(df_original['OrderDate'] >= start_datetime) & (df_original['OrderDate'] <= end_datetime) & (df_original['Region'].isin(selected_regions_global)) & (df_original['Category'].isin(selected_categories_global))].copy()
if df_filtered.empty: st.warning("No data available for selected filters."); st.stop()
st.sidebar.metric("Filtered Orders (Global)", f"{df_filtered.shape[0]:,}")
st.sidebar.caption(f"Data from {start_date_filter} to {end_date_filter}.")

# --- Previous Period Calculation ---
duration = end_datetime - start_datetime;
if duration.days < 0: duration = timedelta(days=0)
prev_end_datetime = start_datetime - timedelta(days=1); prev_start_datetime = prev_end_datetime - duration
df_previous = df_original[(df_original['OrderDate'] >= prev_start_datetime) & (df_original['OrderDate'] <= prev_end_datetime) & (df_original['Region'].isin(selected_regions_global)) & (df_original['Category'].isin(selected_categories_global))].copy()

# --- Sidebar Justification ---
with st.sidebar.expander("ⓘ Sidebar Design Justifications"): st.caption(""" *   **Cognitive Load / Spatial Memory:** Filters consistently located. *   **Recency/Availability Bias Mitigation:** Max date range default. *   **Decision Hierarchy (Awareness):** Filters allow drilling down. *   **User Control:** Enables segment focus. Handles date range selection. """)

# --- Main Dashboard Area ---

# == Level 1: Overview KPIs ==
st.header("📌 Key Performance Indicators (KPIs)")
kpicol1, kpicol2, kpicol3 = st.columns(3)
# KPI Calculation
total_net_sales = df_filtered['NetSalesAmount'].sum(); total_orders = df_filtered['OrderID'].nunique(); total_returned_orders = df_filtered[df_filtered['Returned']]['OrderID'].nunique(); return_rate = (total_returned_orders / total_orders) * 100 if total_orders > 0 else 0
prev_total_net_sales = df_previous['NetSalesAmount'].sum() if not df_previous.empty else 0; prev_total_orders = df_previous['OrderID'].nunique() if not df_previous.empty else 0; prev_total_returned_orders = df_previous[df_previous['Returned']]['OrderID'].nunique() if not df_previous.empty else 0; prev_return_rate = (prev_total_returned_orders / prev_total_orders) * 100 if prev_total_orders > 0 else 0

# KPI Delta Calculation Logic
delta_sales = 0  # Initialize delta value
delta_sales_val = "N/A Prev."
delta_sales_color = "off"
prev_sales_help = "N/A"
can_compare_sales = not df_previous.empty or prev_total_net_sales > 0
if can_compare_sales:
    prev_sales_help = f"€{prev_total_net_sales:,.2f}"
    if prev_total_net_sales > 0:
        delta_sales = ((total_net_sales - prev_total_net_sales) / prev_total_net_sales) * 100  # Calculate delta
        delta_sales_val = f"{delta_sales:.1f}%"
        delta_sales_color = "normal"  # Always "normal"
    elif total_net_sales > 0:
        delta_sales = float('inf')  # Represent infinite increase numerically
        delta_sales_val = "∞%"
        delta_sales_color = "normal"  # Treat increase from zero as positive
elif total_net_sales > 0:
    # If no previous data but current data exists
    delta_sales = float('nan')  # Represent change is not applicable
    delta_sales_val = "(New Period)"
    delta_sales_color = "off"
else:
    # If no current data
    delta_sales = float('nan')

delta_orders = 0  # Initialize delta value
delta_orders_val = "N/A Prev."
delta_orders_color = "off"
prev_orders_help = "N/A"
can_compare_orders = not df_previous.empty or prev_total_orders > 0
if can_compare_orders:
    prev_orders_help = f"{prev_total_orders:,}"
    if prev_total_orders > 0:
        delta_orders = ((total_orders - prev_total_orders) / prev_total_orders) * 100  # Calculate delta
        delta_orders_val = f"{delta_orders:.1f}%"
        delta_orders_color = "normal"  # Always "normal"
    elif total_orders > 0:
        delta_orders = float('inf')
        delta_orders_val = "∞%"
        delta_orders_color = "normal"  # Treat increase from zero as positive
elif total_orders > 0:
    delta_orders = float('nan')
    delta_orders_val = "(New Period)"
    delta_orders_color = "off"
else:
    delta_orders = float('nan')

delta_return_rate = 0  # Initialize delta value
delta_return_rate_val = "N/A"
delta_return_rate_color = "off"
prev_rate_help = "N/A"
current_rate_str = f"{return_rate:.1f}%" if total_orders > 0 else "N/A"
can_calculate_prev_rate = prev_total_orders > 0
if total_orders > 0:
    if can_calculate_prev_rate:
        prev_rate_help = f"{prev_return_rate:.1f}%"
        delta_return_rate = return_rate - prev_return_rate  # Calculate delta
        delta_return_rate_val = f"{delta_return_rate:+.1f} pts"
        delta_return_rate_color = "normal"  # Always "normal"
    else:
        # Current data exists, previous doesn't
        delta_return_rate = float('nan')
        delta_return_rate_val = "(vs N/A)"
        delta_return_rate_color = "off"
elif can_calculate_prev_rate:
    # Previous data exists, current doesn't
    prev_rate_help = f"{prev_return_rate:.1f}%"
    delta_return_rate = float('nan')
    delta_return_rate_val = "(Current N/A)"
    delta_return_rate_color = "off"
else:
    # Neither exists
    delta_return_rate = float('nan')

# Display KPIs
kpicol1.metric("Total Net Sales", f"€{total_net_sales:,.2f}", delta=delta_sales_val if delta_sales_color != "off" else None, delta_color=delta_sales_color, help=f"vs Previous Period ({prev_sales_help})")
kpicol2.metric("Total Orders", f"{total_orders:,}", delta=delta_orders_val if delta_orders_color != "off" else None, delta_color=delta_orders_color, help=f"vs Previous Period ({prev_orders_help})")
kpicol3.metric("Return Rate", current_rate_str, delta=delta_return_rate_val if delta_return_rate_color != "off" else None, delta_color=delta_return_rate_color, help=f"{delta_return_rate_val if delta_return_rate_color == 'off' and delta_return_rate_val not in ['N/A', 'N/A Prev.'] else ''} vs Previous Period ({prev_rate_help})")

with st.expander("ⓘ KPI DESIGN Justifications"): st.caption(""" *   **Visual Hierarchy & Chunking:** KPIs prominent, grouped. *   **Comparative Analysis / Contextual Reference Points:** Change vs previous period shown with robust color logic. *   **Anchoring Bias Mitigation:** Comparison reduces reliance on current numbers. *   **Confirmation Bias / Survivorship Bias Mitigation:** Explicit *Return Rate* shown. *   **Cognitive Offloading:** Changes calculated, handling edge cases. """)
st.markdown("---")

# --- AI Analysis Section ---
st.subheader("🤖 AI Performance Analysis")

def format_data_for_ai(df_current, df_prev, kpi_data, context_filters):
    """Prepares a text summary of the data for the AI prompt."""
    summary = f"Analysis Period: {context_filters['start_date']} to {context_filters['end_date']}\n"
    regions_filtered_str = f"{len(context_filters['regions'])}/{len(all_regions_spain)} regions selected" if len(context_filters['regions']) != len(all_regions_spain) else 'All'
    categories_filtered_str = f"{len(context_filters['categories'])}/{len(all_categories)} categories selected" if len(context_filters['categories']) != len(all_categories) else 'All'
    summary += f"Filters Applied: {regions_filtered_str}, {categories_filtered_str}\n\n"

    summary += "**Key Performance Indicators (KPIs) compared to previous period:**\n"
    # Pass the final display strings for deltas to the formatter
    summary += f"- Total Net Sales: €{kpi_data['net_sales']:,.2f} ({kpi_data['delta_sales_display']})\n"
    summary += f"- Total Orders: {kpi_data['orders']:,} ({kpi_data['delta_orders_display']})\n"
    summary += f"- Return Rate: {kpi_data['return_rate']} ({kpi_data['delta_return_rate_display']})\n\n"

    try:
        sales_trend_data = df_current.groupby('YearMonth')['NetSalesAmount'].sum().rolling(window=3, min_periods=1).mean()
        if len(sales_trend_data) >= 2:
            trend_start = sales_trend_data.iloc[0]
            trend_end = sales_trend_data.iloc[-1]
            # Determine trend direction
            if trend_end > trend_start:
                trend_direction = "upward"
            elif trend_end < trend_start:
                 trend_direction = "downward"
            else:
                 trend_direction = "flat"
            # Add to summary
            summary += f"**Overall Sales Trend (3-Month Avg):** Generally {trend_direction}\n\n"
        else:
            summary += "**Overall Sales Trend (3-Month Avg):** Not enough data points for trend direction.\n\n"
    except Exception:
        summary += "**Overall Sales Trend (3-Month Avg):** Error calculating trend.\n\n"

    # Performance vs Target Summary
    try:
        perf_target = df_current.groupby('YearMonth').agg(Actual=('NetSalesAmount', 'sum'), Target=('SalesTarget', 'sum'))
        perf_target['Perf%'] = perf_target.apply(lambda r: (r['Actual']/r['Target'])*100 if r['Target']>0 else 0, axis=1)
        months_met_target = (perf_target['Perf%'] >= 100).sum()
        total_months = len(perf_target)
        if total_months > 0:
            success_rate = months_met_target / total_months
    
            summary += f"**Performance vs Target:** Target met or exceeded in {months_met_target} out of {total_months} months ({success_rate:.0%} success rate).\n\n"
        else:
            summary += f"**Performance vs Target:** No monthly data to calculate.\n\n"
    except Exception:
         summary += f"**Performance vs Target:** Could not calculate.\n\n"

    # Top/Bottom Performers
    try:
        # Regional Performance
        region_perf = df_current.groupby('Region')['NetSalesAmount'].sum().sort_values(ascending=False)
        if len(region_perf) > 1:
            top_region = region_perf.index[0]
            top_sales = region_perf.iloc[0]
            bottom_region = region_perf.index[-1]
            bottom_sales = region_perf.iloc[-1]
         
            summary += f"**Regional Performance:** Top region: {top_region} (€{top_sales:,.0f}). Bottom: {bottom_region} (€{bottom_sales:,.0f}).\n"
        elif len(region_perf) == 1:
         
            summary += f"**Regional Performance:** Only region: {region_perf.index[0]} (€{region_perf.iloc[0]:,.0f}).\n"

        # Category Performance
        category_perf = df_current.groupby('Category')['NetSalesAmount'].sum().sort_values(ascending=False)
        if len(category_perf) > 1:
            top_cat = category_perf.index[0]
            top_cat_sales = category_perf.iloc[0]
            bottom_cat = category_perf.index[-1]
            bottom_cat_sales = category_perf.iloc[-1]

            summary += f"**Category Performance:** Top category: {top_cat} (€{top_cat_sales:,.0f}). Bottom: {bottom_cat} (€{bottom_cat_sales:,.0f}).\n"
        elif len(category_perf) == 1:

            summary += f"**Category Performance:** Only category: {category_perf.index[0]} (€{category_perf.iloc[0]:,.0f}).\n"

        # Return Rates
        category_returns = df_current.groupby('Category').agg(TotalOrders=('OrderID', 'nunique'), ReturnedOrders=('Returned', lambda x: x[x==True].count())).reset_index()
        if 'TotalOrders' in category_returns.columns and not category_returns.empty:
            category_returns['ReturnRate'] = category_returns.apply(lambda row: (row['ReturnedOrders'] / row['TotalOrders']) * 100 if row['TotalOrders'] > 0 else 0, axis=1)
            category_returns = category_returns.sort_values('ReturnRate', ascending=False)
            if not category_returns.empty:
                 if len(category_returns) > 1:
                     highest_cat = category_returns.iloc[0]['Category']
                     highest_rate = category_returns.iloc[0]['ReturnRate']
                     lowest_cat = category_returns.iloc[-1]['Category']
                     lowest_rate = category_returns.iloc[-1]['ReturnRate']
         
                     summary += f"**Return Rates:** Highest category: {highest_cat} ({highest_rate:.1f}%). Lowest: {lowest_cat} ({lowest_rate:.1f}%).\n"
                 elif len(category_returns) == 1:
           
                     summary += f"**Return Rates:** Only category: {category_returns.iloc[0]['Category']} ({category_returns.iloc[0]['ReturnRate']:.1f}%).\n"
        else:
             summary += "**Return Rates:** Could not calculate summary.\n"

    except Exception as e:
        summary += f"**Top/Bottom Performers:** Error calculating summaries ({e}).\n"

    return summary

def get_ai_analysis(data_summary): # Remove api_key_param
    """Calls OpenAI API to get analysis and suggestions."""
    # Check if the secret exists and is not empty
    if "openai_api_key" not in st.secrets or not st.secrets["openai_api_key"]:
         return "Error: OpenAI API Key not configured in Streamlit Secrets."
    try:
        # Use the key directly from st.secrets
        client = OpenAI(api_key=st.secrets["openai_api_key"])
        prompt = f"""You are an expert E-commerce Analyst reviewing a sales performance report for a business operating in Spain. Based *only* on the following data summary, please provide: 1. A concise **Business Summary** highlighting the most important trends, key performance indicators, and significant findings (like top/bottom performers or notable return rates). 2. 2-3 brief, actionable, and *general* **Suggestions** for potential areas of focus or investigation based *strictly* on the provided data summary. Focus on what the data suggests might need attention. Keep the tone professional, data-driven, and easily understandable for a manager. Do not invent information or metrics not present in the summary below. Use bullet points for the summary and suggestions. **Data Summary:**\n---\n{data_summary}\n---"""
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are an expert E-commerce Analyst."}, {"role": "user", "content": prompt}], temperature=0.5, max_tokens=350)
        return response.choices[0].message.content
    except Exception as e:
        # Add more specific error handling if possible
        st.error(f"Error communicating with OpenAI: {e}")
        return "Error: Could not generate AI analysis."

# Button logic
if st.button("💡 Generate AI Summary & Suggestions", key="generate_ai"):
    # Check if the secret exists via st.secrets
    if "openai_api_key" not in st.secrets or not st.secrets["openai_api_key"]:
        st.error("OpenAI API Key is not configured. Please add it to your Streamlit Cloud app's secrets.")
    else:
        with st.spinner("🧠 AI is analyzing the data... Please wait."):
            # Pass the final display strings for deltas to the formatter
            kpi_summary_data = { "net_sales": total_net_sales, "delta_sales_display": delta_sales_val, "orders": total_orders, "delta_orders_display": delta_orders_val, "return_rate": current_rate_str, "delta_return_rate_display": delta_return_rate_val }
            filter_context = { "start_date": start_date_filter, "end_date": end_date_filter, "regions": selected_regions_global, "categories": selected_categories_global }
            ai_data_input = format_data_for_ai(df_filtered, df_previous, kpi_summary_data, filter_context)
            # Call get_ai_analysis without passing the key
            ai_response = get_ai_analysis(ai_data_input)
            st.session_state.ai_response = ai_response
            if "Error:" not in ai_response: # Check if AI analysis was successful
                 st.success("**AI Analysis Complete!**")
            st.markdown(st.session_state.ai_response) # Display response or error
elif st.session_state.ai_response:
     st.markdown("**Last Generated AI Analysis:**")
     st.markdown(st.session_state.ai_response)

# == Level 2: Trends & Performance vs. Target ==
st.header("📈 Performance Trends & Target Analysis"); trendcol1, trendcol2 = st.columns([2, 1]);
with trendcol1: st.subheader("Net Sales Over Time (with Trend)"); sales_over_time = df_filtered.groupby('YearMonth')['NetSalesAmount'].sum().reset_index().sort_values('YearMonth'); sales_over_time['YearMonthDT'] = pd.to_datetime(sales_over_time['YearMonth'] + '-01'); window_size = 3; sales_over_time['MovingAvg'] = sales_over_time['NetSalesAmount'].rolling(window=window_size, min_periods=1).mean(); fig_trend = go.Figure(); fig_trend.add_trace(go.Scatter(x=sales_over_time['YearMonthDT'], y=sales_over_time['NetSalesAmount'], mode='lines+markers', name='Monthly Net Sales', hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Net Sales:</b> €%{y:,.2f}<extra></extra>')); fig_trend.add_trace(go.Scatter(x=sales_over_time['YearMonthDT'], y=sales_over_time['MovingAvg'], mode='lines', name=f'{window_size}-Month Moving Avg', line=dict(dash='dash', width=1.5), hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Moving Avg:</b> €%{y:,.2f}<extra></extra>')); fig_trend.update_layout(title="Monthly Net Sales Trend (2024-2025)", hovermode="x unified", height=400, yaxis_tickprefix="€", xaxis_title="Month", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)); st.plotly_chart(fig_trend, use_container_width=True); st.caption(f"Dashed line shows the {window_size}-month moving average trend.")
with trendcol2: st.subheader("Performance vs. Target"); perf_vs_target = df_filtered.groupby('YearMonth').agg(ActualSales=('NetSalesAmount', 'sum'), TargetSales=('SalesTarget', 'sum')).reset_index().sort_values('YearMonth'); perf_vs_target['Performance'] = perf_vs_target.apply(lambda row: (row['ActualSales'] / row['TargetSales']) * 100 if row['TargetSales'] > 0 else 0, axis=1); perf_vs_target['MonthLabel'] = pd.to_datetime(perf_vs_target['YearMonth'] + '-01').dt.strftime('%b %Y'); perf_vs_target['Color'] = np.where(perf_vs_target['Performance'] >= 100, 'forestgreen', 'firebrick'); fig_target = go.Figure(data=[go.Bar(x=perf_vs_target['MonthLabel'], y=perf_vs_target['Performance'], marker_color=perf_vs_target['Color'], name='Performance vs Target (%)', hovertemplate='<b>Month:</b> %{x}<br><b>Performance:</b> %{y:.1f}%<extra></extra>')]); fig_target.add_hline(y=100, line_dash="dash", line_color="grey", annotation_text="Target (100%)", annotation_position="bottom right"); fig_target.update_layout(title="Monthly Sales Performance vs. Target (%)", xaxis_title="Month", yaxis_title="Performance (%)", height=400, yaxis_ticksuffix="%"); st.plotly_chart(fig_target, use_container_width=True); st.caption("Green bars indicate performance >= 100% of target, Red bars indicate < 100%.")
with st.expander("ⓘ Trends & Target Justifications"): st.caption(""" *   **Pattern Recognition / Chart Appropriateness:** Line chart for trends, Bar for target comparison. *   **Recency/Availability Bias Mitigation:** Moving average smooths noise. *   **Preattentive Processing:** Color clearly flags target performance. *   **Contextual Reference Points:** Target line included. *   **Focus + Context:** Interactive tooltips. """)
st.markdown("---")

# == Level 3: Detailed Analysis ==
st.header("🔍 Detailed Analysis by Dimension"); tab_region_trends, tab_category, tab_heatmap, tab_sales_qty = st.tabs(["🌍 REGIONAL TRENDS", "🏷️ BY CATEGORY", "🔥 HEATMAP ANALYSIS", "📊 SALES VS QUANTITY"]);
with tab_region_trends:
    st.subheader("Regional Sales Trend Comparison (Select Regions)")
    available_regions = sorted(df_filtered['Region'].unique())
    if available_regions:
        default_trend_regions = available_regions[:min(4, len(available_regions))]
        selected_trend_regions = st.multiselect("Select regions to compare trends:",options=available_regions,default=default_trend_regions)
        if selected_trend_regions:
            region_monthly_sales = df_filtered[df_filtered['Region'].isin(selected_trend_regions)].groupby(['Region', 'YearMonth'])['NetSalesAmount'].sum().reset_index().sort_values(['Region', 'YearMonth']);
            region_monthly_sales['YearMonthDT'] = pd.to_datetime(region_monthly_sales['YearMonth'] + '-01');
            if not region_monthly_sales.empty:
                fig_region_trends_selected = px.line(region_monthly_sales, x='YearMonthDT', y='NetSalesAmount', color='Region', title='Monthly Net Sales Trend for Selected Regions', height=450, labels={'YearMonthDT': 'Month', 'NetSalesAmount': 'Net Sales (€)', 'Region': 'Region'}, markers=True, color_discrete_map=region_color_map);
                fig_region_trends_selected.update_layout(hovermode="x unified", yaxis_tickprefix="€", legend=dict(title="Regions", orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5));
                st.plotly_chart(fig_region_trends_selected, use_container_width=True);
            else: st.info("No monthly data for selected region(s).");
        else: st.warning("Please select region(s).");
    else: st.info("No regions available with current filters.");
    st.subheader("Regional Performance Summary");
    region_perf = df_filtered.groupby('Region').agg(TotalNetSales=('NetSalesAmount', 'sum'), TotalOrders=('OrderID', 'nunique'), ReturnedOrders=('Returned', lambda x: x[x==True].count())).reset_index();
    region_perf['ReturnRate'] = region_perf.apply(lambda row: (row['ReturnedOrders'] / row['TotalOrders']) * 100 if row['TotalOrders'] > 0 else 0, axis=1);
    region_perf['AvgOrderValue'] = region_perf.apply(lambda row: row['TotalNetSales'] / row['TotalOrders'] if row['TotalOrders'] > 0 else 0, axis=1);
    st.dataframe(region_perf.sort_values('TotalNetSales', ascending=False).style.format({'TotalNetSales': '€{:,.2f}', 'TotalOrders': '{:,}', 'ReturnedOrders': '{:,}', 'ReturnRate': '{:.1f}%', 'AvgOrderValue': '€{:,.2f}'}), use_container_width=True)
with tab_category:
    st.subheader("Category Performance Comparison")
    category_perf = df_filtered.groupby('Category').agg(TotalNetSales=('NetSalesAmount', 'sum'), TotalOrders=('OrderID', 'nunique'), ReturnedOrders=('Returned', lambda x: x[x==True].count())).reset_index()
    category_perf['ReturnRate'] = category_perf.apply(lambda row: (row['ReturnedOrders'] / row['TotalOrders']) * 100 if row['TotalOrders'] > 0 else 0, axis=1)
    category_perf['AvgOrderValue'] = category_perf.apply(lambda row: row['TotalNetSales'] / row['TotalOrders'] if row['TotalOrders'] > 0 else 0, axis=1)
    category_perf = category_perf.sort_values('TotalNetSales', ascending=False)
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        fig_cat_sales = px.bar(category_perf, x='Category', y='TotalNetSales', title='Total Net Sales by Category', labels={'Category': 'Category', 'TotalNetSales': 'Total Net Sales (€)'}, color='Category', color_discrete_map=category_color_map, text_auto='.2s')
        fig_cat_sales.update_layout(height=350, showlegend=False, yaxis_tickprefix="€")
        st.plotly_chart(fig_cat_sales, use_container_width=True)
    with cat_col2:
        fig_cat_return = px.bar(category_perf.sort_values('ReturnRate', ascending=False), x='Category', y='ReturnRate', title='Return Rate by Category', labels={'Category': 'Category', 'ReturnRate': 'Return Rate (%)'}, color='ReturnRate', color_continuous_scale=px.colors.sequential.Reds, text_auto='.1f')
        fig_cat_return.update_layout(height=350, coloraxis_showscale=False)
        fig_cat_return.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_cat_return, use_container_width=True)
    st.dataframe(category_perf.style.format({ 'TotalNetSales': '€{:,.2f}', 'TotalOrders': '{:,}', 'ReturnedOrders': '{:,}', 'ReturnRate': '{:.1f}%', 'AvgOrderValue': '€{:,.2f}' }), use_container_width=True)
with tab_heatmap:
    st.subheader("Performance Heatmap: Net Sales by Region & Category")
    if not df_filtered.empty and 'Region' in df_filtered.columns and 'Category' in df_filtered.columns:
        heatmap_data = df_filtered.pivot_table(index='Region', columns='Category', values='NetSalesAmount', aggfunc='sum', fill_value=0);
        if not heatmap_data.empty:
            fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Category", y="Region", color="Net Sales (€)"), x=heatmap_data.columns, y=heatmap_data.index, aspect="auto", color_continuous_scale=px.colors.sequential.Viridis, title="Net Sales Heatmap: Region vs. Category", text_auto='.2s')
            fig_heatmap.update_xaxes(side="bottom")
            fig_heatmap.update_layout(height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.caption("Color intensity and value indicate total net sales.")
        else:
            st.info("Could not generate heatmap data.")
    else:
        st.info("Not enough data diversity for heatmap.")
with tab_sales_qty:
    st.subheader("Order Analysis: Sales Amount Distribution by Quantity")
    fig_box = px.box(df_filtered, x='Quantity', y='SalesAmount', color='Category', title='Distribution of Gross Sales Amount per Order Quantity', labels={'Quantity': 'Quantity per Order', 'SalesAmount': 'Gross Sales Amount (€)', 'Category': 'Category'}, points=False, color_discrete_map=category_color_map)
    fig_box.update_layout(height=500, yaxis_tickprefix="€", xaxis={'type': 'category'})
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Shows distribution of Gross Sales Amount per Order Quantity, by category.")
with st.expander("ⓘ Detailed Analysis Design Justifications"): st.caption(""" *   **Progressive Disclosure / Cognitive Load:** Tabs separate detailed views. Regional trend view uses selection. *   **Comparative Analysis / Pattern Recognition:** Selectable line chart for regional trends. Heatmap for matrix overview. Bar charts for category totals. Box plot visualizes sales distribution per quantity. *   **Chart Appropriateness:** Box plot chosen for visualizing distribution against a discrete variable (Quantity). *   **Preattentive Processing:** Color distinguishes elements consistently. *   **Survivorship/Confirmation Bias Mitigation:** Return Rate by Category chart shown. """)
st.markdown("---")

# --- Download Executive Summary Section ---
# (Code remains the same)
st.header("📋 Executive Summary Download")
def generate_markdown_summary(kpi_data, context_filters, ai_summary):
    """Generates a Markdown formatted summary string."""
    report = f"# EcomAI Plus: Executive Sales Summary\n\n"; report += f"**Report Period:** {context_filters['start_date']} to {context_filters['end_date']}\n"; regions_filtered_str = f"{len(context_filters['regions'])}/{len(all_regions_spain)} regions selected" if len(context_filters['regions']) != len(all_regions_spain) else 'All'; categories_filtered_str = f"{len(context_filters['categories'])}/{len(all_categories)} categories selected" if len(context_filters['categories']) != len(all_categories) else 'All'; report += f"**Filters Applied:** Regions: {regions_filtered_str}, Categories: {categories_filtered_str}\n\n"; report += "---\n\n"; report += "## Key Performance Indicators (KPIs)\n"; report += f"*   **Total Net Sales:** €{kpi_data['net_sales']:,.2f} ({kpi_data['delta_sales_display']} vs Previous Period)\n"; report += f"*   **Total Orders:** {kpi_data['orders']:,} ({kpi_data['delta_orders_display']} vs Previous Period)\n"; report += f"*   **Return Rate:** {kpi_data['return_rate']} ({kpi_data['delta_return_rate_display']} vs Previous Period)\n\n"; report += "---\n\n"; report += "## AI-Generated Analysis & Suggestions\n\n"; report += f"{ai_summary}\n" if ai_summary else "*AI analysis has not been generated for the current filter selection.*\n"; report += "\n---\n\n"; report += f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by EcomAI Plus Dashboard.*"
    return report
kpi_summary_data_dl = { "net_sales": total_net_sales, "delta_sales_display": delta_sales_val, "orders": total_orders, "delta_orders_display": delta_orders_val, "return_rate": current_rate_str, "delta_return_rate_display": delta_return_rate_val }
filter_context_dl = { "start_date": start_date_filter, "end_date": end_date_filter, "regions": selected_regions_global, "categories": selected_categories_global }
ai_summary_for_dl = st.session_state.get('ai_response', None)
markdown_content = generate_markdown_summary(kpi_summary_data_dl, filter_context_dl, ai_summary_for_dl)
st.download_button(label="📥 Download Executive Summary (.md)", data=markdown_content.encode('utf-8'), file_name=f"EcomAI_Plus_Summary_{start_date_filter}_to_{end_date_filter}.md", mime="text/markdown")
st.caption("Downloads a Markdown file summarizing the KPIs and AI analysis based on current filters.")
with st.expander("ⓘ Download Justifications"): st.caption(""" *   **Cognitive Offloading:** Provides a shareable summary. *   **Accessibility:** Offers key insights in simple text format. *   **Session State:** Includes latest AI analysis. Positioned after detailed analysis for logical flow. """)
st.markdown("---")

# == Level 4: Raw Data Exploration ==
st.header("💾 Raw Data Explorer")
with st.expander("View Filtered Raw Data (2024-2025)"):
    st.dataframe(df_filtered.style.format({'SalesAmount': '€{:,.2f}', 'SalesTarget': '€{:,.2f}', 'NetSalesAmount': '€{:,.2f}'}), use_container_width=True)
    @st.cache_data
    def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
    csv = convert_df_to_csv(df_filtered)
    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name=f'filtered_sales_data_spain_2024_2025_{start_date_filter}_to_{end_date_filter}.csv', mime='text/csv')
    st.caption("Allows users to perform their own analysis or verify calculations.")

# --- Final Footer ---
st.markdown("---")
st.caption("EcomAI Plus Enhanced Dashboard | Data Visualization and Decision-Making Subject | © 2025")

