"""
Explainability Dashboard Generator
===================================
Reads CSV files output by `Mask_PPO_explainability_test.py` and
generates a standalone, interactive HTML dashboard using Plotly.

Usage:
  python explainability_dashboard.py

Output:
  model/MaskablePPO/<model_id>/explainability_<dataset>/report.html
"""

import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def load_data(results_dir):
    """Load the CSV data from the results directory."""
    try:
        step_df = pd.read_csv(os.path.join(results_dir, 'step_level_data.csv'))
        trade_df = pd.read_csv(os.path.join(results_dir, 'trade_level_data.csv'))
        episode_df = pd.read_csv(os.path.join(results_dir, 'episode_summary.csv'))
        
        shap_path = os.path.join(results_dir, 'shap_importance.csv')
        shap_df = pd.read_csv(shap_path) if os.path.exists(shap_path) else None
        
        shap_grad_path = os.path.join(results_dir, 'shap_importance_gradient.csv')
        shap_df_grad = pd.read_csv(shap_grad_path) if os.path.exists(shap_grad_path) else None
        
        return step_df, trade_df, episode_df, shap_df, shap_df_grad
    except Exception as e:
        print(f"Error loading data from {results_dir}: {e}")
        return None, None, None, None, None


def create_kpi_section(episode_df):
    """Create HTML for Overall KPIs section."""
    total_pnl = episode_df['pnl'].sum()
    win_rate = (episode_df['pnl'] > 0).mean() * 100
    
    losses = episode_df[episode_df['pnl'] <= 0]['pnl'].mean()
    wins = episode_df[episode_df['pnl'] > 0]['pnl'].mean()
    avg_trade = episode_df['pnl'].mean()
    
    gross_profit = episode_df[episode_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(episode_df[episode_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    max_dd = episode_df['max_drawdown'].max()
    
    html = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px;">
        <div style="flex: 1; min-width: 200px; background: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="margin: 0; color: #555;">Total PnL</h3>
            <h2 style="margin: 10px 0 0; color: {'#28a745' if total_pnl >= 0 else '#dc3545'};">₹{total_pnl:,.2f}</h2>
        </div>
        <div style="flex: 1; min-width: 200px; background: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="margin: 0; color: #555;">Win Rate</h3>
            <h2 style="margin: 10px 0 0; color: #007bff;">{win_rate:.1f}%</h2>
        </div>
        <div style="flex: 1; min-width: 200px; background: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="margin: 0; color: #555;">Profit Factor</h3>
            <h2 style="margin: 10px 0 0; color: #17a2b8;">{profit_factor:.2f}</h2>
        </div>
        <div style="flex: 1; min-width: 200px; background: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="margin: 0; color: #555;">Max Drawdown</h3>
            <h2 style="margin: 10px 0 0; color: #dc3545;">₹{max_dd:,.2f}</h2>
        </div>
         <div style="flex: 1; min-width: 200px; background: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="margin: 0; color: #555;">Avg PnL / Ep</h3>
            <h2 style="margin: 10px 0 0; color: {'#28a745' if avg_trade >= 0 else '#dc3545'};">₹{avg_trade:,.2f}</h2>
        </div>
    </div>
    """
    
    # Cumulative PnL Chart
    episode_df['cumulative_pnl'] = episode_df['pnl'].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=episode_df['date'], y=episode_df['cumulative_pnl'], mode='lines', 
                             name='Cumulative PnL', line=dict(color='blue', width=3)))
    
    fig.update_layout(title='Cumulative Return Curve', xaxis_title='Date', yaxis_title='Total PnL (₹)',
                      template='plotly_white', height=400)
    
    html += fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    return html

def create_time_breakdown_section(episode_df, step_df=None):
    """Create HTML for Time-Period Breakdown sections."""
    episode_df['date'] = pd.to_datetime(episode_df['date'])
    
    # ── Monthly Aggregation ──
    # Generate all 12 months for the year range so every month is plotted
    episode_df['month_yr'] = episode_df['date'].dt.to_period('M').astype(str)
    monthly = episode_df.groupby('month_yr')['pnl'].sum().reset_index()

    # Build complete month index covering full date range
    min_date = episode_df['date'].min()
    max_date = episode_df['date'].max()
    all_months = pd.period_range(start=min_date, end=max_date, freq='M').astype(str)
    all_months_df = pd.DataFrame({'month_yr': all_months})
    monthly = all_months_df.merge(monthly, on='month_yr', how='left').fillna(0)
    # Explicit sort to guarantee chronological order
    monthly = monthly.sort_values('month_yr').reset_index(drop=True)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=monthly['month_yr'], 
        y=monthly['pnl'],
        marker_color=['green' if val >= 0 else 'red' for val in monthly['pnl']]
    ))
    fig1.update_layout(
        title='Monthly PnL', template='plotly_white', height=400,
        xaxis_title='Month', yaxis_title='PnL (₹)',
        xaxis=dict(type='category', categoryorder='array', categoryarray=monthly['month_yr'].tolist())
    )
    
    # ── Day of Week ──
    # Only trading days Mon–Fri; filter out Saturday (5) and Sunday (6)
    trading_dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    trading_dow_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    if step_df is not None and 'day_of_week' in step_df.columns and 'total_pnl' in step_df.columns:
        # Compute per-day PnL from step-level data:
        # For each (episode, date), take the last step's total_pnl minus the first step's total_pnl
        step_df_copy = step_df.copy()
        step_df_copy['timestamp'] = pd.to_datetime(step_df_copy['timestamp'])
        step_df_copy['step_date'] = step_df_copy['timestamp'].dt.date
        step_df_copy['step_dow'] = step_df_copy['timestamp'].dt.weekday

        # Filter out weekends (market off on Sat=5, Sun=6)
        step_df_copy = step_df_copy[step_df_copy['step_dow'] < 5]

        daily_pnl = step_df_copy.groupby(['episode', 'step_date', 'step_dow']).agg(
            pnl_start=('total_pnl', 'first'),
            pnl_end=('total_pnl', 'last')
        ).reset_index()
        daily_pnl['daily_pnl'] = daily_pnl['pnl_end'] - daily_pnl['pnl_start']
        daily_pnl['dow_name'] = daily_pnl['step_dow'].map(trading_dow_map)

        dow = daily_pnl.groupby('dow_name')['daily_pnl'].sum().reset_index()
        dow.columns = ['dow_name', 'pnl']
    else:
        # Fallback: use episode-level (single-day episodes)
        episode_df['dow_name'] = episode_df['day_of_week'].map(trading_dow_map)
        # Filter out any weekend episodes
        episode_df_filtered = episode_df[episode_df['day_of_week'] < 5]
        dow = episode_df_filtered.groupby('dow_name')['pnl'].sum().reset_index()

    # Ensure all 5 trading days are present (fill missing days like Tuesday with 0)
    all_days_df = pd.DataFrame({'dow_name': trading_dow_order})
    dow = all_days_df.merge(dow, on='dow_name', how='left').fillna(0)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=dow['dow_name'], 
        y=dow['pnl'],
        marker_color=['green' if val >= 0 else 'red' for val in dow['pnl']]
    ))
    fig2.update_layout(
        title='PnL by Day of Week', template='plotly_white', height=400,
        xaxis_title='Day of Week', yaxis_title='PnL (₹)',
        xaxis=dict(type='category', categoryorder='array', categoryarray=trading_dow_order)
    )
    
    html = "<div style='display: flex; gap: 20px;'>"
    html += f"<div style='flex: 1;'>{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>"
    html += f"<div style='flex: 1;'>{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>"
    html += "</div>"
    
    return html


def create_action_preference_section(step_df):
    """Create Action Preference Chart."""
    action_counts = step_df['action_name'].value_counts().reset_index()
    action_counts.columns = ['action', 'count']
    
    fig = go.Figure(data=[go.Pie(labels=action_counts['action'], values=action_counts['count'], hole=0.4)])
    fig.update_layout(title='Action Distribution (Overall)', template='plotly_white', height=450)
    
    # By Vol Regime
    vol_action = step_df.groupby(['vol_regime', 'action_name']).size().reset_index(name='count')
    fig2 = go.Figure()
    for act in vol_action['action_name'].unique():
        df_act = vol_action[vol_action['action_name'] == act]
        fig2.add_trace(go.Bar(x=df_act['vol_regime'], y=df_act['count'], name=act))
    fig2.update_layout(title='Action Choice by Volatility Regime', barmode='group', template='plotly_white', height=450)

    html = "<div style='display: flex; gap: 20px;'>"
    html += f"<div style='flex: 1;'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
    html += f"<div style='flex: 1;'>{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>"
    html += "</div>"
    
    return html


def create_market_regime_section(episode_df):
    """Create Market Regime Performance Chart."""
    
    # Aggregate by vol_regime
    vol_perf = episode_df.groupby('vol_regime').agg(
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100),
        count=('pnl', 'count')
    ).reset_index()

    # Aggregate by trend_regime
    trend_perf = episode_df.groupby('trend_regime').agg(
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100),
        count=('pnl', 'count')
    ).reset_index()
    
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Bar(x=vol_perf['vol_regime'], y=vol_perf['total_pnl'], name='Total PnL', marker_color='indigo'), secondary_y=False)
    fig1.add_trace(go.Scatter(x=vol_perf['vol_regime'], y=vol_perf['win_rate'], name='Win Rate %', line=dict(color='red', width=3)), secondary_y=True)
    fig1.update_layout(title='Performance by Volatility Regime', template='plotly_white', height=400)
    fig1.update_yaxes(title_text="Total PnL (₹)", secondary_y=False)
    fig1.update_yaxes(title_text="Win Rate (%)", secondary_y=True)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=trend_perf['trend_regime'], y=trend_perf['total_pnl'], name='Total PnL', marker_color='teal'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=trend_perf['trend_regime'], y=trend_perf['win_rate'], name='Win Rate %', line=dict(color='orange', width=3)), secondary_y=True)
    fig2.update_layout(title='Performance by Trend Regime', template='plotly_white', height=400)
    fig2.update_yaxes(title_text="Total PnL (₹)", secondary_y=False)
    fig2.update_yaxes(title_text="Win Rate (%)", secondary_y=True)
    
    html = "<div style='display: flex; gap: 20px;'>"
    html += f"<div style='flex: 1;'>{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>"
    html += f"<div style='flex: 1;'>{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>"
    html += "</div>"
    
    # Scatter VIX vs Return
    if 'vix' in episode_df.columns and episode_df['vix'].notna().any():
        fig3 = go.Figure()
        for trend in episode_df['trend_regime'].unique():
            df_trend = episode_df[episode_df['trend_regime'] == trend]
            fig3.add_trace(go.Scatter(
                x=df_trend['vix'], y=df_trend['pnl'],
                mode='markers', name=trend,
                marker=dict(size=df_trend['trades'].clip(lower=5, upper=40)),
                text=df_trend['date']
            ))
        fig3.update_layout(title='VIX vs Episode PnL', template='plotly_white', height=400,
                           xaxis_title='VIX', yaxis_title='PnL')
        html += f"<div style='margin-top: 20px;'>{fig3.to_html(full_html=False, include_plotlyjs=False)}</div>"

    return html

def create_intraday_behavior_section(trade_df):
    """Create Intraday behavior (timing) charts."""
    if trade_df is None or len(trade_df) == 0:
         return "<p>No trade data available.</p>"
         
    entry_df = trade_df[trade_df['is_entry'] == True]
    exit_df = trade_df[trade_df['is_exit'] == True]
    
    if len(entry_df) > 0:
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=entry_df['hour'], nbinsx=14))
        fig1.update_layout(title='Entry Time Distribution (Hour of Day)', template='plotly_white', height=350)
    else:
        fig1 = go.Figure()
        
    if len(exit_df) > 0:
        # PnL by holding time
        fig2 = go.Figure()
        for side in exit_df['side'].unique():
            df_side = exit_df[exit_df['side'] == side]
            fig2.add_trace(go.Scatter(
                x=df_side['holding_steps'], y=df_side['pnl'],
                mode='markers', name=side
            ))
        fig2.update_layout(title='PnL by Holding Time (Minutes)', template='plotly_white', height=350,
                           xaxis_title='Holding Time (steps)', yaxis_title='PnL')
    else:
        fig2 = go.Figure()

    html = "<div style='display: flex; gap: 20px;'>"
    html += f"<div style='flex: 1;'>{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>"
    html += f"<div style='flex: 1;'>{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>"
    html += "</div>"
    
    return html

def create_shap_section(shap_df, shap_df_grad=None):
    """Create SHAP importance bar chart."""
    if (shap_df is None or len(shap_df) == 0) and (shap_df_grad is None or len(shap_df_grad) == 0):
        return "<p>No SHAP data available. Run the test script with compute_shap=True.</p>"
        
    html = "<div style='display: flex; gap: 20px;'>"
    
    if shap_df is not None and len(shap_df) > 0:
        top_n = min(20, len(shap_df))
        df_plot = shap_df.head(top_n).sort_values('importance', ascending=True)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=df_plot['importance'], 
            y=df_plot['feature'],
            orientation='h',
            marker=dict(color=df_plot['importance'], colorscale='Viridis')
        ))
        fig1.update_layout(title=f'Perturbation-based SHAP ({top_n} Features)', template='plotly_white', height=600)
        html += f"<div style='flex: 1;'>{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>"
        
    if shap_df_grad is not None and len(shap_df_grad) > 0:
        top_n = min(20, len(shap_df_grad))
        df_plot = shap_df_grad.head(top_n).sort_values('importance', ascending=True)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_plot['importance'], 
            y=df_plot['feature'],
            orientation='h',
            marker=dict(color=df_plot['importance'], colorscale='Plasma')
        ))
        fig2.update_layout(title=f'Gradient-based Attribution ({top_n} Features)', template='plotly_white', height=600)
        html += f"<div style='flex: 1;'>{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>"

    html += "</div>"
    return html

def generate_report(dataset_label, results_dir):
    """Generate the full HTML report for a dataset."""
    print(f"Generating dashboard for {dataset_label} in {results_dir}...")
    
    step_df, trade_df, episode_df, shap_df, shap_df_grad = load_data(results_dir)
    
    if episode_df is None:
        print(f"Skipping {dataset_label} - data not found.")
        return
        
    # Build HTML sections
    kpi_html = create_kpi_section(episode_df)
    time_html = create_time_breakdown_section(episode_df, step_df=step_df)
    action_html = create_action_preference_section(step_df) if step_df is not None else ""
    regime_html = create_market_regime_section(episode_df)
    intraday_html = create_intraday_behavior_section(trade_df)
    shap_html = create_shap_section(shap_df, shap_df_grad)

    # Compile Full HTML
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Explainability Dashboard: {dataset_label.upper()}</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 15px rgba(0,0,0,0.05); border-radius: 10px; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }}
            h1 {{ margin: 0; color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 40px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            .section {{ margin-bottom: 50px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>Model Explainability Dashboard</h1>
                    <p style="color: #7f8c8d; margin: 5px 0 0;">Dataset: <strong>{dataset_label.upper()}</strong></p>
                </div>
                <div>
                    <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>1. Overall Performance</h2>
                {kpi_html}
            </div>
            
            <div class="section">
                <h2>2. Time-Period Breakdown</h2>
                {time_html}
            </div>
            
            <div class="section">
                <h2>3. Action Preferences</h2>
                {action_html}
            </div>
            
            <div class="section">
                <h2>4. Market Regime Analysis</h2>
                {regime_html}
            </div>
            
            <div class="section">
                <h2>5. Intraday Behavior</h2>
                {intraday_html}
            </div>
            
            <div class="section">
                <h2>6. Feature Importance (SHAP Approximation)</h2>
                {shap_html}
            </div>
        </div>
    </body>
    </html>
    """

    output_path = os.path.join(results_dir, 'dashboard_report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"Dashboard saved to: {output_path}")

def main():
    MODEL_ID = "20260424_142141"
    BASE_DIR = os.path.join('model', 'MaskablePPO', MODEL_ID)
    
    datasets = ['train', 'test']
    
    for ds in datasets:
        results_dir = os.path.join(BASE_DIR, f'explainability_{ds}')
        if os.path.exists(results_dir):
            generate_report(ds, results_dir)

if __name__ == "__main__":
    main()
