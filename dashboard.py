
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.intraday_option_env import IntradayOptionEnv
import config
import datetime

# Page Params
st.set_page_config(page_title="RL Trading Simulation", layout="wide")

# --- 1. Load Resources ---
@st.cache_resource
def load_resources():
    # Model
    model_path = os.path.join('model', 'PPO', 'ppo_intraday_model')
    try:
        model = PPO.load(model_path)
    except Exception as e:
        model = None
        st.error(f"Error loading model: {e}")
        
    return model

model = load_resources()

import time

# --- 2. Sidebar Controls ---
st.sidebar.title("Configuration")

# Data File
data_path = st.sidebar.text_input("Data Path", "data/test.csv")
strategy_type = st.sidebar.selectbox("Strategy Type", ["LONG", "SHORT"], index=0)
config.STRATEGY_TYPE = strategy_type

# Simulation Speed
sim_speed = st.sidebar.slider("Simulation Speed (seconds per step)", 0.01, 2.0, 0.1)

# Load Data Dates
try:
    df_preview = pd.read_csv(data_path)
    df_preview['datetime'] = pd.to_datetime(df_preview['datetime'])
    available_dates = sorted(df_preview['datetime'].dt.date.unique())
    
    selected_date = st.sidebar.selectbox("Select Date", available_dates)
    
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    available_dates = []
    selected_date = None

run_btn = st.sidebar.button("Run Simulation")

# --- 3. Main Dashboard ---
st.title(f"Intraday RL Simulation: {strategy_type} Strategy")

if run_btn and selected_date and model:
    try:
        env = IntradayOptionEnv(data_path=data_path, mode='add')
        
        # Find index
        date_idx = -1
        for idx, d in enumerate(env.dates):
            if d == selected_date:
                date_idx = idx
                break
                
        if date_idx == -1:
            st.error("Selected date not found in environment processing.")
        else:
            # Layout Placeholders
            col1, col2, col3, col4 = st.columns(4)
            metric_pnl = col1.empty()
            metric_realized = col2.empty()
            metric_unrealized = col3.empty()
            metric_trades = col4.empty()
            
            st.subheader("Market & Performance")
            chart_placeholder = st.empty()
            
            st.subheader("Live Portfolio")
            portfolio_table = st.empty()
            
            st.subheader("Trading Logs")
            logs_table = st.empty()
            
            # Init Data Containers for Charts
            chart_data = pd.DataFrame(columns=['Underlying', 'PnL'])
            
            with st.spinner(f"Initializing {selected_date}..."):
                obs, info = env.reset(options={'date_index': date_idx})
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    
                    # Capture Data
                    current_step_row = env.day_data.iloc[env.current_step-1]
                    current_time = current_step_row['datetime']
                    price = current_step_row['close']
                    total_pnl = env.pnl_curve[-1]
                    realized_pnl = env.realized_pnl
                    unrealized_pnl = total_pnl - realized_pnl
                    
                    # Update Metrics
                    metric_pnl.metric("Total PnL", f"{total_pnl:.2f}")
                    metric_realized.metric("Realized PnL", f"{realized_pnl:.2f}")
                    metric_unrealized.metric("Unrealized PnL", f"{unrealized_pnl:.2f}")
                    metric_trades.metric("Trades", len(env.trade_logs))
                    
                    # Update Chart
                    # Efficiently append to chart data
                    new_row = pd.DataFrame({'Underlying': [price], 'PnL': [total_pnl]}, index=[current_time])
                    chart_data = pd.concat([chart_data, new_row])
                    
                    # Use st.line_chart for dynamic updates (Dual axis is tricky in native st.line_chart, 
                    # so we might see two lines on different scales. For better UX, we normalize or use plotly, 
                    # but keeping it simple for speed).
                    # Actually, plotting Price vs PnL on same axis is bad scaling.
                    # Let's just plot Underlying Price here for speed, or use Altair/Plotly for dual axis if requested.
                    # For now: Just Price to convert "tick by tick".
                    chart_placeholder.line_chart(chart_data[['Underlying', 'PnL']]) 
                    
                    # Update Portfolio
                    portfolio = [
                        {'Leg': 'CE', 'Strike': env.ce_strike, 'Lots': env.ce_lots, 'Avg Price': f"{env.entry_price_ce:.2f}"},
                        {'Leg': 'PE', 'Strike': env.pe_strike, 'Lots': env.pe_lots, 'Avg Price': f"{env.entry_price_pe:.2f}"}
                    ]
                    portfolio_table.table(pd.DataFrame(portfolio))
                    
                    # Update Logs (Scroll to bottom? Streamlit doesn't auto-scroll tables easily)
                    if env.trade_logs:
                        df_logs = pd.DataFrame(env.trade_logs)
                        logs_table.dataframe(df_logs.tail(5), use_container_width=True) # Show last 5 trades
                    
                    # Animation Delay
                    time.sleep(sim_speed)

            # --- Post-Simulation ---
            st.success("Simulation Complete")
            
            # Final Full Logs
            if env.trade_logs:
                df_logs = pd.DataFrame(env.trade_logs)
                csv = df_logs.to_csv(index=False).encode('utf-8')
                st.download_button("Download Logic (CSV)", csv, f"logs_{selected_date}.csv", "text/csv")

    except Exception as e:
        st.error(f"Simulation Failed: {e}")
        st.exception(e)

elif not model:
    st.warning("Please verify model path. No model loaded.")

elif not selected_date:
    st.warning("Please select a date.")
