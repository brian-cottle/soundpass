import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SoundPass Real-Time")

# --- CUSTOM CSS FOR COMPACT UI ---
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .stPlotlyChart {
        height: 400px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- APP HEADER ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📡 SoundPass Real-Time Dashboard")
with col_h2:
    auto_refresh = st.checkbox("LIVE MODE", value=True)
    refresh_rate = st.slider("Update Rate (s)", 0.1, 2.0, 0.5)

# --- SIDEBAR: TRIGGER & SLICING ---
st.sidebar.header("🎯 Trigger Settings")
threshold_mv = st.sidebar.number_input("Threshold (mV)", value=50, step=5)
pre_trigger = st.sidebar.number_input("Pre-trigger (samples)", value=100, step=10)
pulse_size = st.sidebar.number_input("Pulse Length (samples)", value=4000, step=100)
holdoff_us = st.sidebar.number_input("Holdoff (µs)", value=20, step=5)

st.sidebar.divider()
st.sidebar.header("📊 Processing")
max_pulses = st.sidebar.slider("Max Pulses per Frame", 10, 500, 100)
decimation = st.sidebar.selectbox("Display Decimation", [1, 2, 4, 8], index=0)

# --- DATA LOADING & EXTRACTION ---
def get_latest_data():
    try:
        if not os.path.exists('data_raw.bin'):
            return None, 0
        mtime = os.path.getmtime('data_raw.bin')
        # Use np.fromfile directly (receiver uses os.replace for atomic updates)
        data = np.fromfile('data_raw.bin', dtype=np.int16)
        return data, mtime
    except Exception as e:
        return None, 0

def extract_pulses(buffer, threshold_counts, pre, size, holdoff):
    pulses = []
    i = 1000
    holdoff_samples = int(holdoff * 125)
    
    # Fast-ish NumPy extraction
    while i < (len(buffer) - size) and len(pulses) < max_pulses:
        # Check for trigger (simple threshold)
        if abs(buffer[i]) > threshold_counts:
            start = i - pre
            if start < 0: start = 0
            pulses.append(buffer[start : start + size])
            i += holdoff_samples
        else:
            i += 5 # Skip ahead slightly to speed up search
    return pulses

# --- MAIN EXECUTION LOOP ---
# Trigger counts: (mV / 1000) * (8192 / 20)
trigger_counts = int((threshold_mv / 1000.0) * (8192.0 / 20.0))

big_buffer, last_update = get_latest_data()

if big_buffer is not None:
    # Process
    pulses = extract_pulses(big_buffer, trigger_counts, pre_trigger, pulse_size, holdoff_us)
    
    # Layout: Top row for stats, bottom for plots
    st.write(f"**Buffer:** {len(big_buffer):,} samples | **Detected:** {len(pulses)} pulses | **Last Sync:** {time.strftime('%H:%M:%S', time.localtime(last_update))}")

    if pulses:
        # Convert to volts
        pulse_data = np.column_stack(pulses).astype(np.float32) * (20.0 / 8192.0)
        
        # TABBED VIEW
        t1, t2, t3 = st.tabs(["📈 Oscilloscope", "🌊 Waterfall", "📉 Average"])
        
        with t1:
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            # Plot up to 5 pulses
            for idx in range(min(5, pulse_data.shape[1])):
                ax1.plot(pulse_data[::decimation, idx], alpha=0.7, linewidth=1, label=f"Pulse {idx+1}")
            ax1.set_ylabel("Voltage (V)")
            ax1.set_xlabel(f"Samples (Decimated x{decimation})")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            st.pyplot(fig1)

        with t2:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            # Waterfall is best as an image
            im = ax2.imshow(pulse_data[::decimation, :].T, aspect='auto', cmap='magma', origin='lower')
            plt.colorbar(im, ax=ax2, label="Volts")
            ax2.set_ylabel("Pulse Index")
            ax2.set_xlabel("Time (Samples)")
            st.pyplot(fig2)

        with t3:
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            avg = np.mean(pulse_data, axis=1)
            ax3.plot(avg, color='cyan', linewidth=1.5)
            ax3.set_ylabel("Mean Voltage (V)")
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
    else:
        st.warning(f"No pulses detected. Try lowering the threshold from {threshold_mv}mV.")
else:
    st.info("Waiting for `data_raw.bin`... Is the `stream_receiver.py` running?")

# --- AUTO-REFRESH LOGIC ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
