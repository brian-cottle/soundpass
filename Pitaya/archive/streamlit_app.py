import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Ultrasound AXI Tuner")

st.title("Ultrasound AXI Trigger Tuner")
st.markdown("Use this app to interactively tune the software trigger logic on your raw AXI data (`data_raw.bin`) without needing to recapture from the hardware.")

@st.cache_data
def load_data():
    try:
        # Read raw 16-bit integers
        return np.fromfile('data_raw.bin', dtype=np.int16)
    except FileNotFoundError:
        return None

big_buffer = load_data()

if big_buffer is None:
    st.error("Could not find 'data_raw.bin'. Please ensure you have copied the file from the Red Pitaya to the same directory as this script.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Trigger Settings")

threshold_mv = st.sidebar.number_input("Trigger Threshold (mV)", value=500, step=10, help="Absolute voltage jump from baseline required to trigger.")
# Convert mV to raw counts: (mV / 1000) * (8192 / 20) -> Because 20V = 8192 counts
trigger_counts = int((threshold_mv / 1000.0) * (8192.0 / 20.0))

trigger_edge = st.sidebar.selectbox("Trigger Edge", ["Rising (Positive)", "Falling (Negative)", "Both"], help="Which direction the signal must jump to trigger.")

st.sidebar.header("Pulse Slicing")
pre_trigger = st.sidebar.number_input("Pre-trigger Samples", value=50, step=10, help="How many samples to include BEFORE the trigger hits.")
pulse_size = st.sidebar.number_input("Total Pulse Samples", value=3000, step=100, help="Total length of the extracted pulse.")

holdoff_us = st.sidebar.number_input("Holdoff Time (us)", value=10, step=10, help="Dead time after a trigger before looking for the next one. Prevents double-triggering on ringing.")
holdoff_samples = int(holdoff_us * 125) # 125 MS/s -> 125 samples per microsecond

max_pulses = st.sidebar.number_input("Max Pulses to Extract", value=1000, step=10)

# --- STATS ---
st.write(f"**Loaded Data:** {len(big_buffer):,} samples ({len(big_buffer)*8/1000000:.2f} ms of recording at 125 MS/s)")
st.write(f"**Calculated Trigger Sensitivity:** {trigger_counts} raw counts")

# --- SOFTWARE EXTRACTION LOGIC ---
pulses = []
trigger_indices = []
i = 1000 # Start after edge effects
baseline = int(np.mean(big_buffer[100:500])) 

while i < (len(big_buffer) - pulse_size) and len(pulses) < max_pulses:
    local_baseline = big_buffer[i-100] # Dynamic baseline tracking
    
    triggered = False
    if trigger_edge == "Both":
        triggered = big_buffer[i] > (local_baseline + trigger_counts) or big_buffer[i] < (local_baseline - local_baseline - trigger_counts)
    elif trigger_edge == "Rising (Positive)":
        triggered = big_buffer[i] > (local_baseline + trigger_counts)
    elif trigger_edge == "Falling (Negative)":
        triggered = big_buffer[i] < (local_baseline - trigger_counts)

    if triggered:
        # Found a trigger!
        start_idx = i - pre_trigger
        if start_idx < 0: start_idx = 0
        end_idx = start_idx + pulse_size
        
        if end_idx <= len(big_buffer):
            pulse = big_buffer[start_idx : end_idx]
            pulses.append(pulse)
            trigger_indices.append(i)
            i += holdoff_samples
        else:
            break
    else:
        i += 1

pulse_count = len(pulses)

if pulse_count == 0:
    st.warning("No pulses found with current threshold settings.")
else:
    st.success(f"Successfully extracted {pulse_count} pulses.")
    
    # Convert to Volts (HV mode: 8192 counts = 20V)
    pulse_array_volts = (np.column_stack(pulses).astype(np.float32) * 20.0) / 8192.0

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Pulses (Aligned)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        indices_to_plot = np.linspace(0, pulse_array_volts.shape[1]-1, min(5, pulse_array_volts.shape[1]), dtype=int)
        for idx_i, idx in enumerate(indices_to_plot):
            ax1.plot(pulse_array_volts[:, idx] + (idx_i * 0.1), label=f'Pulse_{idx+1}', alpha=0.8, linewidth=1)
        
        # Draw a vertical line where the trigger point is
        ax1.axvline(x=pre_trigger, color='r', linestyle='--', alpha=0.5, label='Trigger Point')
        ax1.set_xlabel("Sample Index (8ns per sample)")
        ax1.set_ylabel("Voltage (V) - Offset")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

    with col2:
        st.subheader("Averaged Waveform")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        averaged_pulse = np.mean(pulse_array_volts, axis=1)
        ax2.plot(averaged_pulse, color='red', linewidth=1.5)
        ax2.axvline(x=pre_trigger, color='k', linestyle='--', alpha=0.5, label='Trigger Point')
        ax2.set_xlabel("Sample Index (8ns per sample)")
        ax2.set_ylabel("Voltage (V)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    st.subheader("Waterfall Heatmap")
    st.markdown("If the pulses are perfectly aligned, you will see a straight vertical pattern.")
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    im = ax3.imshow(pulse_array_volts.T, aspect='auto', cmap='viridis', origin='lower')
    fig3.colorbar(im, ax=ax3, label='Voltage (V)')
    ax3.set_xlabel("Sample Index (8ns per sample)")
    ax3.set_ylabel("Pulse Number")
    st.pyplot(fig3)

# --- UNIFIED RAW DATA EXPLORER ---
st.divider()
st.header("Raw Data Explorer")
st.markdown("Pan through the entire 2MB raw data buffer to see the true signal and exactly why the software triggers where it does. Red lines indicate extracted trigger points.")

col_raw1, col_raw2 = st.columns(2)
with col_raw1:
    raw_window = st.number_input("Window Size (samples)", value=50000, step=5000, help="How many samples to view at once.")
with col_raw2:
    # Use a slider to pan through the data
    raw_start = st.slider("Start Sample Index", min_value=0, max_value=max(0, len(big_buffer) - raw_window), value=0, step=1000)

fig_raw, ax_raw = plt.subplots(figsize=(14, 4))
raw_end = min(len(big_buffer), raw_start + raw_window)

# Extract segment and convert to volts
segment = big_buffer[raw_start:raw_end]
volts = (segment * 20.0) / 8192.0

# Create x-axis array for correct absolute indices
x_indices = np.arange(raw_start, raw_end)
ax_raw.plot(x_indices, volts, color='gray', alpha=0.7)

# Mark triggers that fall within this window
visible_triggers = [t for t in trigger_indices if raw_start <= t <= raw_end]
for t_idx in visible_triggers:
    ax_raw.axvline(x=t_idx, color='r', linestyle='--', alpha=0.8)

# --- CALCULATE & DISPLAY PULSE RATE ---
if len(visible_triggers) > 1:
    # Calculate the average distance between triggers in samples
    avg_samples_between_triggers = np.mean(np.diff(visible_triggers))
    # Convert samples to time (8ns per sample)
    time_between_triggers_sec = avg_samples_between_triggers * 8e-9
    # Calculate frequency (Hz)
    pulse_rate_hz = 1.0 / time_between_triggers_sec
    
    st.info(f"**Calculated Pulse Rate in Window:** ~{pulse_rate_hz / 1000.0:.2f} kHz "
            f"(Average of {avg_samples_between_triggers:.0f} samples / {time_between_triggers_sec * 1e6:.2f} µs between pulses)")
elif len(visible_triggers) == 1:
    st.info("Only 1 trigger visible in window. Need at least 2 to calculate rate.")

# Draw threshold lines
ax_raw.axhline(y=threshold_mv/1000.0, color='b', linestyle=':', label='+ Threshold')
ax_raw.axhline(y=-threshold_mv/1000.0, color='b', linestyle=':', label='- Threshold')

ax_raw.set_ylabel("Voltage (V)")
ax_raw.set_xlabel("Absolute Sample Index")
ax_raw.set_xlim(raw_start, raw_end)
ax_raw.legend(loc='upper right')
st.pyplot(fig_raw)