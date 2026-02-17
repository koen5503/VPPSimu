"""
NED Baseload Power Simulation Dashboard
=======================================
A Streamlit based dashboard to simulate baseload power generation using renewable energy sources
(Solar, Wind Onshore, Wind Offshore) combined with Battery and Hydrogen storage systems.

Usage:
    streamlit run simulation_dashboard.py
"""

import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants ──────────────────────────────────────────────────────────
EXCEL_FILE = "energy_data_ned.xlsx"

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NED Baseload Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Data Loading ───────────────────────────────────────────────────────
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load all available sheets from energy_data_ned.xlsx and combine them
    into a single continuous DataFrame sorted by time.
    """
    if not os.path.exists(file_path):
        st.error(f"Data file not found: {file_path}")
        return pd.DataFrame()

    try:
        xls = pd.ExcelFile(file_path, engine="openpyxl")
        all_dfs = []
        
        for sheet_name in xls.sheet_names:
            # Check if sheet is a year sheet (e.g. "Y2023", "Y2024")
            if sheet_name.startswith("Y") and sheet_name[1:].isdigit():
                df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0, engine="openpyxl")
                # Ensure index is datetime and tz-aware (UTC)
                df.index = pd.to_datetime(df.index, utc=True)
                all_dfs.append(df)
        
        if not all_dfs:
            st.warning("No valid year sheets found in the Excel file.")
            return pd.DataFrame()
            
        combined_df = pd.concat(all_dfs).sort_index()
        # Remove duplicates if any
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        return combined_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# ── Settings Persistence ──────────────────────────────────────────────
SETTINGS_FILE = "LastSettings.xlsx"

def load_settings():
    """Load simulation settings from LastSettings.xlsx if it exists."""
    if os.path.exists(SETTINGS_FILE):
        try:
            df = pd.read_excel(SETTINGS_FILE, index_col=0, engine="openpyxl")
            return df["Value"].to_dict()
        except:
            pass
    return {}

def save_settings(settings_dict):
    """Save current simulation settings to LastSettings.xlsx."""
    try:
        df = pd.DataFrame.from_dict(settings_dict, orient='index', columns=['Value'])
        df.to_excel(SETTINGS_FILE, engine="openpyxl")
    except Exception as e:
        print(f"Failed to save settings: {e}")

# ── Cost Data Loading ────────────────────────────────────────────────
@st.cache_data
def load_costs():
    """Load cost parameters from costs.xlsx."""
    if os.path.exists("costs.xlsx"):
        try:
            df = pd.read_excel("costs.xlsx", index_col=0, engine="openpyxl")
            # Clean up index (remove trailing whitespace)
            df.index = df.index.str.strip()
            return df
        except:
            pass
    return pd.DataFrame()

def main():
    st.title("🇳🇱 NED Baseload Power Simulation")
    st.caption("Simulate renewable generation + storage to meet a flat baseload target.")

    # Load Data
    with st.spinner("Loading capacity factor data..."):
        df_all = load_data(EXCEL_FILE)
    
    # Load Costs
    df_costs = load_costs()

    if df_all.empty:
        st.stop()

    # Determine date range limits
    min_date = df_all.index.min().date()
    max_date = df_all.index.max().date()
    
    # Load previous settings
    defaults = load_settings()
    
    # ── Sidebar Configuration ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Simulation Settings")
        
        # 1. Generation Parameters
        with st.expander("Generation Mix (GW)", expanded=True):
            cap_solar = st.number_input("Solar Capacity", min_value=0.0, value=float(defaults.get("cap_solar", 10.0)), step=0.5, format="%.1f")
            cap_onshore = st.number_input("Wind Onshore Capacity", min_value=0.0, value=float(defaults.get("cap_onshore", 5.0)), step=0.5, format="%.1f")
            cap_offshore = st.number_input("Wind Offshore Capacity", min_value=0.0, value=float(defaults.get("cap_offshore", 20.0)), step=0.5, format="%.1f")
            
        # 2. Baseload Target
        with st.expander("Demand Settings", expanded=True):
            baseload_target = st.number_input("Baseload Target (GW)", min_value=0.1, value=float(defaults.get("baseload_target", 5.0)), step=0.1, format="%.1f")
            
        # 3. Storage: Battery
        with st.expander("Battery Storage (Li-ion)", expanded=False):
            batt_cap = st.number_input("Battery Capacity (GWh)", min_value=0.0, value=float(defaults.get("batt_cap", 10.0)), step=1.0)
            batt_power = st.number_input("Battery Power (GW)", min_value=0.0, value=float(defaults.get("batt_power", 5.0)), step=0.5)
            batt_eff = 0.95 # Charge/Discharge efficiency
            batt_init_soc = st.slider("Initial SoC (%)", 0, 100, int(defaults.get("batt_init_soc", 50)), key="batt_soc") / 100.0

        # 4. Storage: Hydrogen
        with st.expander("Hydrogen Storage (H2)", expanded=False):
            h2_cap = st.number_input("H2 Capacity (GWh, LHV)", min_value=0.0, value=float(defaults.get("h2_cap", 500.0)), step=50.0)
            h2_ely_power = st.number_input("Electrolyzer Power (GW)", min_value=0.0, value=float(defaults.get("h2_ely_power", 10.0)), step=0.5)
            h2_fc_power = st.number_input("Fuel Cell Power (GW)", min_value=0.0, value=float(defaults.get("h2_fc_power", 5.0)), step=0.5)
            # Efficiencies
            eff_ely = 0.70
            eff_fc = 0.50
            h2_init_soc = st.slider("Initial SoC (%)", 0, 100, int(defaults.get("h2_init_soc", 50)), key="h2_soc") / 100.0
            
    # Save settings for next run
    current_settings = {
        "cap_solar": cap_solar, "cap_onshore": cap_onshore, "cap_offshore": cap_offshore,
        "baseload_target": baseload_target,
        "batt_cap": batt_cap, "batt_power": batt_power, "batt_init_soc": int(batt_init_soc*100),
        "h2_cap": h2_cap, "h2_ely_power": h2_ely_power, "h2_fc_power": h2_fc_power, "h2_init_soc": int(h2_init_soc*100)
    }
    save_settings(current_settings)

    # ── Main Area: Filters & Simulation ────────────────────────────────
    
    # 3. 3. Precise time selection
    col_d1, col_t1, col_d2, col_t2 = st.columns([2, 1, 2, 1])
    with col_d1:
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col_t1:
        start_time = st.time_input("Start Time", value=datetime.time(0, 0))
    with col_d2:
        end_date = st.date_input("End Date", value=min_date + datetime.timedelta(days=14), min_value=min_date, max_value=max_date)
    with col_t2:
        end_time = st.time_input("End Time", value=datetime.time(23, 0))
        
    # Combine date and time
    dt_start = datetime.datetime.combine(start_date, start_time)
    dt_end = datetime.datetime.combine(end_date, end_time)
        
    if dt_start >= dt_end:
        st.error("Start datetime must be before End datetime.")
        st.stop()
        
    # Filter Data
    # Convert dates to timestamps (UTC)
    start_ts = pd.Timestamp(dt_start, tz="UTC")
    end_ts = pd.Timestamp(dt_end, tz="UTC")
    
    mask = (df_all.index >= start_ts) & (df_all.index <= end_ts)
    df_sim = df_all.loc[mask].copy()
    
    if df_sim.empty:
        st.warning("No data found for the selected date range.")
        st.stop()

    st.write(f"Simulating **{len(df_sim)} hours** ({dt_start} to {dt_end})...")

    # ── Simulation Logic (NumPy) ───────────────────────────────────────
    
    # Prepare input arrays
    # Capacity Factors - handle missing columns with 0
    cf_solar = df_sim["Solar"].fillna(0).values if "Solar" in df_sim else np.zeros(len(df_sim))
    cf_onshore = df_sim["Wind Onshore"].fillna(0).values if "Wind Onshore" in df_sim else np.zeros(len(df_sim))
    cf_offshore = df_sim["Wind Offshore"].fillna(0).values if "Wind Offshore" in df_sim else np.zeros(len(df_sim))
    
    # Calculate Renewable Generation Profile
    gen_solar = cap_solar * cf_solar
    gen_onshore = cap_onshore * cf_onshore
    gen_offshore = cap_offshore * cf_offshore
    total_gen = gen_solar + gen_onshore + gen_offshore
    
    # Net Load (target is to meet baseload)
    net_load = total_gen - baseload_target
    
    # Initialize arrays for results
    n_steps = len(df_sim)
    
    batt_soc_arr = np.zeros(n_steps)
    h2_soc_arr = np.zeros(n_steps)
    
    batt_charge_arr = np.zeros(n_steps)
    batt_discharge_arr = np.zeros(n_steps)
    h2_charge_arr = np.zeros(n_steps)
    h2_discharge_arr = np.zeros(n_steps)
    curtailment_arr = np.zeros(n_steps)
    loss_load_arr = np.zeros(n_steps)
    
    # Initial states
    current_batt_soc = batt_init_soc * batt_cap
    current_h2_soc = h2_init_soc * h2_cap
    
    # Simulation Loop
    for i in range(n_steps):
        nl = net_load[i]
        
        # --- Surplus (> 0) ---
        if nl > 0:
            surplus = nl
            
            # 1. Charge Battery
            p_batt_in = min(surplus, batt_power, (batt_cap - current_batt_soc) / batt_eff if batt_eff > 0 else 0)
            
            batt_charge_arr[i] = p_batt_in
            current_batt_soc += p_batt_in * batt_eff
            remaining_surplus = surplus - p_batt_in
            
            # 2. Charge Hydrogen
            if remaining_surplus > 1e-6:
                p_h2_in = min(remaining_surplus, h2_ely_power, (h2_cap - current_h2_soc) / eff_ely if eff_ely > 0 else 0)
                
                h2_charge_arr[i] = p_h2_in
                current_h2_soc += p_h2_in * eff_ely
                remaining_surplus -= p_h2_in
                
            # 3. Curtailment
            if remaining_surplus > 1e-6:
                curtailment_arr[i] = remaining_surplus
                
        # --- Deficit (< 0) ---
        else:
            deficit = -nl # Make positive for calculation
            
            # 1. Discharge Battery
            # Output Power = min(Deficit, PowerRating, EnergyAvailable * Eff)
            p_batt_out = min(deficit, batt_power, current_batt_soc * batt_eff)
            
            batt_discharge_arr[i] = p_batt_out
            current_batt_soc -= p_batt_out / batt_eff
            remaining_deficit = deficit - p_batt_out
            
            # 2. Discharge Hydrogen
            if remaining_deficit > 1e-6:
                p_h2_out = min(remaining_deficit, h2_fc_power, current_h2_soc * eff_fc)
                
                h2_discharge_arr[i] = p_h2_out
                current_h2_soc -= p_h2_out / eff_fc
                remaining_deficit -= p_h2_out
                
            # 3. Loss of Load
            if remaining_deficit > 1e-6:
                loss_load_arr[i] = remaining_deficit
        
        # Store state
        current_batt_soc = max(0.0, min(current_batt_soc, batt_cap))
        current_h2_soc = max(0.0, min(current_h2_soc, h2_cap))
        
        batt_soc_arr[i] = current_batt_soc
        h2_soc_arr[i] = current_h2_soc

    # Add results to df_sim for plotting
    df_sim["Net_Load"] = net_load
    df_sim["Solar_Gen"] = gen_solar
    df_sim["Wind_Onshore_Gen"] = gen_onshore
    df_sim["Wind_Offshore_Gen"] = gen_offshore
    df_sim["Batt_Charge"] = batt_charge_arr
    df_sim["Batt_Discharge"] = batt_discharge_arr
    df_sim["H2_Charge"] = h2_charge_arr
    df_sim["H2_Discharge"] = h2_discharge_arr
    df_sim["Curtailment"] = curtailment_arr
    df_sim["Loss_Load"] = loss_load_arr
    
    df_sim["Batt_SoC"] = batt_soc_arr / batt_cap * 100 if batt_cap > 0 else 0
    df_sim["H2_SoC"] = h2_soc_arr / h2_cap * 100 if h2_cap > 0 else 0
    
    # ── Outputs & Visuals ────────────────────────────────────────────────
    
    st.divider()
    
    # 0. Investment & Land Use (New Row)
    if not df_costs.empty:
        # Helper to get value
        def get_cost_param(name, col):
            try:
                # Assuming index is the parameter name from costs.xlsx
                return float(df_costs.loc[name, col])
            except:
                return 0.0

        # Calculate Costs (Billion €)
        # Power (GW) * 1e9 (W) * Cost (€/W) / 1e9 (B€) = GW * Cost
        # Energy (GWh) * 1e9 (Wh) * Cost (€/Wh) / 1e9 (B€) = GWh * Cost
        
        cost_solar = cap_solar * 1e9 * get_cost_param("Solar Capacity", "Investment Cost")
        cost_onshore = cap_onshore * 1e9 * get_cost_param("Wind Onshore Capacity", "Investment Cost")
        cost_offshore = cap_offshore * 1e9 * get_cost_param("Wind Offshore Capacity", "Investment Cost")
        cost_batt_cap = batt_cap * 1e9 * get_cost_param("Battery Capacity", "Investment Cost")
        cost_batt_pow = batt_power * 1e9 * get_cost_param("Battery Power", "Investment Cost")
        cost_h2_cap = h2_cap * 1e9 * get_cost_param("H2 Capacity", "Investment Cost")
        cost_ely = h2_ely_power * 1e9 * get_cost_param("Electrolyzer Power", "Investment Cost")
        cost_fc = h2_fc_power * 1e9 * get_cost_param("Fuel Cell Power", "Investment Cost")
        
        total_investment_eur = (cost_solar + cost_onshore + cost_offshore + 
                                cost_batt_cap + cost_batt_pow + 
                                cost_h2_cap + cost_ely + cost_fc)
        total_investment_beur = total_investment_eur / 1e9
        
        # Calculate Land Use (km²)
        # Power (GW) * 1e9 (W) * LandUse (m2/W) / 1e6 (km2) = GW * 1000 * LandUse
        # Energy (GWh) * 1e9 (Wh) * LandUse (m2/Wh) / 1e6 (km2) = GWh * 1000 * LandUse
        
        area_solar = cap_solar * 1000 * get_cost_param("Solar Capacity", "Land use")
        area_onshore = cap_onshore * 1000 * get_cost_param("Wind Onshore Capacity", "Land use")
        area_offshore = cap_offshore * 1000 * get_cost_param("Wind Offshore Capacity", "Land use")
        area_batt_cap = batt_cap * 1000 * get_cost_param("Battery Capacity", "Land use") # m2/Wh
        area_batt_pow = batt_power * 1000 * get_cost_param("Battery Power", "Land use")
        area_h2_cap = h2_cap * 1000 * get_cost_param("H2 Capacity", "Land use")
        area_ely = h2_ely_power * 1000 * get_cost_param("Electrolyzer Power", "Land use")
        area_fc = h2_fc_power * 1000 * get_cost_param("Fuel Cell Power", "Land use")
        
        total_area_km2 = (area_solar + area_onshore + area_offshore +
                          area_batt_cap + area_batt_pow +
                          area_h2_cap + area_ely + area_fc)
        
        m1, m2 = st.columns(2)
        m1.metric("Total Investment", f"€ {total_investment_beur:.2f} B")
        m2.metric("Land/Sea Use", f"{total_area_km2:.1f} km²")
        
        # Detailed Breakdown Table
        with st.expander("Investment & Land Use Details", expanded=False):
            cost_details = {
                "Technology": [
                    "Solar", "Wind Onshore", "Wind Offshore", 
                    "Battery Capacity", "Battery Power", 
                    "H2 Capacity", "Electrolyzer", "Fuel Cell"
                ],
                "Capacity/Power": [
                    f"{cap_solar} GW", f"{cap_onshore} GW", f"{cap_offshore} GW", 
                    f"{batt_cap} GWh", f"{batt_power} GW", 
                    f"{h2_cap} GWh", f"{h2_ely_power} GW", f"{h2_fc_power} GW"
                ],
                "Investment (B€)": [
                    cost_solar/1e9, cost_onshore/1e9, cost_offshore/1e9,
                    cost_batt_cap/1e9, cost_batt_pow/1e9,
                    cost_h2_cap/1e9, cost_ely/1e9, cost_fc/1e9
                ],
                "Land Use (km²)": [
                    area_solar, area_onshore, area_offshore,
                    area_batt_cap, area_batt_pow,
                    area_h2_cap, area_ely, area_fc
                ]
            }
            df_breakdown = pd.DataFrame(cost_details)
            # Add a Total row
            df_breakdown.loc[len(df_breakdown)] = [
                "**Total**", "", total_investment_beur, total_area_km2
            ]
            st.dataframe(df_breakdown.style.format({
                "Investment (B€)": "{:.3f}", 
                "Land Use (km²)": "{:.3f}"
            }), width="stretch")

        
        st.divider()

    # 1. KPI Metrics
    total_hours = len(df_sim)
    blackout_hours = np.count_nonzero(loss_load_arr > 1e-3)
    reliability = 1.0 - (blackout_hours / total_hours) if total_hours > 0 else 0
    
    total_curtailment = np.sum(curtailment_arr) / 1000.0 # TWh
    
    batt_cycles = np.sum(batt_discharge_arr) / batt_cap if batt_cap > 0 else 0
    h2_cycles = np.sum(h2_discharge_arr) / h2_cap if h2_cap > 0 else 0
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Reliability", f"{reliability:.2%}", help=f"Hours with blackout: {blackout_hours}")
    kpi2.metric("Total Curtailment", f"{total_curtailment:.2f} TWh")
    kpi3.metric("Battery Cycles", f"{batt_cycles:.1f}")
    kpi4.metric("H2 Cycles", f"{h2_cycles:.1f}")
    
    # ── 1. Annual Energy Table ──
    st.subheader("Annual Energy Summary (TWh)")
    
    # Prepare aggregation
    # Group by Year
    df_sim["Year"] = df_sim.index.year
    df_yearly = df_sim.groupby("Year").sum()
    
    # Columns requested: Solar, Wind On, Wind Off, Batt Charge, Batt Dis, Batt Loss, H2 Charge, H2 Dis, H2 Loss
    # All values currently in GW/h (energy in GWh per hour step). Sum = GWh total. Convert to TWh (/1000).
    
    summary_data = []
    
    # Rows: Solar, Wind On, Wind Off, Batt Charge, Batt Dis, Batt Loss, H2 Charge, H2 Dis, H2 Loss
    metrics_map = [
        ("Solar Generation", "Solar_Gen"),
        ("Wind Onshore Gen", "Wind_Onshore_Gen"),
        ("Wind Offshore Gen", "Wind_Offshore_Gen"),
        ("Battery Charged", "Batt_Charge"),
        ("Battery Discharged", "Batt_Discharge"),
        ("Battery Loss", None), # Calc
        ("H2 Charged", "H2_Charge"),
        ("H2 Discharged", "H2_Discharge"),
        ("H2 Loss", None), # Calc
    ]
    
    table_index = [m[0] for m in metrics_map]
    table_dict = {}
    
    for year in df_yearly.index:
        year_vals = []
        row_data = df_yearly.loc[year]
        
        # Solar
        year_vals.append(row_data["Solar_Gen"] / 1000)
        # Wind On
        year_vals.append(row_data["Wind_Onshore_Gen"] / 1000)
        # Wind Off
        year_vals.append(row_data["Wind_Offshore_Gen"] / 1000)
        
        # Batt
        b_ch = row_data["Batt_Charge"] / 1000
        b_dis = row_data["Batt_Discharge"] / 1000
        year_vals.append(b_ch)
        year_vals.append(b_dis)
        year_vals.append(b_ch - b_dis)
        
        # H2
        h_ch = row_data["H2_Charge"] / 1000
        h_dis = row_data["H2_Discharge"] / 1000
        year_vals.append(h_ch)
        year_vals.append(h_dis)
        year_vals.append(h_ch - h_dis)
        
        table_dict[str(year)] = year_vals
        
    df_table = pd.DataFrame(table_dict, index=table_index)
    st.dataframe(df_table.style.format("{:.2f}"), width="stretch")

    # ── 2. Power Balance Graph ──
    st.divider()
    
    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        graph_type = st.radio("Graph Type", ["Stacked Area", "Line (Non-stacked)"], horizontal=True)
    
    all_traces = [
        "Solar", "Wind Onshore", "Wind Offshore", 
        "Batt Discharge", "H2 Discharge",
        "Batt Charge", "H2 Charge", "Curtailment", "Baseload Target"
    ]
    
    selected_traces = all_traces # Default all
    
    if graph_type == "Line (Non-stacked)":
        with col_g2:
            selected_traces = st.multiselect("Select Traces", all_traces, default=all_traces)
    
    fig_bal = go.Figure()
    
    # Helper to add trace if selected
    def add_trace_if_selected(name, y_data, color, stackgroup=None, fill=None, is_neg=False):
        if name in selected_traces:
            y_plot = -y_data if is_neg and graph_type == "Stacked Area" else y_data
            # In line mode, maybe keep everything positive? Or keep sign?
            # Usually demand is shown as negative in stacked, but in line comparison often positive.
            # Let's keep signs for consistency with stacked view logic unless specified.
            # User requirement: "Option to show a non-stacked graph... with individual selection".
            
            # If line mode, remove stackgroup and fill (optional)
            sg = stackgroup if graph_type == "Stacked Area" else None
            fl = fill if graph_type == "Stacked Area" else None
            
            fig_bal.add_trace(go.Scatter(
                x=df_sim.index, y=y_plot,
                mode='lines', name=name, stackgroup=sg,
                line=dict(width=1 if graph_type == "Line (Non-stacked)" else 0, color=color), 
                fillcolor=fl
            ))

    # Add traces using helper
    add_trace_if_selected("Solar", df_sim["Solar_Gen"], '#FFB300', 'pos', 'rgba(255, 179, 0, 0.7)')
    add_trace_if_selected("Wind Onshore", df_sim["Wind_Onshore_Gen"], '#43A047', 'pos', 'rgba(67, 160, 71, 0.7)')
    add_trace_if_selected("Wind Offshore", df_sim["Wind_Offshore_Gen"], '#1E88E5', 'pos', 'rgba(30, 136, 229, 0.7)')
    add_trace_if_selected("Batt Discharge", df_sim["Batt_Discharge"], '#6D4C41', 'pos', 'rgba(109, 76, 65, 0.7)')
    add_trace_if_selected("H2 Discharge", df_sim["H2_Discharge"], '#8E24AA', 'pos', 'rgba(142, 36, 170, 0.7)')
    
    # Negatives
    add_trace_if_selected("Batt Charge", df_sim["Batt_Charge"], '#8D6E63', 'neg', 'rgba(141, 110, 99, 0.5)', is_neg=True)
    add_trace_if_selected("H2 Charge", df_sim["H2_Charge"], '#BA68C8', 'neg', 'rgba(186, 104, 200, 0.5)', is_neg=True)
    add_trace_if_selected("Curtailment", df_sim["Curtailment"], '#EF5350', 'neg', 'rgba(239, 83, 80, 0.5)', is_neg=True)
    
    if "Baseload Target" in selected_traces:
        fig_bal.add_trace(go.Scatter(
            x=df_sim.index, y=[baseload_target]*len(df_sim),
            mode='lines', name='Baseload Target',
            line=dict(color='black', width=3, dash='dash')
        ))

    fig_bal.update_layout(
        title="Power Balance (GW)",
        xaxis_title="Time",
        yaxis_title="Power (GW)",
        height=500,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        template="plotly_white",
        hovermode="x unified"
    )
    # Option 4: Download PNG is enabled by default in modebar
    st.plotly_chart(fig_bal, width="stretch", config={'displayModeBar': True, 'toImageButtonOptions': {'filename': 'power_balance'}})
    
    # 3. Graph 2: Storage Levels
    # Linked time window is automatic because we filter df_sim based on start/end date inputs.
    
    fig_soc = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_soc.add_trace(
        go.Scatter(x=df_sim.index, y=df_sim["Batt_SoC"], name="Battery SoC", line=dict(color='#6D4C41')),
        secondary_y=False
    )
    fig_soc.add_trace(
        go.Scatter(x=df_sim.index, y=df_sim["H2_SoC"], name="H2 SoC", line=dict(color='#8E24AA', dash='dot')),
        secondary_y=True
    )
    
    fig_soc.update_layout(
        title="Storage Levels (%)",
        xaxis_title="Time",
        height=400,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        template="plotly_white",
        hovermode="x unified"
    )
    fig_soc.update_yaxes(title_text="Battery SoC (%)", secondary_y=False, range=[0, 105])
    fig_soc.update_yaxes(title_text="H2 SoC (%)", secondary_y=True, range=[0, 105])
    
    st.plotly_chart(fig_soc, width="stretch", config={'displayModeBar': True, 'toImageButtonOptions': {'filename': 'storage_levels'}})

if __name__ == "__main__":
    main()
