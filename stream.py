import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Import your actual classes ---
# Ensure these files are in the same folder as this script
try:
    from Jacob_experiment_base4_gui import ExperimentRunner
    from Jacob_experiment_case_gen import ExperimentSweepRunner
except ImportError:
    st.error("Could not import custom modules (Jacob_experiment_base4_gui / Jacob_experiment_case_gen). Please ensure they are in the app directory.")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Breeding Strategy Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Data Loading & Caching ---
@st.cache_data
def load_data(filename='results.csv'):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def run_diversity_selection(df, k):
    """
    Selects K diverse scenarios. 
    Cached so it doesn't re-run unless K or the Dataframe changes.
    """
    if df is None:
        return None
    # Assuming ExperimentSweepRunner is stateless or lightweight enough to init here
    sweep = ExperimentSweepRunner(sweep_size=1000000, random_state=42)
    df_selected = sweep.select_diverse_scenarios(df, k=k, method='greedy')
    return df_selected

# --- 2. State Management Callbacks ---
def load_scenario_callback():
    """
    Called when the User changes the 'Scenario Index' slider.
    Updates the session state for parameters (p, c, gamma, etc.)
    """
    idx = st.session_state.scen_idx
    df_selected = st.session_state.get('df_selected')
    
    if df_selected is not None and 0 <= idx < len(df_selected):
        row = df_selected.iloc[int(idx)]
        
        # Update Session State keys directly
        st.session_state.p_var = float(row['p'])
        st.session_state.c_var = float(row['c'])
        st.session_state.gamma_var = float(row['gamma'])
        st.session_state.rg_var = float(row['r_g'])
        st.session_state.R_var = float(row['R'])
        
        st.session_state.g_fixed_x = float(row['g_fix_1'])
        st.session_state.g_fixed_y = float(row['g_fix_2'])
        st.session_state.g_mut_x = float(row['g_mut_1'])
        st.session_state.g_mut_y = float(row['g_mut_2'])

# --- 3. Sidebar UI ---
with st.sidebar:
    st.header("Parameter Controls")
    
    # -- Manual Inputs (Initialized with defaults if state doesn't exist) --
    # Note: We use 'key' to bind them to session_state so the callback can update them.
    
    st.subheader("Current Parameters")
    p = st.slider("Prob (p)", 0.0, 1.0, 0.5, 0.01, key="p_var")
    c = st.slider("Env. con. (c)", 0.0, 1.0, 0.1, 0.01, key="c_var")
    gamma = st.slider("Gamma", 0.0, 1000.0, 21.0, 1.0, key="gamma_var")
    rg = st.slider("Gen Corr (r_g)", -1.0, 1.0, -0.7, 0.05, key="rg_var")
    R = st.slider("R", 0.0, 1.0, 0.1, 0.1, key="R_var")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fixed Asset**")
        gfx = st.slider("X", 0.0, 1.0, 0.6, 0.05, key="g_fixed_x")
        gfy = st.slider("Y", 0.0, 1.0, 0.2, 0.05, key="g_fixed_y")
    with col2:
        st.markdown("**Mutable Asset**")
        gmx = st.slider("X", 0.0, 1.0, 0.1, 0.05, key="g_mut_x")
        gmy = st.slider("Y", 0.0, 1.0, 0.2, 0.05, key="g_mut_y")

    st.markdown("---")

    # -- Scenario Loader --
    st.subheader("Scenario Selection")
    
    # Load Data
    df_full = load_data('results.csv')
    
    if df_full is not None:
        k_val = st.slider("Diversity Count (K)", 2, 50, 20)
        
        # Run Selection (Cached)
        df_selected = run_diversity_selection(df_full, k_val)
        
        # Store in session state so the callback can access it
        st.session_state['df_selected'] = df_selected
        
        # Scenario Selector
        # IMPORTANT: on_change=load_scenario_callback triggers the update of the sliders above
        max_idx = len(df_selected) - 1
        st.slider(
            "Select Scenario Index", 
            0, max_idx, 0, 
            key="scen_idx", 
            on_change=load_scenario_callback
        )
        
        # Show details of selected scenario
        current_idx = st.session_state.get('scen_idx', 0)
        if 0 <= current_idx < len(df_selected):
            row_info = df_selected.iloc[current_idx]
            orig_id = row_info.name if hasattr(row_info, 'name') else "-"
            st.info(f"Selected Item {current_idx+1}/{len(df_selected)} (Orig Row: {orig_id})")
    else:
        st.error("results.csv not found.")

# --- 4. Main Execution Area ---

# Tabs for layout
tab_report, tab_viz = st.tabs(["📄 Text Report", "📊 Visualizations"])

# Prepare Data
gf = np.array([st.session_state.g_fixed_x, st.session_state.g_fixed_y])
gm = np.array([st.session_state.g_mut_x, st.session_state.g_mut_y])

label_txt = f"Scenario {st.session_state.get('scen_idx', 'Custom')}"
scenario_pairs = [{"label": label_txt, "g_fixed": gf, "g_mutable": gm}]

# Run Simulation
# We put this in a spinner so the UI shows activity
with st.spinner("Simulating..."):
    try:
        runner = ExperimentRunner(
            p=st.session_state.p_var, 
            c=st.session_state.c_var, 
            gamma=st.session_state.gamma_var, 
            r_g=st.session_state.rg_var, 
            R=st.session_state.R_var, 
            scenario_pairs=scenario_pairs, 
            replace=False, 
            n=201
        )
        
        # Note: We assume runner.run returns (text_string, matplotlib_figure)
        text_result, fig = runner.run(
            plot_extensive=True, 
            easy_base=False, 
            no_plot=False, 
            plot_right=True
        )

        with tab_viz:
            if fig:
                st.pyplot(fig)
            else:
                st.warning("No plot generated.")

        with tab_report:
            st.text_area("Output", text_result, height=400, disabled=True)

    except Exception as e:
        st.error(f"Simulation Error: {e}")