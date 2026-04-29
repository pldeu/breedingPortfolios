import os
import sys
import io
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

# Ensure local modules are importable when running with `streamlit run`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from experiment_core import ExperimentRunner, STRATEGY_REGISTRY, DEFAULT_STRATEGY_KEYS
    from subplot_registry import SUBPLOT_REGISTRY, DEFAULT_SUBPLOT_IDS
    from Jacob_experiment_case_gen import ExperimentSweepRunner
except ImportError as e:
    st.error(f"Could not import custom modules: {e}")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="Breeding Strategy Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- 1. Data Loading & Caching ---
@st.cache_data
def load_data(filename="results.csv"):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return None


@st.cache_data
def run_diversity_selection(df, k):
    if df is None:
        return None
    sweep = ExperimentSweepRunner(sweep_size=1_000_000, random_state=42)
    return sweep.select_diverse_scenarios(df, k=k, method="greedy")


# --- 2. State Management Callbacks ---
def load_scenario_callback():
    idx = st.session_state.scen_idx
    df_selected = st.session_state.get("df_selected")
    if df_selected is not None and 0 <= idx < len(df_selected):
        row = df_selected.iloc[int(idx)]
        st.session_state.p_var = float(row["p"])
        st.session_state.c_var = float(row["c"])
        st.session_state.gamma_var = float(row["gamma"])
        st.session_state.rg_var = float(row["r_g"])
        st.session_state.R_var = float(row["R"])
        st.session_state.g_fixed_x = float(row["g_fix_1"])
        st.session_state.g_fixed_y = float(row["g_fix_2"])
        st.session_state.g_mut_x = float(row["g_mut_1"])
        st.session_state.g_mut_y = float(row["g_mut_2"])


# --- 3. Sidebar UI ---
with st.sidebar:
    st.header("Parameter Controls")

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
    # -- Strategy Selection --
    st.subheader("Strategy Selection")
    all_strategy_labels = {v["label"]: k for k, v in STRATEGY_REGISTRY.items()}
    default_labels = [STRATEGY_REGISTRY[k]["label"] for k in DEFAULT_STRATEGY_KEYS
                      if k in STRATEGY_REGISTRY]
    selected_strategy_labels = st.multiselect(
        "Strategies to run",
        options=list(all_strategy_labels.keys()),
        default=default_labels,
    )
    selected_strategy_keys = [all_strategy_labels[l] for l in selected_strategy_labels]

    # -- Subplot Selection --
    st.subheader("Subplot Selection")
    all_subplot_labels = {v.label: k for k, v in SUBPLOT_REGISTRY.items()}
    default_subplot_labels = [SUBPLOT_REGISTRY[k].label for k in DEFAULT_SUBPLOT_IDS
                               if k in SUBPLOT_REGISTRY]
    selected_subplot_labels = st.multiselect(
        "Subplots to show",
        options=list(all_subplot_labels.keys()),
        default=default_subplot_labels,
    )
    selected_subplot_ids = [all_subplot_labels[l] for l in selected_subplot_labels]

    # -- Simulation Options --
    st.subheader("Simulation Options")
    replace = st.toggle("Replace", value=False)

    # -- Visualization Options --
    st.subheader("Visualization")
    marker_alpha = st.slider("Marker opacity", 0.1, 1.0, 0.55, 0.05)
    dpi = st.slider("Figure DPI", 50, 1000, 100, step=50)


# --- 4. Main Execution Area ---
tab_report, tab_viz = st.tabs(["📄 Text Report", "📊 Visualizations"])

gf = np.array([st.session_state.g_fixed_x, st.session_state.g_fixed_y])
gm = np.array([st.session_state.g_mut_x, st.session_state.g_mut_y])
label_txt = f"Scenario {st.session_state.get('scen_idx', 'Custom')}"
scenario_pairs = [{"label": label_txt, "g_fixed": gf, "g_mutable": gm}]

if not selected_strategy_keys:
    st.warning("Select at least one strategy in the sidebar.")
    st.stop()

if not selected_subplot_ids:
    st.warning("Select at least one subplot in the sidebar.")
    st.stop()

with st.spinner("Simulating…"):
    try:
        runner = ExperimentRunner(
            p=st.session_state.p_var,
            c=st.session_state.c_var,
            gamma=st.session_state.gamma_var,
            r_g=st.session_state.rg_var,
            R=st.session_state.R_var,
            scenario_pairs=scenario_pairs,
            replace=replace,
            n=401,
        )

        text_result, scenario_data_list = runner.compute(
            strategy_keys=selected_strategy_keys,
            replace=replace
        )

        fig = Figure(figsize=(16, 12), dpi=dpi)
        runner.build_figure(
            scenario_data_list, fig,
            subplot_ids=selected_subplot_ids,
            marker_alpha=marker_alpha,
        )

        with tab_report:
            st.text_area("Output", text_result, height=400, disabled=True)

        with tab_viz:
            if scenario_data_list:
                # 1. Use BytesIO for binary formats like PNG
                buf = io.BytesIO() 
                fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
                
                # 2. Seek to the start of the buffer so Streamlit can read it
                buf.seek(0)
                
                # 3. Display the image
                st.image(buf, width='stretch')

                with st.sidebar:
                    st.markdown("---")
                    st.download_button(
                        label="Download figure (PNG)",
                        data=buf.getvalue(),
                        file_name="breeding_portfolio.png",
                        mime="image/png",
                    )
            else:
                st.warning("No plot generated.")

    except Exception as e:
        st.error(f"Simulation Error: {e}")
        raise
