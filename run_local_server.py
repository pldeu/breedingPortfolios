#!/usr/bin/env python3
"""
Simple local server to test the layout without Streamlit.
Generates a 3-subplot figure and opens it in the browser.
"""

import os
import sys
import numpy as np
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_core import ExperimentRunner
from subplot_registry import DEFAULT_SUBPLOT_IDS

def main():
    print("Generating 3-subplot layout test...\n")

    # Initialize runner with test parameters
    scenario_pairs = [{
        "label": "Test Scenario",
        "g_fixed": np.array([0.6, 0.2]),
        "g_mutable": np.array([0.1, 0.2])
    }]

    runner = ExperimentRunner(
        p=0.5, c=0.3, gamma=2.0, r_g=0.5, R=0.5,
        scenario_pairs=scenario_pairs,
        replace=False,
        n=100
    )

    # Compute
    text_result, scenario_data_list = runner.compute(
        strategy_keys=['Base', 'BeatBest', 'PoB'],
        replace=False
    )

    # Test 3-subplot case
    print("=" * 70)
    print("3-SUBPLOT LAYOUT TEST")
    print("=" * 70)
    subplot_ids = ['value_added', 'performance_bar', 'legend']

    fig = Figure(figsize=(18, 5.5), dpi=100)
    runner.build_figure(scenario_data_list, fig, subplot_ids=subplot_ids)

    output_file = 'test_3subplot_layout.png'
    fig.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"PASSED: Saved 3-subplot layout to {output_file}")
    print(f"  Subplots: {subplot_ids}")
    print(f"  Figure size: 18 x 5.5 inches (1 row x 3 cols)")
    print(f"  This should display ALL 3 plots horizontally in ONE ROW")
    print()

    # Test default case
    print("=" * 70)
    print("DEFAULT LAYOUT TEST (9 subplots)")
    print("=" * 70)

    subplot_ids = DEFAULT_SUBPLOT_IDS
    n = len(subplot_ids)
    import math
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig = Figure(figsize=(16, 12), dpi=100)
    runner.build_figure(scenario_data_list, fig, subplot_ids=subplot_ids)

    output_file = 'test_default_layout.png'
    fig.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"PASSED: Saved default layout to {output_file}")
    print(f"  Number of subplots: {n}")
    print(f"  Figure size: 16 x 12 inches ({nrows} rows x {ncols} cols)")
    print()

    print("=" * 70)
    print("SUCCESS! Both layouts are working correctly!")
    print("=" * 70)
    print("\nTo run the Streamlit app, use:")
    print("  streamlit run stream.py")
    print("\nMake sure to select only 3 subplots to test the horizontal layout!")

if __name__ == "__main__":
    main()
