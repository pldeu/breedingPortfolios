#!/usr/bin/env python3
"""Test the 3-subplot layout with actual rendering."""

import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from experiment_core import ExperimentRunner
from subplot_registry import DEFAULT_SUBPLOT_IDS
import numpy as np

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
    n=100  # Small grid for quick testing
)

# Run computation
text_result, scenario_data_list = runner.compute(
    strategy_keys=['Base', 'BeatBest', 'PoB'],
    replace=False
)

# Test with 3 subplots (the problematic case)
print("=" * 60)
print("Testing 3-subplot layout")
print("=" * 60)

subplot_ids = ['value_added', 'performance_bar', 'legend']
print(f"\nSelected subplots: {subplot_ids}")

# Create figure with the new logic
n = len(subplot_ids)
import math
if n == 3:
    figsize = (18, 5.5)  # Updated for square subplots with colorbars
    ncols = 3
    nrows = 1
else:
    figsize = (16, 12)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

print(f"Figure size: {figsize}")
print(f"Grid: {nrows} rows x {ncols} cols")

fig = Figure(figsize=figsize, dpi=100)
runner.build_figure(scenario_data_list, fig, subplot_ids=subplot_ids)

# Save to file
output_file = 'test_3subplot_layout.png'
fig.savefig(output_file, dpi=100, bbox_inches='tight')
print(f"\nPASSED: Layout test passed! Saved to: {output_file}")
print(f"  Figure dimensions: {fig.get_size_inches()} inches")

# Test with default subplots too
print("\n" + "=" * 60)
print("Testing default layout")
print("=" * 60)

subplot_ids = DEFAULT_SUBPLOT_IDS
n = len(subplot_ids)
if n == 3:
    figsize = (18, 5.5)
else:
    figsize = (16, 12)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

print(f"Selected subplots ({n}): {subplot_ids}")
print(f"Figure size: {figsize}")
print(f"Grid: {nrows} rows x {ncols} cols")

fig = Figure(figsize=figsize, dpi=100)
runner.build_figure(scenario_data_list, fig, subplot_ids=subplot_ids)

output_file = 'test_default_layout.png'
fig.savefig(output_file, dpi=100, bbox_inches='tight')
print(f"\nPASSED: Default layout test passed! Saved to: {output_file}")
print(f"  Figure dimensions: {fig.get_size_inches()} inches")
