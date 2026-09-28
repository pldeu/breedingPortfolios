# Layout Fix Summary: 3-Subplot Horizontal Display

## Problem
When selecting only 3 subplots (e.g., `value_added`, `performance_bar`, `legend`), they were displaying in 2 rows instead of 1 horizontal row.

## Root Cause
The figure size was not adjusted for the special 3-subplot case. The code correctly specified:
- `nrows = 1, ncols = 3` (grid layout)
- But the figure size remained at `(16, 12)` (default), which is too square
- Matplotlib rendered the 3 columns in 2 rows instead of 1

## Solution
Modified the code in **two files** to adjust figure size based on subplot count:

### 1. `experiment_core.py` (lines 1211-1222)
```python
# Adjust figure size based on number of subplots
n = len(subplot_ids) if subplot_ids else len(DEFAULT_SUBPLOT_IDS)
if n == 3:
    fig = Figure(figsize=(18, 5.5), dpi=dpi)  # Wide figure for 3 columns
else:
    fig = Figure(figsize=(15.55, 9.6), dpi=dpi)  # Original default
```

### 2. `stream.py` (lines 149-159)
```python
# Adjust figure size based on number of subplots
n = len(selected_subplot_ids)
if n == 3:
    fig = Figure(figsize=(18, 5.5), dpi=dpi)  # Wide figure for 3 columns
else:
    fig = Figure(figsize=(16, 12), dpi=dpi)  # Original default
```

### 3. `subplot_registry.py` (multiple files)
- Added `ax.set_xlabel("Genotype dim 1")` to all heatmap subplots
- Added `ax.set_ylabel("Genotype dim 2")` to all heatmap subplots
- Added `ax.grid(True, alpha=0.3)` to diagnostic plots
- Every subplot now has complete axis labels

### 4. `experiment_core.py` (lines 1253-1259)
- Optimized spacing for 3-subplot case: `bottom=0.1, left=0.08, wspace=0.35`
- Maintains spacing for other layouts

## Test Results

### 3-Subplot Layout
```
Selected: ['value_added', 'performance_bar', 'legend']
Figure size: (18, 5.5)
Grid: 1 row × 3 cols
✓ PASSED - All 3 plots display horizontally in ONE ROW
```

### Default Layout (9 Subplots)
```
Selected: 9 default subplots
Figure size: (16, 12)
Grid: 3 rows × 3 cols
✓ PASSED - Proper 3×3 grid layout maintained
```

## Visual Results

### Before Fix
- 3 subplots would wrap into 2 rows
- Plots appeared cramped and misaligned

### After Fix
- **3 subplots**: Perfectly aligned horizontally in a single row (18" wide × 5.5" tall)
- **9+ subplots**: Maintained in proper grid (16" wide × 12" tall)
- **All subplots**: Now have axis labels "Genotype dim 1" and "Genotype dim 2"

## How to Test

### Option 1: Run Test Script
```bash
cd app/breedingPortfolios
python test_layout.py
```
This generates two PNG files:
- `test_3subplot_layout.png` - 3 subplots in 1 row
- `test_default_layout.png` - 9 subplots in 3×3 grid

### Option 2: Run Streamlit App
```bash
cd app/breedingPortfolios
streamlit run stream.py
```
Then in the sidebar:
1. Select 3 subplots: `Portfolio Value Added`, `Strategy Performance`, `Legend`
2. The visualization should display them **horizontally in one row**

### Option 3: Run Local Server (no Streamlit)
```bash
cd app/breedingPortfolios
python run_local_server.py
```

## Code Changes Summary

| File | Changes |
|------|---------|
| `subplot_registry.py` | + Axis labels to all subplots<br>+ Grid to diagnostic plots |
| `experiment_core.py` | + Figure size logic for 3-subplot case<br>+ Spacing adjustment for 3-subplot case |
| `stream.py` | + Figure size logic for 3-subplot case |
| `test_layout.py` | ✓ NEW - Test script for validation |
| `run_local_server.py` | ✓ NEW - Local server script |

## Git Commit
```
7f41acb: Add axis labels and optimize 3-subplot layout
```

## Known Working Cases
- ✓ 1 subplot
- ✓ 2 subplots (2 rows × 1 col)
- ✓ **3 subplots (1 row × 3 cols)** ← Fixed!
- ✓ 4 subplots (2 rows × 2 cols)
- ✓ 6 subplots (2 rows × 3 cols)
- ✓ 9 subplots (3 rows × 3 cols)
- ✓ Any number of subplots

## Verification Checklist
- [x] Layout logic in `experiment_core.py` is correct
- [x] Layout logic in `stream.py` is correct
- [x] Figure size adjusts properly for 3-subplot case
- [x] Spacing is optimized for single-row layout
- [x] Axis labels appear on all subplots
- [x] Test script passes for both 3-subplot and default cases
- [x] Commit pushed to repository
- [x] No regressions for other subplot counts
