import sys
import builtins

if not hasattr(builtins, "help"):
    def _dummy_help(*args, **kwargs):
        pass
    builtins.help = _dummy_help
    
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading

# --- Import your actual classes ---
from Jacob_experiment_base4_gui import ExperimentRunner
from Jacob_experiment_case_gen import ExperimentSweepRunner


class BreedingExperimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breeding Strategy Simulator - Scenario Selector")
        
        # Start Full Screen
        try:
            self.root.state('zoomed')
        except:
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            self.root.geometry(f"{w}x{h}+0+0")

        self.update_timer = None
        
        # Data Containers
        self.df_full = None      # The raw CSV data
        self.df_selected = None  # The K selected rows
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        self.sidebar = ttk.Frame(self.paned_window, width=350, relief=tk.RIDGE)
        self.paned_window.add(self.sidebar, weight=1)
        
        self.create_inputs()

        # Main Area
        self.main_area = ttk.Frame(self.paned_window, relief=tk.RIDGE)
        self.paned_window.add(self.main_area, weight=4)

        self.notebook = ttk.Notebook(self.main_area)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Text
        self.tab_report = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_report, text=" 📄 Text Report ")
        self.txt_output = scrolledtext.ScrolledText(self.tab_report, font=("Consolas", 15))
        self.txt_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 2: Visuals
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text=" 📊 Visualizations ")
        self.canvas_frame = tk.Frame(self.tab_plot)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = None

        # --- INITIALIZATION ---
        # Load Data immediately on startup
        self.load_fixed_data()

    def create_inputs(self):
        # --- FIX: Create the label object FIRST so callbacks can find it ---
        self.lbl_status = ttk.Label(self.sidebar, text="Ready", foreground="gray")
        
        row = 0
        
        # --- BLOCK 1: CURRENT PARAMETERS ---
        lbl_title = ttk.Label(self.sidebar, text="Current Parameters", font=("Helvetica", 12, "bold"))
        lbl_title.grid(row=row, column=0, columnspan=2, pady=(10, 5), sticky="ew")
        row += 1

        def add_slider(label, var_name, min_val, max_val, step, default):
            nonlocal row
            ttk.Label(self.sidebar, text=label).grid(row=row, column=0, sticky="w", padx=10)
            
            val_var = tk.DoubleVar(value=default)
            setattr(self, var_name, val_var)
            
            val_lbl = ttk.Label(self.sidebar, text=f"{default:.2f}")
            val_lbl.grid(row=row, column=1, sticky="e", padx=10)
            setattr(self, f"lbl_{var_name}", val_lbl)

            def on_manual_slide(v):
                val_lbl.config(text=f"{float(v):.2f}")
                self.schedule_update()

            scale = ttk.Scale(self.sidebar, from_=min_val, to=max_val, variable=val_var, command=on_manual_slide)
            scale.grid(row=row+1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
            setattr(self, f"scale_{var_name}", scale)
            row += 2

        add_slider("Prob (p)", "p_var", 0.0, 1.0, 0.01, 0.5)
        add_slider("Env. con. (c)", "c_var", 0.0, 1.0, 0.01, 0.1)
        add_slider("Gamma", "gamma_var", 0.0, 1000.0, 1.0, 21.0)
        add_slider("Gen Corr (r_g)", "rg_var", -1.0, 1.0, 0.05, -0.7)
        add_slider("R", "R_var", 0.0, 1.0, 0.1, 0.1)
        
        ttk.Separator(self.sidebar, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        add_slider("Fixed Asset X", "g_fixed_x", -1.0, 1.0, 0.05, 0.6)
        add_slider("Fixed Asset Y", "g_fixed_y", -1.0, 1.0, 0.05, 0.2)
        add_slider("Mutable Asset X", "g_mut_x", -1.0, 1.0, 0.05, -0.1)
        add_slider("Mutable Asset Y", "g_mut_y", -1.0, 1.0, 0.05, 0.2)

        ttk.Separator(self.sidebar, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
        row += 1

        # --- BLOCK 2: SCENARIO SELECTION ---
        lbl_head = ttk.Label(self.sidebar, text="Scenario Selection", font=("Helvetica", 12, "bold"))
        lbl_head.grid(row=row, column=0, columnspan=2, pady=(5, 5), sticky="ew")
        row += 1

        # K Slider
        ttk.Label(self.sidebar, text="Diversity Count (K)").grid(row=row, column=0, sticky="w", padx=10)
        self.lbl_k = ttk.Label(self.sidebar, text="20")
        self.lbl_k.grid(row=row, column=1, sticky="e", padx=10)
        
        self.scale_k = ttk.Scale(self.sidebar, from_=2, to=50, command=self.on_k_change)
        self.scale_k.set(20) # This triggers the callback! Now safe because lbl_status exists.
        self.scale_k.grid(row=row+1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        row += 2

        # Index Slider
        ttk.Label(self.sidebar, text="Select Scenario Index").grid(row=row, column=0, sticky="w", padx=10)
        self.lbl_idx = ttk.Label(self.sidebar, text="0")
        self.lbl_idx.grid(row=row, column=1, sticky="e", padx=10)
        
        self.scale_idx = ttk.Scale(self.sidebar, from_=0, to=19, command=self.on_index_change)
        self.scale_idx.set(0)
        self.scale_idx.grid(row=row+1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        row += 2

        self.lbl_scen_info = ttk.Label(self.sidebar, text="ID: -", foreground="blue")
        self.lbl_scen_info.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # --- BLOCK 3: STATUS LABEL (Place it now) ---
        # We created it at the top, now we place it at the bottom
        self.lbl_status.grid(row=row, column=0, columnspan=2, pady=20)
        

    # --- DATA & LOGIC ---

    def load_fixed_data(self):
        filename = 'results.csv'
        try:
            self.df_full = pd.read_csv(filename)
            self.run_diversity_selection(k=20)
            
        except FileNotFoundError:
            messagebox.showerror("Error", f"File '{filename}' not found in directory.\nPlease ensure it exists.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load data:\n{e}")

    def run_diversity_selection(self, k):
        """Implements Greedy Max-Min Diversity Selection locally."""
        if self.df_full is None:
            return

        sweep = ExperimentSweepRunner(sweep_size=1000000, random_state=42)
        
        self.df_selected = sweep.select_diverse_scenarios(self.df_full, k=k, method='greedy')

        # Update GUI controls based on new K
        self.scale_idx.config(to=len(self.df_selected)-1)
        self.scale_idx.set(0) # Reset to first element
        self.lbl_k.config(text=f"{int(k)}")
        
        # Load the first scenario of the new set
        self.load_scenario_by_index(0)

    def on_k_change(self, val):
        """Called when K slider moves."""
        k = int(float(val))
        self.lbl_k.config(text=str(k))
        
        # 1. Update Status Immediately
        self.lbl_status.config(text="Waiting (Selection)...", foreground="orange")
        
        # 2. Debounce
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(300, lambda: self.run_diversity_selection(k))

    def on_index_change(self, val):
        """Called when Index slider moves."""
        idx = int(float(val))
        self.lbl_idx.config(text=str(idx))
        
        if self.df_selected is None:
            return

        # 1. Update Status Immediately
        self.lbl_status.config(text="Waiting (Loading)...", foreground="orange")
            
        # 2. Update sliders visually immediately (optional but feels snappier)
        self.load_scenario_by_index(idx, trigger_run=False)
        
        # 3. Debounce the heavy simulation run
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(300, self.start_worker_thread)

    def load_scenario_by_index(self, idx, trigger_run=True):
        """Updates global parameter sliders based on the selected row."""
        if self.df_selected is None or idx >= len(self.df_selected):
            return

        row = self.df_selected.iloc[int(idx)]
        
        # Update Info Label
        orig_idx = row.name if hasattr(row, 'name') else "-"
        self.lbl_scen_info.config(text=f"Selected Item {int(idx)+1}/{len(self.df_selected)} (Orig Row: {orig_idx})")

        # Helper to update GUI slider
        def set_val(name, val):
            if hasattr(self, name):
                getattr(self, name).set(val)
                getattr(self, f"lbl_{name}").config(text=f"{val:.2f}")

        # Map Columns
        set_val('p_var', row['p'])
        set_val('c_var', row['c'])
        set_val('gamma_var', row['gamma'])
        set_val('rg_var', row['r_g'])
        set_val('R_var', row['R'])
        
        set_val('g_fixed_x', row['g_fix_1'])
        set_val('g_fixed_y', row['g_fix_2'])
        set_val('g_mut_x', row['g_mut_1'])
        set_val('g_mut_y', row['g_mut_2'])

        if trigger_run:
            self.schedule_update()

    # --- SIMULATION RUNNER ---
    def schedule_update(self):
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.lbl_status.config(text="Waiting...", foreground="orange")
        self.update_timer = self.root.after(300, self.start_worker_thread)

    def start_worker_thread(self):
        self.lbl_status.config(text="Computing...", foreground="blue")
        thread = threading.Thread(target=self.run_experiment_logic)
        thread.start()

    def run_experiment_logic(self):
        try:
            p_val = self.p_var.get()
            c_val = self.c_var.get()
            g_val = self.gamma_var.get()
            rg_val = self.rg_var.get()
            R_val = self.R_var.get()
            
            gf = np.array([self.g_fixed_x.get(), self.g_fixed_y.get()])
            gm = np.array([self.g_mut_x.get(), self.g_mut_y.get()])

            label_txt = f"Scenario {int(self.scale_idx.get())}"

            scenario_pairs = [{"label": label_txt, "g_fixed": gf, "g_mutable": gm}]

            runner = ExperimentRunner(
                p=p_val, c=c_val, gamma=g_val, r_g=rg_val, R=R_val, 
                scenario_pairs=scenario_pairs, replace=False, n=201
            )
            text_result, fig = runner.run(
                plot_extensive=True, 
                easy_base=False, no_plot=False, plot_right=True
            )

            self.root.after(0, self.update_results, text_result, fig)

        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def update_results(self, text_result, fig):
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, text_result)
        
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        if fig:
            self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
            toolbar.update()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lbl_status.config(text="Done", foreground="green")

    def show_error(self, msg):
        self.lbl_status.config(text="Error", foreground="red")
        print(msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = BreedingExperimentApp(root)
    root.mainloop()