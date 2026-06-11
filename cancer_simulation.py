"""
╔══════════════════════════════════════════════════════════════════════════╗
║       CANCER CELL GROWTH PREDICTION via Conway's Game of Life          ║
║       Multi-State Cellular Automaton | Tumor Evolution Simulator       ║
╚══════════════════════════════════════════════════════════════════════════╝

Cell States:
  0 - Empty / ECM         (black)
  1 - Healthy Cell        (green)
  2 - Cancer Cell         (red)
  3 - Resistant Cancer    (magenta)
  4 - Necrotic Core       (gray)
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import random
import math

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
EMPTY    = 0
HEALTHY  = 1
CANCER   = 2
RESISTANT = 3
NECROTIC = 4

COLORS = {
    EMPTY:     "#0a0a1a",
    HEALTHY:   "#00e676",
    CANCER:    "#ff1744",
    RESISTANT: "#e040fb",
    NECROTIC:  "#546e7a",
}

# Colormap for imshow
CMAP = ListedColormap([
    COLORS[EMPTY],
    COLORS[HEALTHY],
    COLORS[CANCER],
    COLORS[RESISTANT],
    COLORS[NECROTIC],
])

BG_DARK    = "#0d0d1f"
BG_PANEL   = "#12122a"
BG_CARD    = "#1a1a3a"
ACCENT1    = "#ff1744"
ACCENT2    = "#00e676"
ACCENT3    = "#e040fb"
TEXT_MAIN  = "#e8eaf6"
TEXT_DIM   = "#7986cb"
BTN_PLAY   = "#00c853"
BTN_PAUSE  = "#ff6d00"
BTN_STEP   = "#0288d1"
BTN_RESET  = "#546e7a"
BTN_CHEMO  = "#aa00ff"
BTN_RADIO  = "#ff6d00"

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class CancerAutomaton:
    def __init__(self, grid_size=80):
        self.N = grid_size
        self.grid = np.zeros((self.N, self.N), dtype=np.int8)
        self.generation = 0
        self.history = {HEALTHY: [], CANCER: [], RESISTANT: [], NECROTIC: [], EMPTY: []}

        # Tunable parameters
        self.cancer_birth_thresh  = 3   # neighbors needed to birth a cancer cell
        self.cancer_overpop_thresh = 6  # cancer dies above this many neighbors
        self.mutation_rate        = 0.02
        self.healthy_invasion_thresh = 3  # cancer neighbors that kill a healthy cell

    def count_neighbors(self, grid, state):
        """Count how many neighbors of each cell equal `state` using conv."""
        mask = (grid == state).astype(np.int16)
        count = np.zeros_like(mask)
        count[1:,  :]  += mask[:-1, :]
        count[:-1, :]  += mask[1:,  :]
        count[:,  1:]  += mask[:,  :-1]
        count[:, :-1]  += mask[:,  1:]
        count[1:,  1:] += mask[:-1, :-1]
        count[:-1, 1:] += mask[1:,  :-1]
        count[1:, :-1] += mask[:-1, 1:]
        count[:-1,:-1] += mask[1:,  1:]
        return count

    def count_neighbors_multi(self, grid, states):
        """Count neighbors that are any of the given states."""
        total = np.zeros((self.N, self.N), dtype=np.int16)
        for s in states:
            total += self.count_neighbors(grid, s)
        return total

    def step(self):
        g = self.grid.copy()
        new = np.zeros_like(g)

        n_healthy   = self.count_neighbors(g, HEALTHY)
        n_cancer    = self.count_neighbors(g, CANCER)
        n_resistant = self.count_neighbors(g, RESISTANT)
        n_necrotic  = self.count_neighbors(g, NECROTIC)
        n_cancer_like = n_cancer + n_resistant
        n_all_cells   = n_healthy + n_cancer_like + n_necrotic

        rand = np.random.random((self.N, self.N))

        # ── HEALTHY cells ──────────────────────────────────────────────────
        alive_healthy = (g == HEALTHY)
        # Survival: 2-4 healthy neighbours (contact inhibition / homeostasis)
        survive_h = alive_healthy & (n_healthy >= 2) & (n_healthy <= 4)
        # Invaded by cancer
        invaded    = alive_healthy & (n_cancer_like >= self.healthy_invasion_thresh)
        # Birth: empty cell with exactly 3 healthy neighbours
        born_h = (g == EMPTY) & (n_healthy == 3) & (n_cancer_like == 0)

        new[survive_h & ~invaded] = HEALTHY
        new[born_h] = HEALTHY

        # ── CANCER cells ──────────────────────────────────────────────────
        alive_cancer = (g == CANCER)
        # Survival: 1-6 total neighbours
        survive_c = alive_cancer & (n_all_cells >= 1) & (n_all_cells <= self.cancer_overpop_thresh)
        # Necrosis: surrounded (≥ 7 cancer/necrotic neighbours) → necrotic
        necrose = alive_cancer & (n_cancer_like + n_necrotic >= 7)
        # Birth: empty with ≥ threshold cancer neighbours
        born_c = (g == EMPTY) & (n_cancer_like >= self.cancer_birth_thresh)
        # Also replace invaded healthy cells
        invade_h = (g == HEALTHY) & invaded

        # Apply mutation on birth
        mutation_mask = rand < self.mutation_rate

        new[survive_c & ~necrose] = CANCER
        new[necrose] = NECROTIC
        new[born_c & ~mutation_mask] = CANCER
        new[born_c &  mutation_mask] = RESISTANT
        new[invade_h & ~mutation_mask] = CANCER
        new[invade_h &  mutation_mask] = RESISTANT

        # ── RESISTANT cells ───────────────────────────────────────────────
        alive_resistant = (g == RESISTANT)
        survive_r = alive_resistant & (n_all_cells >= 1) & (n_all_cells <= self.cancer_overpop_thresh)
        necrose_r = alive_resistant & (n_cancer_like + n_necrotic >= 7)
        born_r = (g == EMPTY) & (n_resistant >= self.cancer_birth_thresh)

        new[survive_r & ~necrose_r] = RESISTANT
        new[necrose_r] = NECROTIC
        new[born_r] = RESISTANT

        # ── NECROTIC cells ────────────────────────────────────────────────
        alive_necrotic = (g == NECROTIC)
        decay_mask = rand < 0.08
        new[alive_necrotic & ~decay_mask] = NECROTIC
        # decay → empty (already 0)

        self.grid = new
        self.generation += 1
        self._record_history()

    def _record_history(self):
        g = self.grid
        total = self.N * self.N
        for state in [HEALTHY, CANCER, RESISTANT, NECROTIC, EMPTY]:
            self.history[state].append(int(np.sum(g == state)))

    def apply_chemotherapy(self, efficacy=0.70, healthy_damage=0.15):
        """Kill efficacy% of cancer, healthy_damage% of healthy; resistant survive."""
        mask_c = (self.grid == CANCER)  & (np.random.random(self.grid.shape) < efficacy)
        mask_h = (self.grid == HEALTHY) & (np.random.random(self.grid.shape) < healthy_damage)
        self.grid[mask_c] = EMPTY
        self.grid[mask_h] = EMPTY
        self._record_history()

    def apply_radiotherapy(self, efficacy=0.80, healthy_damage=0.25):
        """Radiotherapy: higher efficacy vs cancer but more healthy damage."""
        mask_c = (self.grid == CANCER)  & (np.random.random(self.grid.shape) < efficacy)
        mask_h = (self.grid == HEALTHY) & (np.random.random(self.grid.shape) < healthy_damage)
        self.grid[mask_c] = EMPTY
        self.grid[mask_h] = EMPTY
        self._record_history()

    def reset(self, grid_size=None):
        if grid_size:
            self.N = grid_size
        self.grid = np.zeros((self.N, self.N), dtype=np.int8)
        self.generation = 0
        self.history = {HEALTHY: [], CANCER: [], RESISTANT: [], NECROTIC: [], EMPTY: []}

    # ── PRESETS ─────────────────────────────────────────────────────────────
    def preset_tumor_inception(self):
        """Single cancer cell in healthy tissue."""
        self.reset()
        cx, cy = self.N // 2, self.N // 2
        # Fill with healthy cells (leaving a border)
        self.grid[2:self.N-2, 2:self.N-2] = HEALTHY
        # Place one cancer cell in center
        self.grid[cx, cy] = CANCER
        self._record_history()

    def preset_therapy_relapse(self):
        """Established tumor with some resistant cells – ready for therapy demo."""
        self.reset()
        cx, cy = self.N // 2, self.N // 2
        r_healthy = int(self.N * 0.4)
        r_tumor   = int(self.N * 0.15)

        for x in range(self.N):
            for y in range(self.N):
                d = math.hypot(x - cx, y - cy)
                if d < r_tumor * 0.5:
                    # Necrotic core
                    self.grid[x, y] = NECROTIC if random.random() < 0.7 else CANCER
                elif d < r_tumor:
                    # Active cancer
                    p = random.random()
                    if p < 0.08:
                        self.grid[x, y] = RESISTANT
                    else:
                        self.grid[x, y] = CANCER
                elif d < r_healthy:
                    self.grid[x, y] = HEALTHY if random.random() < 0.9 else EMPTY
        self._record_history()

    def preset_competitive(self):
        """Cancer and healthy cells in competitive patches."""
        self.reset()
        half = self.N // 2
        # Left half: healthy
        self.grid[2:self.N-2, 2:half] = HEALTHY
        # Add noise
        noise = np.random.random((self.N-4, half-2))
        self.grid[2:self.N-2, 2:half][noise < 0.15] = EMPTY
        # Right half: cancer blob
        cx, cy = self.N // 2, int(self.N * 0.65)
        r = int(self.N * 0.12)
        for x in range(self.N):
            for y in range(self.N):
                if math.hypot(x - cx, y - cy) < r:
                    self.grid[x, y] = CANCER if random.random() < 0.85 else RESISTANT
        self._record_history()

    def estimate_tumor_diameter(self):
        """Estimate tumor diameter in cell units."""
        cancer_cells = np.argwhere(np.isin(self.grid, [CANCER, RESISTANT, NECROTIC]))
        if len(cancer_cells) < 2:
            return 0
        cx = cancer_cells[:, 0].mean()
        cy = cancer_cells[:, 1].mean()
        dists = np.sqrt((cancer_cells[:, 0] - cx)**2 + (cancer_cells[:, 1] - cy)**2)
        return round(float(dists.max() * 2), 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class CancerSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Cancer Cell Growth Prediction — Conway's Game of Life")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1440x860")
        self.root.minsize(1100, 700)

        self.sim = CancerAutomaton(grid_size=80)
        self.sim.preset_tumor_inception()

        self.running   = False
        self.speed_ms  = 80
        self.after_id  = None
        self._build_ui()
        self._update_display()

    # ── UI CONSTRUCTION ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Title bar
        title_frame = tk.Frame(self.root, bg=BG_DARK, pady=6)
        title_frame.pack(fill="x", padx=10)

        tk.Label(title_frame,
                 text="🔬  CANCER GROWTH EVOLUTION  —  Conway's Game of Life",
                 font=("Segoe UI", 16, "bold"), fg=TEXT_MAIN, bg=BG_DARK
                 ).pack(side="left", padx=10)
        self.gen_label = tk.Label(title_frame,
                 text="Generation: 0",
                 font=("Segoe UI", 12), fg=TEXT_DIM, bg=BG_DARK)
        self.gen_label.pack(side="right", padx=14)

        # ── Main layout: left (plots) | right (controls)
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        left  = tk.Frame(main, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(main, bg=BG_PANEL, width=310, padx=10, pady=8)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_plots(left)
        self._build_controls(right)

    def _build_plots(self, parent):
        self.fig = Figure(figsize=(10, 7.5), facecolor=BG_DARK)
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.38, wspace=0.28,
                               left=0.05, right=0.97, top=0.93, bottom=0.08)

        # Grid view (large, spans both rows on left)
        self.ax_grid = self.fig.add_subplot(gs[:, 0])
        self.ax_grid.set_facecolor(COLORS[EMPTY])
        self.ax_grid.set_title("Tumor Evolution Grid", color=TEXT_MAIN,
                               fontsize=11, fontweight="bold", pad=8)
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])

        # Population chart (top right)
        self.ax_pop = self.fig.add_subplot(gs[0, 1])
        self.ax_pop.set_facecolor(BG_CARD)
        self.ax_pop.set_title("Cell Population Over Time", color=TEXT_MAIN,
                              fontsize=10, fontweight="bold")
        self.ax_pop.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in self.ax_pop.spines.values():
            spine.set_edgecolor(BG_CARD)
        self.ax_pop.set_xlabel("Generation", color=TEXT_DIM, fontsize=8)
        self.ax_pop.set_ylabel("Cell Count", color=TEXT_DIM, fontsize=8)

        # Fraction stacked area chart (bottom right)
        self.ax_area = self.fig.add_subplot(gs[1, 1])
        self.ax_area.set_facecolor(BG_CARD)
        self.ax_area.set_title("Population Fraction (Stacked)", color=TEXT_MAIN,
                               fontsize=10, fontweight="bold")
        self.ax_area.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in self.ax_area.spines.values():
            spine.set_edgecolor(BG_CARD)
        self.ax_area.set_xlabel("Generation", color=TEXT_DIM, fontsize=8)
        self.ax_area.set_ylabel("Fraction", color=TEXT_DIM, fontsize=8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        # Init image
        self.img_obj = self.ax_grid.imshow(
            self.sim.grid, cmap=CMAP, vmin=0, vmax=4,
            interpolation="nearest", aspect="equal"
        )

    def _build_controls(self, parent):
        def section(text):
            f = tk.Frame(parent, bg=BG_PANEL)
            f.pack(fill="x", pady=(10, 2))
            tk.Label(f, text=text, font=("Segoe UI", 9, "bold"),
                     fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w")
            tk.Frame(parent, bg="#2a2a5a", height=1).pack(fill="x")
            return f

        def make_btn(parent, text, color, cmd):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=color, fg=TEXT_MAIN, activebackground=color,
                          activeforeground="white", relief="flat",
                          font=("Segoe UI", 9, "bold"), cursor="hand2",
                          padx=8, pady=5, bd=0)
            return b

        # ── Playback controls
        section("▶  PLAYBACK")
        btn_row1 = tk.Frame(parent, bg=BG_PANEL)
        btn_row1.pack(fill="x", pady=4)
        self.btn_play  = make_btn(btn_row1, "▶  Play",  BTN_PLAY,  self.play)
        self.btn_pause = make_btn(btn_row1, "⏸  Pause", BTN_PAUSE, self.pause)
        self.btn_step  = make_btn(btn_row1, "⏭  Step",  BTN_STEP,  self.step_once)
        self.btn_play .pack(side="left", padx=2, expand=True, fill="x")
        self.btn_pause.pack(side="left", padx=2, expand=True, fill="x")
        self.btn_step .pack(side="left", padx=2, expand=True, fill="x")

        btn_row2 = tk.Frame(parent, bg=BG_PANEL)
        btn_row2.pack(fill="x", pady=(0, 4))
        make_btn(btn_row2, "⟳  Reset", BTN_RESET, self.reset_sim).pack(
            fill="x", padx=2)

        # ── Speed slider
        section("⚡  SPEED")
        self.speed_var = tk.IntVar(value=80)
        speed_slider = ttk.Scale(parent, from_=10, to=500,
                                 variable=self.speed_var, orient="horizontal",
                                 command=self._on_speed)
        speed_slider.pack(fill="x", pady=4)
        self.speed_lbl = tk.Label(parent, text="80 ms / step",
                                  font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_PANEL)
        self.speed_lbl.pack(anchor="e")

        # ── Cancer parameters
        section("🧬  CANCER PARAMETERS")

        tk.Label(parent, text="Mutation Rate", font=("Segoe UI", 8),
                 fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", pady=(6,0))
        self.mut_var = tk.DoubleVar(value=0.02)
        ttk.Scale(parent, from_=0.0, to=0.15, variable=self.mut_var,
                  orient="horizontal",
                  command=lambda v: setattr(self.sim, "mutation_rate", float(v))
                  ).pack(fill="x")
        self.mut_lbl = tk.Label(parent, text="2%",
                                font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_PANEL)
        self.mut_lbl.pack(anchor="e")
        self.mut_var.trace_add("write",
            lambda *a: self.mut_lbl.config(
                text=f"{self.mut_var.get()*100:.1f}%"))

        tk.Label(parent, text="Cancer Birth Threshold",
                 font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", pady=(6,0))
        self.birth_var = tk.IntVar(value=3)
        ttk.Scale(parent, from_=1, to=6, variable=self.birth_var,
                  orient="horizontal",
                  command=lambda v: setattr(self.sim, "cancer_birth_thresh", int(float(v)))
                  ).pack(fill="x")
        self.birth_lbl = tk.Label(parent, text="3 neighbours",
                                  font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_PANEL)
        self.birth_lbl.pack(anchor="e")
        self.birth_var.trace_add("write",
            lambda *a: self.birth_lbl.config(
                text=f"{self.birth_var.get()} neighbours"))

        tk.Label(parent, text="Overpopulation Threshold",
                 font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", pady=(6,0))
        self.overpop_var = tk.IntVar(value=6)
        ttk.Scale(parent, from_=3, to=8, variable=self.overpop_var,
                  orient="horizontal",
                  command=lambda v: setattr(self.sim, "cancer_overpop_thresh", int(float(v)))
                  ).pack(fill="x")
        self.overpop_lbl = tk.Label(parent, text="6 neighbours",
                                    font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_PANEL)
        self.overpop_lbl.pack(anchor="e")
        self.overpop_var.trace_add("write",
            lambda *a: self.overpop_lbl.config(
                text=f"{self.overpop_var.get()} neighbours"))

        # ── Therapy
        section("💊  THERAPY")
        make_btn(parent, "💊  Apply Chemotherapy", BTN_CHEMO,
                 self.apply_chemo).pack(fill="x", pady=3)
        make_btn(parent, "☢  Apply Radiotherapy",  BTN_RADIO,
                 self.apply_radio).pack(fill="x", pady=3)
        self.therapy_lbl = tk.Label(parent, text="",
                                    font=("Segoe UI", 8), fg=ACCENT3, bg=BG_PANEL,
                                    wraplength=260, justify="left")
        self.therapy_lbl.pack(fill="x", pady=(2, 0))

        # ── Presets
        section("🎬  SIMULATION PRESETS")
        make_btn(parent, "1. Tumor Inception",     "#1a237e", self.load_inception
                 ).pack(fill="x", pady=2)
        make_btn(parent, "2. Therapy & Relapse",   "#4a148c", self.load_relapse
                 ).pack(fill="x", pady=2)
        make_btn(parent, "3. Competitive Dynamics","#006064", self.load_competitive
                 ).pack(fill="x", pady=2)

        # ── Stats card
        section("📊  LIVE STATISTICS")
        stats_frame = tk.Frame(parent, bg=BG_CARD, pady=8, padx=10)
        stats_frame.pack(fill="x", pady=4)

        self.stat_labels = {}
        stat_rows = [
            ("Healthy",    COLORS[HEALTHY],   "healthy"),
            ("Cancer",     COLORS[CANCER],    "cancer"),
            ("Resistant",  COLORS[RESISTANT], "resistant"),
            ("Necrotic",   COLORS[NECROTIC],  "necrotic"),
            ("Tumor Ø",    "#ffd740",         "diameter"),
            ("Growth Rate",ACCENT2,           "growth"),
        ]
        for label, color, key in stat_rows:
            row = tk.Frame(stats_frame, bg=BG_CARD)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"● {label}:", width=12, anchor="w",
                     font=("Segoe UI", 8), fg=color, bg=BG_CARD).pack(side="left")
            lbl = tk.Label(row, text="—", anchor="e",
                           font=("Segoe UI", 8, "bold"), fg=TEXT_MAIN, bg=BG_CARD)
            lbl.pack(side="right")
            self.stat_labels[key] = lbl

        # ── Legend
        section("🗺  LEGEND")
        legend_items = [
            ("Empty / ECM",    COLORS[EMPTY],     "■"),
            ("Healthy Cells",  COLORS[HEALTHY],   "■"),
            ("Cancer Cells",   COLORS[CANCER],    "■"),
            ("Resistant",      COLORS[RESISTANT], "■"),
            ("Necrotic Core",  COLORS[NECROTIC],  "■"),
        ]
        for name, color, sym in legend_items:
            row = tk.Frame(parent, bg=BG_PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=sym, fg=color, bg=BG_PANEL,
                     font=("Segoe UI", 11)).pack(side="left")
            tk.Label(row, text=f" {name}", fg=TEXT_DIM, bg=BG_PANEL,
                     font=("Segoe UI", 8)).pack(side="left")

    # ── CONTROLS ─────────────────────────────────────────────────────────────
    def play(self):
        if not self.running:
            self.running = True
            self._loop()

    def pause(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def step_once(self):
        self.pause()
        self.sim.step()
        self._update_display()

    def reset_sim(self):
        self.pause()
        preset = self._current_preset
        preset()
        self._update_display()

    def _on_speed(self, val):
        self.speed_ms = int(float(val))
        self.speed_lbl.config(text=f"{self.speed_ms} ms / step")

    def apply_chemo(self):
        self.sim.apply_chemotherapy()
        self.therapy_lbl.config(
            text="✔ Chemotherapy applied. Resistant cells survive!")
        self._update_display()

    def apply_radio(self):
        self.sim.apply_radiotherapy()
        self.therapy_lbl.config(
            text="☢ Radiotherapy applied. Higher efficacy, more damage!")
        self._update_display()

    def _loop(self):
        if self.running:
            self.sim.step()
            self._update_display()
            self.after_id = self.root.after(self.speed_ms, self._loop)

    # ── PRESETS ──────────────────────────────────────────────────────────────
    def load_inception(self):
        self.pause()
        self.sim.preset_tumor_inception()
        self._current_preset = self.load_inception
        self.therapy_lbl.config(text="")
        self._update_display()

    def load_relapse(self):
        self.pause()
        self.sim.preset_therapy_relapse()
        self._current_preset = self.load_relapse
        self.therapy_lbl.config(text="")
        self._update_display()

    def load_competitive(self):
        self.pause()
        self.sim.preset_competitive()
        self._current_preset = self.load_competitive
        self.therapy_lbl.config(text="")
        self._update_display()

    # ── DISPLAY UPDATE ───────────────────────────────────────────────────────
    def _update_display(self):
        g = self.sim.grid
        h = self.sim.history

        # ── Grid
        self.img_obj.set_data(g)
        self.ax_grid.set_title(
            f"Tumor Evolution Grid  [Gen {self.sim.generation}]",
            color=TEXT_MAIN, fontsize=11, fontweight="bold", pad=8)

        # ── Population line chart
        if len(h[HEALTHY]) > 0:
            gens = list(range(len(h[HEALTHY])))
            self.ax_pop.cla()
            self.ax_pop.set_facecolor(BG_CARD)
            self.ax_pop.set_title("Cell Population Over Time", color=TEXT_MAIN,
                                  fontsize=10, fontweight="bold")
            for spine in self.ax_pop.spines.values():
                spine.set_edgecolor("#2a2a5a")
            self.ax_pop.tick_params(colors=TEXT_DIM, labelsize=7)
            self.ax_pop.set_xlabel("Generation", color=TEXT_DIM, fontsize=8)
            self.ax_pop.set_ylabel("Cell Count", color=TEXT_DIM, fontsize=8)
            self.ax_pop.plot(gens, h[HEALTHY],   color=COLORS[HEALTHY],   lw=1.5, label="Healthy")
            self.ax_pop.plot(gens, h[CANCER],    color=COLORS[CANCER],    lw=1.5, label="Cancer")
            self.ax_pop.plot(gens, h[RESISTANT], color=COLORS[RESISTANT], lw=1.5, label="Resistant")
            self.ax_pop.plot(gens, h[NECROTIC],  color=COLORS[NECROTIC],  lw=1.2, label="Necrotic",
                             linestyle="--")
            self.ax_pop.legend(loc="upper left", fontsize=7,
                               facecolor=BG_CARD, edgecolor="#2a2a5a",
                               labelcolor=TEXT_MAIN)
            self.ax_pop.grid(True, color="#1e1e40", linewidth=0.5)

        # ── Stacked area chart
        if len(h[HEALTHY]) > 1:
            gens = list(range(len(h[HEALTHY])))
            total = self.sim.N ** 2
            frac_h = [v / total for v in h[HEALTHY]]
            frac_c = [v / total for v in h[CANCER]]
            frac_r = [v / total for v in h[RESISTANT]]
            frac_n = [v / total for v in h[NECROTIC]]
            frac_e = [v / total for v in h[EMPTY]]

            self.ax_area.cla()
            self.ax_area.set_facecolor(BG_CARD)
            self.ax_area.set_title("Population Fraction (Stacked)", color=TEXT_MAIN,
                                   fontsize=10, fontweight="bold")
            for spine in self.ax_area.spines.values():
                spine.set_edgecolor("#2a2a5a")
            self.ax_area.tick_params(colors=TEXT_DIM, labelsize=7)
            self.ax_area.set_xlabel("Generation", color=TEXT_DIM, fontsize=8)
            self.ax_area.set_ylabel("Fraction", color=TEXT_DIM, fontsize=8)
            self.ax_area.set_ylim(0, 1)
            self.ax_area.stackplot(
                gens,
                frac_n, frac_r, frac_c, frac_h, frac_e,
                colors=[COLORS[NECROTIC], COLORS[RESISTANT], COLORS[CANCER],
                        COLORS[HEALTHY], COLORS[EMPTY]],
                alpha=0.85,
                labels=["Necrotic", "Resistant", "Cancer", "Healthy", "Empty"]
            )
            handles = [
                mpatches.Patch(color=COLORS[NECROTIC],  label="Necrotic"),
                mpatches.Patch(color=COLORS[RESISTANT], label="Resistant"),
                mpatches.Patch(color=COLORS[CANCER],    label="Cancer"),
                mpatches.Patch(color=COLORS[HEALTHY],   label="Healthy"),
                mpatches.Patch(color=COLORS[EMPTY],     label="Empty"),
            ]
            self.ax_area.legend(handles=handles, loc="upper right", fontsize=7,
                                facecolor=BG_CARD, edgecolor="#2a2a5a",
                                labelcolor=TEXT_MAIN)
            self.ax_area.grid(True, color="#1e1e40", linewidth=0.5, alpha=0.5)

        self.canvas.draw_idle()

        # ── Gen label
        self.gen_label.config(text=f"Generation: {self.sim.generation}")

        # ── Stats
        n_h = int(np.sum(g == HEALTHY))
        n_c = int(np.sum(g == CANCER))
        n_r = int(np.sum(g == RESISTANT))
        n_n = int(np.sum(g == NECROTIC))
        diam = self.sim.estimate_tumor_diameter()

        # Growth rate (change in cancer+resistant over last 5 steps)
        if len(h[CANCER]) >= 5:
            prev_total = h[CANCER][-5] + h[RESISTANT][-5]
            cur_total  = n_c + n_r
            rate = cur_total - prev_total
            rate_text = f"+{rate}" if rate >= 0 else str(rate)
        else:
            rate_text = "—"

        self.stat_labels["healthy"]  .config(text=f"{n_h:,}")
        self.stat_labels["cancer"]   .config(text=f"{n_c:,}")
        self.stat_labels["resistant"].config(text=f"{n_r:,}")
        self.stat_labels["necrotic"] .config(text=f"{n_n:,}")
        self.stat_labels["diameter"] .config(text=f"{diam} cells")
        self.stat_labels["growth"]   .config(text=rate_text)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    # Dark title bar on Windows
    try:
        root.wm_attributes("-alpha", 1.0)
        from ctypes import windll
        windll.dwmapi.DwmSetWindowAttribute(
            windll.user32.GetForegroundWindow(), 20,
            byref := __import__("ctypes").c_int(1),
            __import__("ctypes").sizeof(__import__("ctypes").c_int)
        )
    except Exception:
        pass

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TScale",
                    background=BG_PANEL,
                    troughcolor="#2a2a5a",
                    sliderthickness=14)
    style.configure("Horizontal.TScale",
                    background=BG_PANEL)

    app = CancerSimApp(root)
    app._current_preset = app.load_inception
    root.mainloop()
