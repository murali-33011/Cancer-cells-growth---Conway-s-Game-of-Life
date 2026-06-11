# 🔬 Cancer Cell Growth Prediction : Conway's Game of Life

A **multi-state cellular automaton** that models tumor evolution in living tissue, built on an extended version of Conway's Game of Life. Watch a single cancer cell invade healthy tissue, form a necrotic core, develop drug-resistant mutations, and respond to chemotherapy or radiotherapy all in real time.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-informational)
![Matplotlib](https://img.shields.io/badge/Plots-Matplotlib-orange)
![NumPy](https://img.shields.io/badge/Compute-NumPy-013243?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### 🧬 Multi-State Cellular Automaton
Five distinct cell states replace the binary alive/dead of classic Conway's Game of Life:

| State | Color | Description |
|---|---|---|
| **Empty / ECM** | ⬛ Dark | Extracellular matrix open space |
| **Healthy Cell** | 🟢 Green | Normal tissue with contact inhibition |
| **Cancer Cell** | 🔴 Red | Invasive cells with altered survival rules |
| **Resistant Cancer** | 🟣 Magenta | Drug-resistant mutants that survive therapy |
| **Necrotic Core** | ⬜ Gray | Dead tissue formed at the tumor's overcrowded center |

### 🔁 Biologically-Inspired Rules
- **Healthy cells** survive via homeostasis (2–4 healthy neighbours) and die when overwhelmed by ≥3 cancer-cell neighbours.
- **Cancer cells** are born from empty cells with ≥3 cancer neighbours, survive under broader conditions, and collapse into a necrotic core when overly dense (≥7 cancer/necrotic neighbours).
- **Necrotic cells** decay stochastically (~8% chance per generation) back to empty space.
- **Resistant cells** arise via random mutation during cancer cell birth or invasion, at a tunable rate.

### 🎛️ Interactive Controls
- **Play / Pause / Step** : run the simulation continuously or advance one generation at a time.
- **Speed Slider** : adjust simulation speed from 10 ms to 500 ms per step.
- **Reset** : restart the current scenario from scratch.

### ⚙️ Live Parameter Tuning
Adjust key biological parameters mid-simulation using sliders:
- **Mutation Rate** (0–15%) : probability that a new cancer cell becomes drug-resistant.
- **Cancer Birth Threshold** (1–6 neighbours) : how aggressive cancer spread is.
- **Overpopulation Threshold** (3–8 neighbours) : tolerance before necrosis sets in.

### 💊 Therapy Simulation
Apply treatments at any point in the simulation:
- **Chemotherapy** : kills ~70% of cancer cells but spares resistant mutants; causes ~15% collateral healthy cell damage.
- **Radiotherapy** : higher efficacy (~80%) against cancer, but greater damage to healthy tissue (~25%).

### 🎬 Simulation Presets
Three ready-to-run biological scenarios:

| Preset | Description |
|---|---|
| **Tumor Inception** | A single cancer cell placed in a field of healthy tissue. Observe organic tumor growth from scratch. |
| **Therapy & Relapse** | An established tumor with a necrotic core and seeded resistant cells, ready for therapy. Apply chemo and watch resistant cells repopulate. |
| **Competitive Dynamics** | Healthy tissue (left) vs. a cancer blob (right) competing for space. |

### 📊 Real-Time Visualizations
Three synchronized views update every generation:

1. **Tumor Evolution Grid** : 80×80 cell canvas with color-coded cell states.
2. **Cell Population Chart** : live line chart tracking healthy, cancer, resistant, and necrotic cell counts over time.
3. **Population Fraction (Stacked Area)** : normalized view showing the shifting balance of the entire grid at a glance.

### 📈 Live Statistics Panel
- Per-type cell counts (Healthy, Cancer, Resistant, Necrotic)
- **Tumor diameter** estimate in cell units (computed from bounding radius of all tumor cells)
- **5-generation growth rate** : rolling delta of total cancerous cells

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib
```

Tkinter is included with most Python distributions. On Linux, install it via:

```bash
sudo apt-get install python3-tk
```

### Run

```bash
python cancer_simulation.py
```

---

## 🖥️ Interface Overview

```
┌──────────────────────────────────────────────────┬──────────────────┐
│                                                  │  ▶ PLAYBACK      │
│   Tumor Evolution Grid    │  Population Chart    │  ⚡ SPEED        │
│       (80×80 cells)       │  (line chart)        │  🧬 PARAMETERS   │
│                           ├──────────────────────│  💊 THERAPY      │
│                           │  Population Fraction │  🎬 PRESETS      │
│                           │  (stacked area)      │  📊 STATS        │
└──────────────────────────────────────────────────┴──────────────────┘
```

---

## 🧪 How It Works

The simulation engine (`CancerAutomaton`) applies transition rules simultaneously to all cells each generation using vectorized NumPy operations : no per-cell Python loops. Neighbour counts for each state are computed using 8-directional manual convolution on boolean masks, keeping each step fast even at 80×80 resolution.

Key computational methods:
- `count_neighbors(grid, state)` : counts same-state 8-connected neighbours for every cell at once.
- `step()` : applies all birth, survival, invasion, necrosis, and decay rules in a single vectorized pass.
- `estimate_tumor_diameter()` : computes the bounding radius from the centroid of all tumor cells.

---

## 📦 Project Structure

```
cancer_simulation.py
├── CancerAutomaton        # Simulation engine (rules, presets, therapy)
│   ├── step()             # Core CA transition
│   ├── apply_chemotherapy()
│   ├── apply_radiotherapy()
│   └── presets: inception / relapse / competitive
└── CancerSimApp           # Tkinter GUI + Matplotlib visualizations
    ├── _build_plots()     # Grid + population charts
    ├── _build_controls()  # Sliders, buttons, stats panel
    └── _update_display()  # Live re-render every generation
```

---

## 🔭 Potential Extensions

- Export simulation frames as GIF or video
- Add targeted therapy that also kills resistant cells (at higher cost)
- Angiogenesis: nutrient gradient affecting cancer survival probability
- Immune cell state that hunts cancer
- Save/load simulation states

---

## 📄 License

MIT License. See `LICENSE` for details.
