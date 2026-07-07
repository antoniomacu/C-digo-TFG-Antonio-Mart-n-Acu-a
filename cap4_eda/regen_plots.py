"""
Regenerate fig-eda-distributions.pdf and fig-eda-boxplots-by-pump.pdf
with reduced variable set: one representative per coupled group (A/B/C, A/B).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path("<PATH_TO_EDA_DIR>")
DATA = BASE / "../new_data"
PLOTS_DIR = BASE / "plots"
THESIS_FIGS = Path("<PATH_TO_THESIS_FIGURES>")
THESIS_FIGS.mkdir(parents=True, exist_ok=True)

# ── data loading ─────────────────────────────────────────────────────────────
RENAME_COLS = {
    "Ambient temperature":                      "Temperatura ambiente",
    "Main HTF Pump Speed":                      "Velocidad del motor",
    "Main HTF Pump Inlet Temperature":          "Temperatura fluido entrada",
    "Main HTF Pump Current Consumption":        "Consumo de corriente",
    "Main HTF Pump Flow":                       "Caudal de descarga",
    "Main HTF Pump Outlet Pressure":            "Presión de salida",
    "Main HTF Pump NDE Outboard bearing":       "Temperatura de rodamiento A",
    "Main HTF Pump NDE Inboard bearing":        "Temperatura de rodamiento B",
    "Main HTF Pump DE bearing":                 "Temperatura de rodamiento C",
    "Main HTF Pump Motor bearing Temp 1":       "Rodamiento motor A",
    "Main HTF Pump Motor bearing Temp 2":       "Rodamiento motor B",
    "Main HTF Pump Motor U winding Temp 1":     "Temperatura motor A",
    "Main HTF Pump Motor U winding Temp 2":     "Temperatura motor B",
    "Main HTF Pump Motor U winding Temp 3":     "Temperatura motor C",
    "Main HTF Pump DE Side Bearing vibration":  "Vibración rodamiento A",
    "Main HTF Pump NDE Side Bearing vibration": "Vibración rodamiento B",
}
PUMP_LABELS = {1: "Bomba 1", 2: "Bomba 2", 3: "Bomba 3", 4: "Bomba 4"}


def load_all_data(folder_path):
    all_files = list(Path(folder_path).glob("*.csv"))
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        parts = file.stem.split('_')
        df['pump_id'] = int(parts[1])
        df['date'] = parts[2]
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    return combined


print("Loading data...")
train_df = load_all_data(DATA / "train")
train_df['dataset'] = 'train'
test_df = load_all_data(DATA / "test")
test_df['dataset'] = 'test'
combined_df = pd.concat([train_df, test_df], ignore_index=True)

train_df    = train_df.rename(columns=RENAME_COLS)
test_df     = test_df.rename(columns=RENAME_COLS)
combined_df = combined_df.rename(columns=RENAME_COLS)
print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

# ── reduced sensor set ────────────────────────────────────────────────────────
# One representative per coupled group; drop B/C duplicates
SENSOR_COLS_REDUCED = [
    "Temperatura ambiente",
    "Velocidad del motor",
    "Temperatura fluido entrada",
    "Consumo de corriente",
    "Caudal de descarga",
    "Presión de salida",
    "Temperatura de rodamiento A",   # representative for A/B/C
    "Rodamiento motor A",            # representative for A/B
    "Temperatura motor A",           # representative for A/B/C
    "Vibración rodamiento A",        # representative for A/B
]

SHORT_NAMES = {
    "Temperatura ambiente":        "T. Ambiente",
    "Velocidad del motor":         "Velocidad",
    "Temperatura fluido entrada":  "T. Entrada",
    "Consumo de corriente":        "Corriente",
    "Caudal de descarga":          "Caudal",
    "Presión de salida":           "Presión",
    "Temperatura de rodamiento A": "T. Rodamiento A",
    "Rodamiento motor A":          "Rod. Motor A",
    "Temperatura motor A":         "T. Motor A",
    "Vibración rodamiento A":      "Vibración A",
}

# clean data (drop NaN rows for sensor cols only)
train_clean    = train_df.dropna(subset=SENSOR_COLS_REDUCED)
test_clean     = test_df.dropna(subset=SENSOR_COLS_REDUCED)
combined_clean = combined_df.dropna(subset=SENSOR_COLS_REDUCED)

# ── plot 1: distributions ─────────────────────────────────────────────────────
print("Generating fig-eda-distributions.pdf ...")
n_vars = len(SENSOR_COLS_REDUCED)
n_cols = 5
n_rows = int(np.ceil(n_vars / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for idx, col in enumerate(SENSOR_COLS_REDUCED):
    ax = axes[idx]
    train_data = train_clean[col].dropna()
    test_data  = test_clean[col].dropna()

    all_data = pd.concat([train_data, test_data])
    q_low  = all_data.quantile(0.005)
    q_high = all_data.quantile(0.995)
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low >= q_high:
        q_low, q_high = all_data.min(), all_data.max()

    train_plot = train_data[(train_data >= q_low) & (train_data <= q_high)]
    test_plot  = test_data[(test_data >= q_low) & (test_data <= q_high)]
    bins = np.linspace(q_low, q_high, 50)

    ax.hist(train_plot, bins=bins, alpha=0.55, label='Entrenamiento', color='steelblue', density=True)
    ax.hist(test_plot,  bins=bins, alpha=0.55, label='Test',          color='coral',     density=True)

    ax.set_title(SHORT_NAMES.get(col, col), fontsize=11, fontweight='bold')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidad')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_low, q_high)

for extra_ax in axes[n_vars:]:
    extra_ax.set_visible(False)

plt.suptitle(
    'Densidades de probabilidad: operación normal (entrenamiento) vs anómala (test)',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()

out = PLOTS_DIR / 'fig-eda-distributions.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
plt.savefig(THESIS_FIGS / 'fig-eda-distributions.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved → {out}")

# ── plot 2: boxplots by pump ──────────────────────────────────────────────────
print("Generating fig-eda-boxplots-by-pump.pdf ...")
n_cols = 5
n_rows = int(np.ceil(n_vars / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for idx, col in enumerate(SENSOR_COLS_REDUCED):
    ax = axes[idx]
    plot_data = combined_clean[['pump_id', 'dataset', col]].copy()
    plot_data['Bomba'] = plot_data['pump_id'].map(PUMP_LABELS)

    series = plot_data[col].dropna()
    y_low  = series.quantile(0.01)
    y_high = series.quantile(0.99)
    use_robust = np.isfinite(y_low) and np.isfinite(y_high) and (y_low < y_high)

    sns.boxplot(
        data=plot_data, x='Bomba', y=col, hue='dataset',
        ax=ax,
        palette={'train': 'steelblue', 'test': 'coral'},
        showfliers=False,
        hue_order=['train', 'test'],
    )

    if use_robust:
        ax.set_ylim(y_low, y_high)

    # Fix legend labels
    handles, labels = ax.get_legend_handles_labels()
    label_map = {'train': 'Entrenamiento', 'test': 'Test'}
    ax.legend(handles, [label_map.get(l, l) for l in labels], title='', fontsize=8)

    ax.set_title(SHORT_NAMES.get(col, col), fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True, alpha=0.3, axis='y')

for extra_ax in axes[n_vars:]:
    extra_ax.set_visible(False)

plt.suptitle(
    'Diagramas de caja por bomba: operación normal vs anómala (percentiles 1%–99%)',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()

out = PLOTS_DIR / 'fig-eda-boxplots-by-pump.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
plt.savefig(THESIS_FIGS / 'fig-eda-boxplots-by-pump.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved → {out}")

print("\n✅ Done")
