"""
Regenerate all remaining EDA plots with correct bearing temperature names
and fixed legend positions.

Plots generated:
  - fig-eda-distributions.pdf          (10-var reduced, legend fix)
  - fig-eda-boxplots-by-pump.pdf       (10-var reduced, legend lower-left for speed & bearing temp)
  - fig-eda-correlation-heatmaps.pdf   (all 16 vars, correct T. Rod. A/B/C labels)
  - fig-eda-correlation-diff.pdf       (all 16 vars, correct labels)
  - fig-eda-pca.pdf                    (all 16 vars, correct labels in loading chart)
  - fig-eda-speed-regimes.pdf          (legend lower-left)

Seasonal heatmap unchanged (no bearing column references).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 19
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 14

# ── paths ─────────────────────────────────────────────────────────────────────
BASE       = Path("<PATH_TO_EDA_DIR>")
DATA       = BASE / "../new_data"
PLOTS_DIR  = BASE / "plots"
THESIS_DIR = Path("<PATH_TO_THESIS_FIGURES>")
THESIS_DIR.mkdir(parents=True, exist_ok=True)

# ── rename mapping ────────────────────────────────────────────────────────────
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

# ── full 16-var list and short labels ─────────────────────────────────────────
SENSOR_COLS_FULL = [
    "Temperatura ambiente",
    "Velocidad del motor",
    "Temperatura fluido entrada",
    "Consumo de corriente",
    "Caudal de descarga",
    "Presión de salida",
    "Temperatura de rodamiento A",
    "Temperatura de rodamiento B",
    "Temperatura de rodamiento C",
    "Rodamiento motor A",
    "Rodamiento motor B",
    "Temperatura motor A",
    "Temperatura motor B",
    "Temperatura motor C",
    "Vibración rodamiento A",
    "Vibración rodamiento B",
]

SHORT_FULL = {
    "Temperatura ambiente":        "T. Ambiente",
    "Velocidad del motor":         "Velocidad",
    "Temperatura fluido entrada":  "T. Entrada",
    "Consumo de corriente":        "Corriente",
    "Caudal de descarga":          "Caudal",
    "Presión de salida":           "Presión",
    "Temperatura de rodamiento A": "T. Rod. A",
    "Temperatura de rodamiento B": "T. Rod. B",
    "Temperatura de rodamiento C": "T. Rod. C",
    "Rodamiento motor A":          "Rod. Motor A",
    "Rodamiento motor B":          "Rod. Motor B",
    "Temperatura motor A":         "T. Motor A",
    "Temperatura motor B":         "T. Motor B",
    "Temperatura motor C":         "T. Motor C",
    "Vibración rodamiento A":      "Vibr. A",
    "Vibración rodamiento B":      "Vibr. B",
}

# ── reduced 10-var list (one per coupled group) ───────────────────────────────
SENSOR_COLS_REDUCED = [
    "Temperatura ambiente",
    "Velocidad del motor",
    "Temperatura fluido entrada",
    "Consumo de corriente",
    "Caudal de descarga",
    "Presión de salida",
    "Temperatura de rodamiento A",
    "Rodamiento motor A",
    "Temperatura motor A",
    "Vibración rodamiento A",
]

SHORT_REDUCED = {
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

# ── data loading ──────────────────────────────────────────────────────────────
def load_all_data(folder_path):
    dfs = []
    for file in Path(folder_path).glob("*.csv"):
        df = pd.read_csv(file)
        parts = file.stem.split('_')
        df['pump_id'] = int(parts[1])
        df['date'] = parts[2]
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    return combined


print("Loading data...")
train_df = load_all_data(DATA / "train"); train_df['dataset'] = 'train'
test_df  = load_all_data(DATA / "test");  test_df['dataset']  = 'test'
combined_df = pd.concat([train_df, test_df], ignore_index=True)
for _df in [train_df, test_df, combined_df]:
    _df.rename(columns=RENAME_COLS, inplace=True)
print(f"  Train {len(train_df):,}  Test {len(test_df):,}")

train_clean_full    = train_df.dropna(subset=SENSOR_COLS_FULL)
test_clean_full     = test_df.dropna(subset=SENSOR_COLS_FULL)
combined_clean_full = combined_df.dropna(subset=SENSOR_COLS_FULL)

train_clean_red    = train_df.dropna(subset=SENSOR_COLS_REDUCED)
test_clean_red     = test_df.dropna(subset=SENSOR_COLS_REDUCED)
combined_clean_red = combined_df.dropna(subset=SENSOR_COLS_REDUCED)


def save(fig, name):
    p = PLOTS_DIR / name
    t = THESIS_DIR / name
    fig.savefig(p, bbox_inches='tight', dpi=200)
    fig.savefig(t, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  ✓ {name}")


# ── 1. Distributions (10 vars) ────────────────────────────────────────────────
print("\n[1/5] fig-eda-distributions.pdf")
n_vars = len(SENSOR_COLS_REDUCED)
fig, axes = plt.subplots(2, 5, figsize=(22, 11))
axes = axes.flatten()

for idx, col in enumerate(SENSOR_COLS_REDUCED):
    ax = axes[idx]
    td = train_clean_red[col].dropna()
    sd = test_clean_red[col].dropna()
    all_d = pd.concat([td, sd])
    q_lo, q_hi = all_d.quantile(0.005), all_d.quantile(0.995)
    if not (np.isfinite(q_lo) and np.isfinite(q_hi) and q_lo < q_hi):
        q_lo, q_hi = all_d.min(), all_d.max()
    bins = np.linspace(q_lo, q_hi, 50)
    ax.hist(td[(td >= q_lo) & (td <= q_hi)], bins=bins, alpha=0.55,
            label='Entrenamiento', color='steelblue', density=True)
    ax.hist(sd[(sd >= q_lo) & (sd <= q_hi)], bins=bins, alpha=0.55,
            label='Test', color='coral', density=True)
    ax.set_title(SHORT_REDUCED.get(col, col), fontsize=18, fontweight='bold')
    ax.set_xlabel('Valor', fontsize=16); ax.set_ylabel('Densidad', fontsize=16)
    ax.legend(fontsize=14); ax.grid(True, alpha=0.3); ax.set_xlim(q_lo, q_hi)

fig.suptitle(
    'Densidades de probabilidad: operación normal (entrenamiento) vs anómala (test)',
    fontsize=22, fontweight='bold', y=1.02
)
plt.tight_layout()
save(fig, 'fig-eda-distributions.pdf')


# ── 2. Boxplots by pump (10 vars, legend lower-left for speed & bearing temp) ─
print("[2/5] fig-eda-boxplots-by-pump.pdf")
fig, axes = plt.subplots(2, 5, figsize=(22, 13))
axes = axes.flatten()

LEGEND_LOWER_LEFT = {"Velocidad del motor", "Temperatura de rodamiento A"}
LABEL_MAP = {'train': 'Entrenamiento', 'test': 'Test'}

for idx, col in enumerate(SENSOR_COLS_REDUCED):
    ax = axes[idx]
    plot_data = combined_clean_red[['pump_id', 'dataset', col]].copy()
    plot_data['Bomba'] = plot_data['pump_id'].map(PUMP_LABELS)

    series = plot_data[col].dropna()
    y_lo, y_hi = series.quantile(0.01), series.quantile(0.99)
    use_robust = np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo < y_hi

    sns.boxplot(
        data=plot_data, x='Bomba', y=col, hue='dataset', ax=ax,
        order=['Bomba 1', 'Bomba 2', 'Bomba 3', 'Bomba 4'],
        palette={'train': 'steelblue', 'test': 'coral'},
        showfliers=False, hue_order=['train', 'test'],
    )
    if use_robust:
        ax.set_ylim(y_lo, y_hi)

    handles, labels = ax.get_legend_handles_labels()
    leg_loc = 'lower left' if col in LEGEND_LOWER_LEFT else 'best'
    ax.legend(handles, [LABEL_MAP.get(l, l) for l in labels],
              title='', fontsize=14, loc=leg_loc)

    ax.set_title(SHORT_REDUCED.get(col, col), fontsize=18, fontweight='bold')
    ax.set_xlabel('', fontsize=16); ax.set_ylabel('', fontsize=16)
    ax.tick_params(labelsize=15)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle(
    'Diagramas de caja por bomba: operación normal vs anómala (percentiles 1%–99%)',
    fontsize=22, fontweight='bold', y=1.01
)
plt.tight_layout()
save(fig, 'fig-eda-boxplots-by-pump.pdf')


# ── 3. Correlation heatmaps (all 16 vars, corrected short labels) ─────────────
print("[3/5] fig-eda-correlation-heatmaps.pdf")
train_corr = train_clean_full[SENSOR_COLS_FULL].rename(columns=SHORT_FULL).corr()
test_corr  = test_clean_full[SENSOR_COLS_FULL].rename(columns=SHORT_FULL).corr()

fig, axes = plt.subplots(1, 2, figsize=(28, 14))
mask = np.triu(np.ones_like(train_corr, dtype=bool), k=1)

sns.heatmap(train_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=axes[0], annot_kws={'size': 11},
            square=True, linewidths=0.5)
axes[0].set_title('Operación normal (entrenamiento) — Matriz de correlación',
                  fontsize=18, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=14)
axes[0].tick_params(axis='y', rotation=0, labelsize=14)

sns.heatmap(test_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=axes[1], annot_kws={'size': 11},
            square=True, linewidths=0.5)
axes[1].set_title('Operación anómala (test) — Matriz de correlación',
                  fontsize=18, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45, labelsize=14)
axes[1].tick_params(axis='y', rotation=0, labelsize=14)

plt.tight_layout()
save(fig, 'fig-eda-correlation-heatmaps.pdf')


# ── 4. Correlation difference ─────────────────────────────────────────────────
print("[4/5] fig-eda-correlation-diff.pdf")
corr_diff = test_corr - train_corr

fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
sns.heatmap(corr_diff, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-0.8, vmax=0.8, ax=ax, annot_kws={'size': 8},
            square=True, linewidths=0.5)
ax.set_title(
    'Diferencia de correlación (test − entrenamiento)\n'
    'Positivo = correlación más fuerte en test; negativo = más fuerte en entrenamiento',
    fontsize=11, fontweight='bold'
)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0, labelsize=8)
plt.tight_layout()
save(fig, 'fig-eda-correlation-diff.pdf')


# ── 5. PCA (all 16 vars, correct short labels in loading chart) ───────────────
print("[5/5] fig-eda-pca.pdf")
X = combined_clean_full[SENSOR_COLS_FULL].dropna()
y = combined_df.loc[X.index, 'dataset']
pump_ids = combined_df.loc[X.index, 'pump_id']

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Subsample to avoid 70k+ vector paths that bloat PDF and cause loading issues.
# rasterized=True renders the scatter as a bitmap layer so axes/labels stay crisp.
rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 3, figsize=(22, 8))

# PC1 vs PC2 — coloured by dataset
ax = axes[0]
SCATTER_COLORS = {'train': '#1565C0', 'test': '#C62828'}  # deep blue vs deep red
SCATTER_LABELS = {'train': 'Entrenamiento (normal)', 'test': 'Test (anómalo)'}
for ds in ['train', 'test']:
    idx = np.where((y == ds).values)[0]
    if len(idx) > 8000:
        idx = rng.choice(idx, size=8000, replace=False)
        idx.sort()
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
               c=SCATTER_COLORS[ds], label=SCATTER_LABELS[ds],
               alpha=0.45, s=12, linewidths=0, rasterized=True)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=16)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=16)
ax.set_title('PC1 vs PC2: normal vs anómalo', fontweight='bold', fontsize=19)
ax.legend(fontsize=14, markerscale=2); ax.grid(True, alpha=0.3)

# PC1 vs PC2 — coloured by pump
ax = axes[1]
pump_palette = {1: '#1565C0', 2: '#6A1B9A', 3: '#E65100', 4: '#2E7D32'}
for pid in sorted(pump_ids.unique()):
    idx = np.where((pump_ids == pid).values)[0]
    if len(idx) > 5000:
        idx = rng.choice(idx, size=5000, replace=False)
        idx.sort()
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=pump_palette[pid],
               label=PUMP_LABELS[pid], alpha=0.45, s=12,
               linewidths=0, rasterized=True)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=16)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=16)
ax.set_title('PC1 vs PC2: por bomba', fontweight='bold', fontsize=19)
ax.legend(fontsize=14, markerscale=2); ax.grid(True, alpha=0.3)

# Feature loadings
ax = axes[2]
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=[SHORT_FULL.get(c, c) for c in SENSOR_COLS_FULL]
)
loadings[['PC1', 'PC2']].plot(kind='barh', ax=ax, color=['#1565C0', '#C62828'])
ax.set_title('Cargas de las variables (PC1 y PC2)', fontweight='bold', fontsize=19)
ax.set_xlabel('Carga', fontsize=16)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.legend(fontsize=14); ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
save(fig, 'fig-eda-pca.pdf')


# ── 6. Speed regimes — legend lower-left ─────────────────────────────────────
# (already correct names — only legend position changes)
print("[+] fig-eda-speed-regimes.pdf  (legend fix)")
speed_col = "Velocidad del motor"
speed_data = train_clean_full[[speed_col, "pump_id"]].dropna().copy()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(speed_data[[speed_col]])
cluster_centers = sorted(kmeans.cluster_centers_.flatten())

pump_colors_hist = {1: "#2196F3", 2: "#9C27B0", 3: "#FF9800", 4: "#4CAF50"}
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for idx, pid in enumerate(sorted(train_clean_full["pump_id"].unique())):
    ax = axes[idx // 2][idx % 2]
    pump_spd = speed_data[speed_data["pump_id"] == pid][speed_col]
    ax.hist(pump_spd, bins=60, color=pump_colors_hist[pid], alpha=0.75,
            edgecolor="white", linewidth=0.3)
    for center in cluster_centers:
        ax.axvline(center, color="#E53935", linestyle="--", linewidth=2.0,
                   label=f"{center:.0f} rpm")
    ax.set_title(PUMP_LABELS[pid], fontsize=19, fontweight="bold")
    ax.set_xlabel("Velocidad del motor (rpm)", fontsize=16)
    ax.set_ylabel("Registros", fontsize=16)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=14, framealpha=0.85, loc='lower left')

fig.suptitle(
    "Histograma de velocidad del motor por bomba — operación normal (entrenamiento)",
    fontsize=22, fontweight="bold"
)
plt.tight_layout()
save(fig, 'fig-eda-speed-regimes.pdf')

# ── 7. Seasonal heatmap ───────────────────────────────────────────────────────
print("[+] fig-eda-seasonal-heatmap.pdf")
import calendar

train_clean = train_df.copy()
train_clean["month"] = pd.to_datetime(train_clean.get("date", train_clean.index), errors='coerce').dt.month

# build month from filename-derived 'date' column if present
if 'date' not in train_clean.columns:
    raise RuntimeError("'date' column not found in train_df — check load_all_data")

train_clean["month"] = pd.to_datetime(train_clean["date"]).dt.month

monthly_ops = (
    train_clean.groupby(["pump_id", "month"])
    .size()
    .unstack(fill_value=0)
)
monthly_ops.index = [PUMP_LABELS[i] for i in monthly_ops.index]
month_names_es = {
    1:"Ene", 2:"Feb", 3:"Mar", 4:"Abr", 5:"May", 6:"Jun",
    7:"Jul", 8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dic",
}
monthly_ops.columns = [month_names_es.get(m, str(m)) for m in monthly_ops.columns]

fig, ax = plt.subplots(figsize=(18, 5))
sns.heatmap(
    monthly_ops, annot=True, fmt="d", cmap="Blues", ax=ax,
    linewidths=0.5, linecolor="white", annot_kws={"size": 15},
    cbar_kws={"label": "Registros de operación normal (5 min)"},
)
ax.set_title(
    "Distribución mensual de registros de operación normal por bomba",
    fontsize=20, fontweight="bold"
)
ax.set_xlabel("Mes", fontsize=16)
ax.set_ylabel("", fontsize=16)
ax.tick_params(axis="x", rotation=0, labelsize=15)
ax.tick_params(axis="y", rotation=0, labelsize=15)
plt.tight_layout()
save(fig, 'fig-eda-seasonal-heatmap.pdf')

print("\n✅ All plots done")
