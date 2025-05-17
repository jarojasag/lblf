import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from ema_workbench import (Model, RealParameter, ScalarOutcome,
                           Policy, SequentialEvaluator, Scenario)
from ema_workbench.analysis import parcoords
from ema_workbench.analysis import prim
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from lblf_Policy_natural_resources import policySIRModel

np.random.seed(17)
random.seed(17)

# Model
ema_model = Model('PolicySIR', function=policySIRModel)

# Uncertainties
ema_model.uncertainties = [
    RealParameter('a_w', 0.01, 3.0), # how wage deviation affects radicalization
    RealParameter('a_e', 0.01, 150.0), # how elite fraction affects radicalization
    RealParameter('nat_res_regen', 0.01,  0.2), # resource regeneration
    RealParameter('delta_extract', 0.005, 0.1), # resource extraction
    RealParameter('delta', 0.1, 1.0),  # Fraction of radicals converting to moderate per time step
    RealParameter('alpha_w', -3.0, 3.0),  # Exponent controlling wage response to resource ratio
    RealParameter('conservation_effectiveness', 0.1, 1.0),  # From 10% to 100% effective
    RealParameter('conservation_unit_cost', 0.1, 30.0),
    RealParameter('eta_deplet', 0.1,  3.0),   # wage-to-depletion exponent
    RealParameter('mu_elite_extr', 0.1,  1.0),   # extraction multiplier from elites
    RealParameter('mu_0', 0.0005, 0.01),# upward mobility to elites
    RealParameter('e_0', 0.005, 0.05)  # expected elite share
]

# Levers
ema_model.levers = [
    RealParameter('eta_w', 0.0, 2.0), # wage sensitivity
    RealParameter('eta_a', 0.0, 2.0),  # radicalization threshold policy
    RealParameter('w_T', 0.5, 1.0), # wage target
    RealParameter('I_Ta', 0.01, 0.4), # radicalization threshold for each cohort
    RealParameter('conservation_effort', 0.0, 1.0)   
]

# Putcomes
ema_model.outcomes = [
    ScalarOutcome('max_radicalized'),
    ScalarOutcome('final_radicalized'),
    ScalarOutcome('wage_cost'),
    ScalarOutcome('conservation_cost'),
    ScalarOutcome('final_resource'),
]

# Baseline (No Levers)

baseline_pol = Policy("baseline", **{lv.name: 0 for lv in ema_model.levers})

with SequentialEvaluator(ema_model) as evaluator:
    experiments_base, outcomes_base = evaluator.perform_experiments(
        scenarios=2000,             
        policies=baseline_pol
    )

scenarios = [Scenario(f"scen_{i}", **row.drop(['policy']).to_dict())
    for i, row in experiments_base.iterrows()] # Saving Xs 

experiments_base_df = pd.DataFrame.from_dict(experiments_base)
outcomes_base_df = pd.DataFrame.from_dict(outcomes_base)
uncertainty_base_df = pd.concat([experiments_base_df, outcomes_base_df], axis=1)

base_df_numeric = uncertainty_base_df.select_dtypes(include=[float, int])
base_df_numeric = base_df_numeric.replace([np.inf, -np.inf], np.nan)
base_df_numeric = base_df_numeric.dropna() # 6.5% dropped

(base_df_numeric["final_resource"] < 0).mean() # 0.42% of Columns removed
base_df_numeric = base_df_numeric[base_df_numeric["final_resource"] >= 0]

base_df_numeric.to_csv('EMA_Output_BaseRun.csv')

### Exploration 

with SequentialEvaluator(ema_model) as evaluator:
    experiments_unc, outcomes_unc = evaluator.perform_experiments(scenarios=scenarios, policies=25)

results_unc_df = pd.DataFrame.from_dict(experiments_unc)
outcomes_unc_df = pd.DataFrame.from_dict(outcomes_unc)
uncertainty_df = pd.concat([results_unc_df, outcomes_unc_df], axis=1)

uncertainty_df_numeric = uncertainty_df.select_dtypes(include=[float, int])
uncertainty_df_numeric = uncertainty_df_numeric.replace([np.inf, -np.inf], np.nan)
uncertainty_df_numeric = uncertainty_df_numeric.dropna() # 19.65 % dropped
(uncertainty_df_numeric["final_resource"] < 0).mean() # 0.6% of Columns removed
uncertainty_df_numeric = uncertainty_df_numeric[uncertainty_df_numeric["final_resource"] >= 0]

uncertainty_df_numeric.to_csv('EMA_Output_LeverRun.csv')
# uncertainty_df_numeric = pd.read_csv("EMA_Output_LeverRun.csv")
# uncertainty_df_numeric = uncertainty_df_numeric.drop(columns="Unnamed: 0")

### Renaming
rename_map = {
    # ─────────────────────  UNCERTAINTIES  ─────────────────────
    "a_w":                       "Wage → Radicalization",
    "a_e":                       "Elite → Radicalization",
    "nat_res_regen":             "Resource Regeneration",
    "delta_extract":             "Extraction Rate",
    "delta":                     "De-radicalization Rate",
    "alpha_w":                   "Wage–Resource Exp.",
    "conservation_effectiveness":"Conservation Effectiveness",
    "conservation_unit_cost":    "Conservation Unit Cost",
    "eta_deplet":                "Wage→Depletion Exp.",
    "mu_elite_extr":             "Elite Extraction Mult.",
    "mu_0":                      "Elite Mobility Rate",
    "e_0":                       "Elite Proportion",

    # ─────────────────────────  LEVERS  ─────────────────────────
    "eta_w":                     "Wage Target Sensitivity",
    "eta_a":                     "Radical. Treshold Sensitivity",
    "w_T":                       "Target Wage",
    "I_Ta":                      "Radical Threshold",
    "conservation_effort":       "Conservation Effort",

    # ─────────────────────────  OUTCOMES  ───────────────────────
    "max_radicalized":           "Peak Radicalization",
    "final_radicalized":         "Final Radicalization",
    "wage_cost":                 "Wage Cost",
    "conservation_cost":         "Conservation Cost",
    "final_resource":            "Final Resource"
}

uncertainty_df_numeric = uncertainty_df_numeric.rename(columns=rename_map)
ordered_cols = list(rename_map.values()) 
uncertainty_df_numeric = uncertainty_df_numeric[ordered_cols]

base_df_numeric = base_df_numeric.rename(columns=rename_map)
base_df_numeric = base_df_numeric[ordered_cols]

### Plotting 

def plot_highlighted_experiments(
        df: pd.DataFrame,
        outcome_col: str,
        n: int = 20,                 # how many runs to highlight
        mode: str = "low",           # 'low' or 'high'
        color: str = "blue",
        decimals: int = 1,           # rounding
        fig_width: float = 14,
        fig_height: float | None = None,
        label_fontsize: int = 8,
        background_sample: int = 1000,
        line_width_bg: float = 0.4,
        line_width_fg: float = 1.2):
    """Parallel-coordinates plot that highlights best / worst experiments."""

    # ubset to highlight
    if mode == "low":
        subset = df.nsmallest(n, outcome_col)
        sel = "Lowest"
    elif mode == "high":
        subset = df.nlargest(n, outcome_col)
        sel = "Highest"
    else:
        raise ValueError("mode must be 'low' or 'high'")

    limits = parcoords.get_limits(df)

    # random grey background sample
    bg = df.sample(background_sample, random_state=0) if len(df) > background_sample else df

    # draw
    paraxes = parcoords.ParallelAxes(limits, fontsize=label_fontsize)
    if fig_height is None:
        fig_height = fig_width / 2.5
    paraxes.fig.set_size_inches(fig_width, fig_height)

    paraxes.plot(bg, color="lightgray", linewidth=line_width_bg,
                 alpha=0.4, rasterized=True)       
    paraxes.plot(subset, color=color, linewidth=line_width_fg)  

    for ax in paraxes.axes:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(labelsize=label_fontsize)

    # Background
    n_uncert = 12   # first 12 columns = uncertainties
    n_levers =  5   # next 5 columns  = levers

    colors = {
        'uncertainty': '#fffadc',   # pale yellow
        'lever':       '#dceeff',   # pale blue
        'outcome':     '#dbf5db'    # pale green
    }

    for i, ax in enumerate(paraxes.axes):
        if i < n_uncert:
            ax.patch.set_facecolor(colors['uncertainty'])
        elif i < n_uncert + n_levers:
            ax.patch.set_facecolor(colors['lever'])
        else:
            ax.patch.set_facecolor(colors['outcome'])
        ax.patch.set_zorder(0)          

    handles = [
        mpatches.Patch(color=colors['uncertainty'], label='Uncertainty'),
        mpatches.Patch(color=colors['lever'], label='Lever'),
        mpatches.Patch(color=colors['outcome'], label='Outcome'),
    ]
    plt.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.45), ncol=3,
               frameon=False, fontsize=label_fontsize)

    plt.suptitle(f"{n} Experiments with {sel} '{outcome_col}'",
                 x=0.0, ha='left', fontsize=20) 
    plt.show()

    return limits


## Parallel plots 

# Base

plot_highlighted_experiments(base_df_numeric, "Peak Radicalization", n=1500, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Peak Radicalization", n=20, mode="high", color="red")

plot_highlighted_experiments(base_df_numeric, "Final Radicalization", n=20, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Final Radicalization", n=20, mode="high", color="red")

plot_highlighted_experiments(base_df_numeric, "Wage Cost", n=20, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Wage Cost", n=20, mode="high", color="red")

plot_highlighted_experiments(base_df_numeric, "Conservation Cost", n=20, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Conservation Cost", n=20, mode="high", color="red")

plot_highlighted_experiments(base_df_numeric, "Final Resource", n=20, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Final Resource", n=20, mode="high", color="red")


# Levers

plot_highlighted_experiments(uncertainty_df_numeric, "Peak Radicalization", n=10000, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Peak Radicalization", n=2000, mode="high", color="red")

plot_highlighted_experiments(uncertainty_df_numeric, "Final Radicalization", n=20, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Final Radicalization", n=20, mode="high", color="red")

plot_highlighted_experiments(uncertainty_df_numeric, "Wage Cost", n=20, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Wage Cost", n=20, mode="high", color="red")

plot_highlighted_experiments(uncertainty_df_numeric, "Conservation Cost", n=20, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Conservation Cost", n=20, mode="high", color="red")

plot_highlighted_experiments(uncertainty_df_numeric, "Final Resource", n=20, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Final Resource", n=20, mode="high", color="red")

# Pareto Frontier


# 2.  Pareto filter (minimise both columns) ------------------------

uncertainty_df_numeric["Total Policy Cost"] = uncertainty_df_numeric["Wage Cost"] + uncertainty_df_numeric["Conservation Cost"] 

objs = uncertainty_df_numeric[["Peak Radicalization", "Total Policy Cost"]].values

# non-dominated mask: True if no other row is better on BOTH objectives
is_pareto = np.ones(len(objs), dtype=bool)
for i, (r1, c1) in enumerate(objs):
    if is_pareto[i]:
        # any point that dominates row i?
        dominated = (objs[:, 0] <= r1) & (objs[:, 1] <= c1) & \
                    ((objs[:, 0] < r1) | (objs[:, 1] < c1))
        is_pareto[dominated] = False           # mark those as dominated
        is_pareto[i] = True                    # keep current row

uncertainty_df_numeric["pareto"] = np.where(is_pareto, "front", "dominated")

# 3.  Plot ---------------------------------------------------------
sns.set_style("whitegrid")

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=uncertainty_df_numeric,
    x="Total Policy Cost",
    y="Peak Radicalization",
    hue="pareto",
    palette={"front": "red", "dominated": "grey"},
    alpha=0.7,
    s=40
)

plt.title("Pareto Front: Peak Radicalization vs Total Policy Cost")
plt.xlabel("Total Policy Cost")
plt.ylabel("Peak Radicalization")
plt.legend(title="Status", loc="best")
plt.tight_layout()
plt.show()

# red on top

dom    = uncertainty_df_numeric[uncertainty_df_numeric.pareto == "dominated"]
front  = uncertainty_df_numeric[uncertainty_df_numeric.pareto == "front"]

sns.set_style("whitegrid")
plt.figure(figsize=(7, 5))

# 1️⃣  draw dominated points first  (grey, lower z-order)
sns.scatterplot(
    data=dom,
    x="Total Policy Cost",
    y="Peak Radicalization",
    color="grey",
    alpha=0.6,
    s=35,
    label="dominated",
    zorder=1           # ↓ sits behind
)

# 2️⃣  overlay Pareto front  (red, higher z-order)
sns.scatterplot(
    data=front,
    x="Total Policy Cost",
    y="Peak Radicalization",
    color="red",
    alpha=0.9,
    s=45,
    label="front",
    zorder=3           # ↑ sits on top
)

plt.title("Pareto Front: Peak Radicalization vs Total Policy Cost")
plt.xlabel("Total Policy Cost")
plt.ylabel("Peak Radicalization")

# 3️⃣  move the legend outside, on the right
plt.legend(
    title="Status",
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0
)

plt.tight_layout()
plt.show()


### PRIM

from ema_workbench.analysis import prim

x = uncertainty_df_numeric.drop(columns=[
    'max_radicalized', 
    'final_radicalized', 
    'wage_cost', 
    'conservation_cost', 
    'final_resource']

# ['max_radicalized'] < 0.3
y = uncertainty_df_numeric['max_radicalized'] < 0.3
prim_alg = prim.Prim(x, y, threshold=0.8)
box1 = prim_alg.find_box()
box1.show_tradeoff()
box1.inspect(1, style='table')
box1.inspect(4, style='table')

# ['wage_cost'] < 1
y = uncertainty_df_numeric['wage_cost'] < 1
prim_alg = prim.Prim(x, y, threshold=0.6)
box1 = prim_alg.find_box()
box1.show_tradeoff()
box1.inspect(3, style='table')
box1.inspect(9, style='table')

# ['conservation_cost'] < 280
y = uncertainty_df_numeric['conservation_cost'] < 255
prim_alg = prim.Prim(x, y, threshold=0.6)
box1 = prim_alg.find_box()
box1.show_tradeoff()
box1.inspect(1, style='table')
box1.inspect(3, style='table')

# ['final_resource'] > 1.8
y = uncertainty_df_numeric['final_resource'] > 1.8
prim_alg = prim.Prim(x, y, threshold=0.6)
box1 = prim_alg.find_box()
box1.show_tradeoff()
box1.inspect(1, style='table')
box1.inspect(45, style='table')


### Feature Scoring
from ema_workbench.analysis import feature_scoring

outcome_columns = ['max_radicalized', 'final_radicalized', 'wage_cost', 'conservation_cost', 'final_resource']
filtered_df = uncertainty_df_numeric.dropna(subset=outcome_columns)
x_fs = filtered_df.drop(columns=outcome_columns, errors='ignore')
y_fs = {col: filtered_df[col] for col in outcome_columns}
fs = feature_scoring.get_feature_scores_all(x_fs, y_fs)

sns.heatmap(fs, cmap="viridis", annot=True)
plt.title("Feature Scoring Heatmap")
plt.show()