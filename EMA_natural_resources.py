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
from ema_workbench.em_framework.samplers import (
    LHSSampler, sample_levers)
from ema_workbench.analysis import parcoords
from ema_workbench.analysis import prim
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools

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

ordered_cols = list(rename_map.values()) 

base_df_numeric = pd.read_csv("EMA_Output_BaseRun.csv")
#base_df_numeric = base_df_numeric.drop(columns="Unnamed: 0")
base_df_numeric = base_df_numeric[base_df_numeric["final_resource"] >= 0] 

uncertainty_df_numeric = pd.read_csv("EMA_Output_LeverRun.csv")
#uncertainty_df_numeric = uncertainty_df_numeric.drop(columns="Unnamed: 0")
uncertainty_df_numeric = uncertainty_df_numeric[uncertainty_df_numeric["final_resource"] >= 0] 

uncertainty_df_numeric = uncertainty_df_numeric.rename(columns=rename_map)
uncertainty_df_numeric = uncertainty_df_numeric[ordered_cols]

base_df_numeric = base_df_numeric.rename(columns=rename_map)
base_df_numeric = base_df_numeric[ordered_cols]

### Plotting 

def plot_highlighted_experiments(
        df: pd.DataFrame,
        outcome_col: str,
        global_limits: None,
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

    if global_limits is None:
        limits = parcoords.get_limits(df)
    else:
        limits = global_limits

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

global_limits = parcoords.get_limits(uncertainty_df_numeric)

# Base

plot_highlighted_experiments(base_df_numeric, "Peak Radicalization", global_limits = global_limits, n=2000, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Peak Radicalization", n=500, mode="high", color="red")

plot_highlighted_experiments(base_df_numeric, "Final Radicalization", n=20, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Final Radicalization", n=20, mode="high", color="red")

plot_highlighted_experiments(base_df_numeric, "Wage Cost", n=20, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Wage Cost", n=20, mode="high", color="red")

plot_highlighted_experiments(base_df_numeric, "Conservation Cost", n=20, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Conservation Cost", n=20, mode="high", color="red")

plot_highlighted_experiments(base_df_numeric, "Final Resource", n=20, mode="low", color="blue")
plot_highlighted_experiments(base_df_numeric, "Final Resource", n=20, mode="high", color="red")


# Levers

plot_highlighted_experiments(uncertainty_df_numeric, "Peak Radicalization", global_limits = global_limits, n=2000, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Peak Radicalization", n=2000, mode="high", color="red")

plot_highlighted_experiments(uncertainty_df_numeric, "Final Radicalization", n=20, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Final Radicalization", n=20, mode="high", color="red")

plot_highlighted_experiments(uncertainty_df_numeric, "Wage Cost", n=20, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Wage Cost", n=20, mode="high", color="red")

plot_highlighted_experiments(uncertainty_df_numeric, "Conservation Cost", n=20, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Conservation Cost", n=20, mode="high", color="red")

plot_highlighted_experiments(uncertainty_df_numeric, "Final Resource", n=20, mode="low", color="blue")
plot_highlighted_experiments(uncertainty_df_numeric, "Final Resource", n=20, mode="high", color="red")

# Characterizing Successful Policies in Stay the Course

uncertainty_cols = [
    "Wage → Radicalization", "Elite → Radicalization",
    "Resource Regeneration", "Extraction Rate", "De-radicalization Rate",
    "Wage–Resource Exp.", "Conservation Effectiveness",
    "Conservation Unit Cost", "Wage→Depletion Exp.",
    "Elite Extraction Mult.", "Elite Mobility Rate", "Elite Proportion"
]

X = base_df_numeric[uncertainty_cols]
success = (base_df_numeric["Peak Radicalization"] < 0.10).values
fail = (base_df_numeric["Peak Radicalization"] > 0.90).values

# Success
prim_alg = prim.Prim(X, success, threshold=0.8)
box      = prim_alg.find_box()

box.show_tradeoff()
plt.show()

box.inspect(1)
box.inspect(10)
box.inspect(9)
box.inspect(1, style="graph")
plt.show()

# Fail 
prim_alg_f = prim.Prim(X, fail, threshold=0.8)
box_f      = prim_alg_f.find_box()

box_f.show_tradeoff()
plt.show()


box_f.inspect(1)
box_f.inspect(5))
box_f.inspect(10)
box_f.inspect(30)
box_f.inspect(60)
box_f.inspect(1, style="graph")
plt.show()

# Successful policies

uncertainty_cols_orig = [
    "a_w",              # Wage → Radicalization
    "a_e",              # Elite → Radicalization
    "nat_res_regen",    # Resource Regeneration
    "delta_extract",    # Extraction Rate
    "delta",            # De-radicalization Rate
    "alpha_w",          # Wage–Resource Exp.
    "conservation_effectiveness",
    "conservation_unit_cost",
    "eta_deplet",       # Wage→Depletion Exp.
    "mu_elite_extr",    # Elite Extraction Mult.
    "mu_0",             # Elite Mobility Rate
    "e_0"               # Elite Proportion
]

lever_cols = [
    "Wage Target Sensitivity",
    "Radical. Treshold Sensitivity",
    "Target Wage",
    "Radical Threshold",
    "Conservation Effort"
]

failed_scenarios = base_df_numeric[base_df_numeric["max_radicalized"] > 0.9]

def dataframe_to_scenarios(df: pd.DataFrame,
                           cols: list[str],
                           prefix: str = "fail") -> list[Scenario]:
    """Turn each row of `df[cols]` into an EMA-Workbench Scenario."""
    return [
        Scenario(f"{prefix}_{i}", **row[cols].to_dict())
        for i, row in df.iterrows()
    ]

failed_scenarios = dataframe_to_scenarios(failed_scenarios, uncertainty_cols_orig)

with SequentialEvaluator(ema_model) as evaluator:
    experiments_f, outcomes_f = evaluator.perform_experiments(
        scenarios=failed_scenarios,
        policies=25,
    )

experiments_f = pd.DataFrame.from_dict(experiments_f)
outcomes_f = pd.DataFrame.from_dict(outcomes_f)
fails_df = pd.concat([experiments_f, outcomes_f], axis=1)
fails_df = fails_df.rename(columns=rename_map)
fails_df = fails_df.select_dtypes(include=[float, int])
fails_df = fails_df.replace([np.inf, -np.inf], np.nan)
fails_df = fails_df.dropna()

plot_highlighted_experiments(fails_df, "Peak Radicalization", n=2500, mode="low", color="blue")
plot_highlighted_experiments(fails_df, "Peak Radicalization", n=1000, mode="high", color="red")

X_crisis = fails_df[uncertainty_cols + lever_cols]
y_crisis = (fails_df["Peak Radicalization"] < 0.10).values

prim_alg_c = prim.Prim(X_crisis, y_crisis, threshold=0.8)
box_c      = prim_alg_c.find_box()

box_c.show_tradeoff()
plt.show()

box_c.inspect(1)
box_c.inspect(3)
box_c.inspect(5)
box_c.inspect(10)
box_c.inspect(15)
box_c.inspect(30)
plt.show()

# What Drives Policy failure?

X_crisis_f = fails_df[uncertainty_cols + lever_cols]
y_crisis_f = (fails_df["Peak Radicalization"] > 0.90).values

prim_alg_cf = prim.Prim(X_crisis_f, y_crisis_f, threshold=0.8)
box_cf      = prim_alg_cf.find_box()

box_cf.show_tradeoff()
plt.show()

box_cf.inspect(1)
box_cf.inspect(3)
box_cf.inspect(5)
box_cf.inspect(10)
box_cf.inspect(15)
box_cf.inspect(30)
plt.show()


# Pareto Front
uncertainty_df_numeric["Total Policy Cost"] = uncertainty_df_numeric["Wage Cost"] + uncertainty_df_numeric["Conservation Cost"] 
objs = uncertainty_df_numeric[["Peak Radicalization", "Total Policy Cost"]].values


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

dom    = uncertainty_df_numeric[uncertainty_df_numeric.pareto == "dominated"]
front  = uncertainty_df_numeric[uncertainty_df_numeric.pareto == "front"]

sns.set_style("whitegrid")
plt.figure(figsize=(7, 5))

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

plt.legend(
    title="Status",
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0
)

plt.tight_layout()
plt.show()
