import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator
import matplotlib.patches as mpatches
import seaborn as sns
from ema_workbench import (Model, RealParameter, CategoricalParameter, BooleanParameter, 
                           ScalarOutcome, Policy, SequentialEvaluator, Scenario)
from ema_workbench.em_framework.samplers import (
    LHSSampler, sample_levers)
from ema_workbench.analysis import parcoords
from ema_workbench.analysis import prim
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools
import sys
import os

from core_model import CoreClimateInequalityModel

np.random.seed(17)
random.seed(17)

def prim_analysis(X, cat_dict, threshold=0.8, mass_min=0.05, 
                  box_steps=2, print_graphs=False):

    y = cat_dict
    prim_model = prim.Prim(X, y, threshold = threshold, mass_min = mass_min)

    box = prim_model.find_box()
    box.show_tradeoff()
    box_list = list(range(0, len(box.box_lims), box_steps))
    for b in box_list: 
        box.inspect(b)
        if print_graphs:
            box.inspect(b, style="graph")
    
    return box

def climate_inequality_model_wrapper(
    # UNCERTAINTIES (X)
    # Base radicalization rate 
    a_0=0.1,
    # Inequality sensitivity
    a_inequality=1.0,
    # De-radicalization delay
    tau=10,
    # Climate damage coefficient
    eta=0.00236,
    # Adaptation effectiveness
    kappa=0.6,
    # Moderation effectiveness
    gamma=1.0,
    # Trust erosion rates
    omega=0.04,
    # Economic growth rate
    g=0.03,
    # Return on capital
    r=0.05,
    
    # POLICY LEVERS (L) 
    # Carbon tax implementation (Boolean)
    carbon_tax=False,
    # Wealth tax implementation (Boolean)
    wealth_tax=False,
    # Social programs implementation (Boolean)
    social_programs=False,
    # Policy start year (relative to simulation start)
    policy_start=5
):
    """
    Wrapper function for the CoreClimateInequalityModel to interface with EMA Workbench.
    """
    
    try:
        # Initialize model with temporary config modifications
        model = CoreClimateInequalityModel()
        
        # Update model parameters based on uncertainties
        # Base radicalization parameters
        model.config['radicalization']['a_0'] = a_0
        model.config['radicalization']['a_inequality'] = a_inequality
        model.config['radicalization']['tau'] = int(tau)
        model.config['radicalization']['gamma'] = gamma
        
        # Climate parameters
        model.config['climate']['damage_functions']['nordhaus_coeff'] = eta
        model.config['climate']['kappa_adaptation'] = kappa
        
        # Trust parameters
        model.config['trust']['omega_beta'] = omega
        model.config['trust']['omega_damage'] = omega * 1.25  # Slightly higher for climate
        model.config['trust']['omega_I'] = omega * 2.0  # Highest for radicalization
        
        # Economic parameters
        model.config['economic']['growth_rate_base'] = g
        model.config['economic']['return_on_capital'] = r
        
        # Determine active policies based on boolean levers
        policies = []
        if carbon_tax:
            policies.append('carbon_tax')
        if wealth_tax:
            policies.append('wealth_tax') 
        if social_programs:
            policies.append('social_programs')
        
        # Run simulation
        time_horizon = 46  # 2020-2066
        start_year = 2020
        policy_start_year = start_year + int(policy_start)
        
        results = model.simulate_age_structured(
            time_horizon=time_horizon,
            start_year=start_year,
            policies=policies if policies else None,
            policy_start_year=policy_start_year if policies else None
        )
        
        PSI_series = np.zeros(len(results['years']))
        for i in range(len(results['years'])):
            PSI_series[i], _, _, _ = model.compute_political_stress_index(
                results['I_sum'][i], 
                results['elit'][i], 
                results['trust'][i], 
                results['gini'][i],
                results['debt_ratio'][i]
            )
        
        crisis_time = len(results['years'])  # Default: no crisis
        crisis_indices = np.where(PSI_series > 1.0)[0]
        if len(crisis_indices) > 0:
            crisis_time = crisis_indices[0]
        
        # Maximum radicalization (peak instability)
        max_radicalization = np.max(results['I_sum'])
        
        # Cumulative climate damage (economic loss)
        cumulative_damage = np.trapz(results['damage'], dx=1) / len(results['damage'])
        
        # Final temperature anomaly
        final_temperature = results['temperature'][-1]
        
        # Final Gini coefficient
        final_gini = results['gini'][-1]
        
        # Trust collapse indicator (binary)
        trust_collapse = 1.0 if np.min(results['trust']) < 0.2 else 0.0
        
        # Youth vs adult radicalization ratio
        youth_ages = model.T_ages // 3  # First third of ages = "youth"
        final_youth_radical = np.sum(results['I'][:youth_ages, -1])
        final_adult_radical = np.sum(results['I'][youth_ages:, -1])
        youth_adult_ratio = (final_youth_radical / final_adult_radical) if final_adult_radical > 1e-6 else 0.0
        
        # Policy cost-effectiveness
        total_policy_cost = np.sum(results['policy_spending'])
        radicalization_reduction = max(0, 0.05 - max_radicalization)  # Reduction from baseline expectation
        if total_policy_cost > 1e-6:
            policy_effectiveness = radicalization_reduction / total_policy_cost
        else:
            policy_effectiveness = radicalization_reduction  # Free reduction (no policy scenario)
        
        # Return all outcomes as dictionary
        return {
            'time_to_crisis': crisis_time,
            'max_radicalization': max_radicalization,
            'cumulative_damage': cumulative_damage,
            'final_temperature': final_temperature,
            'final_gini': final_gini,
            'trust_collapse': trust_collapse,
            'youth_adult_ratio': youth_adult_ratio,
            'policy_effectiveness': policy_effectiveness
        }
        
    except Exception as e:
        # Return safe default values in case of simulation failure
        return {
            'time_to_crisis': 999,  # No crisis
            'max_radicalization': 0.05,  # Baseline
            'cumulative_damage': 0.5,  # High damage
            'final_temperature': 5.0,  # High temperature
            'final_gini': 0.7,  # High inequality
            'trust_collapse': 1.0,  # Trust collapsed
            'youth_adult_ratio': 2.0,  # Youth dominated
            'policy_effectiveness': 0.0  # No effectiveness
        }

# Create EMA Model
ema_model = Model('ClimateInequalityModel', function=climate_inequality_model_wrapper)

# UNCERTAINTIES (X) - Following XLRM framework
ema_model.uncertainties = [
    # Base radicalization rate (a_0)
    RealParameter('a_0', 0.05, 0.5), 
    
    # Inequality sensitivity (a_inequality)
    RealParameter('a_inequality', 0.5, 2.0),
    
    # De-radicalization delay (τ)
    RealParameter('tau', 5, 20),
    
    # Climate damage coefficient (η)
    RealParameter('eta', 0.001, 0.01),
    
    # Adaptation effectiveness (κ)
    RealParameter('kappa', 0.3, 0.8),
    
    # Moderation effectiveness (γ)
    RealParameter('gamma', 0.5, 1.5),
    
    # Trust erosion rates (ω)
    RealParameter('omega', 0.01, 0.1),
    
    # Economic growth rate (g)
    RealParameter('g', 0.01, 0.05),
    
    # Return on capital (r)
    RealParameter('r', 0.03, 0.07)
]

# POLICY LEVERS (L) - Following XLRM framework
ema_model.levers = [
    # Carbon tax implementation (Boolean)
    BooleanParameter('carbon_tax'),
    
    # Wealth tax implementation (Boolean)
    BooleanParameter('wealth_tax'),
    
    # Social programs implementation (Boolean)
    BooleanParameter('social_programs'),
    
    # Policy start year
    RealParameter('policy_start', 1, 15)
]

# OUTCOMES/METRICS (M) - Following XLRM framework
ema_model.outcomes = [
    ScalarOutcome('time_to_crisis'),      # Time to PSI > 1 (political crisis threshold)
    ScalarOutcome('max_radicalization'),  # Maximum radicalization (L_sum peak)
    ScalarOutcome('cumulative_damage'),   # Cumulative climate damage (economic loss)
    ScalarOutcome('final_temperature'),   # Final temperature anomaly
    ScalarOutcome('final_gini'),          # Final Gini coefficient
    ScalarOutcome('trust_collapse'),      # Trust collapse (binary)
    ScalarOutcome('youth_adult_ratio'),   # Youth vs adult radicalization ratio
    ScalarOutcome('policy_effectiveness') # Policy cost-effectiveness
]

# =============================================================================
# BASELINE ANALYSIS 
# =============================================================================

print("Running baseline analysis (no policy intervention)...")

# Baseline Policy: All boolean levers set to False, others to minimum
baseline_policy = Policy("baseline", 
                        carbon_tax=False,
                        wealth_tax=False, 
                        social_programs=False,
                        policy_start=1,
                        policy_intensity=0.1)

# Run baseline experiments
with SequentialEvaluator(ema_model) as evaluator:
    experiments_base, outcomes_base = evaluator.perform_experiments(
        scenarios=20000,             
        policies=baseline_policy
    )

# Convert to DataFrames and clean
scenarios = [Scenario(f"scen_{i}", **row.drop(['policy']).to_dict())
    for i, row in experiments_base.iterrows()]

experiments_base_df = pd.DataFrame.from_dict(experiments_base)
outcomes_base_df = pd.DataFrame.from_dict(outcomes_base)
baseline_df = pd.concat([experiments_base_df, outcomes_base_df], axis=1)

# Clean data
baseline_numeric = baseline_df.select_dtypes(include=[float, int])
baseline_numeric = baseline_numeric.replace([np.inf, -np.inf], np.nan)
baseline_numeric = baseline_numeric.dropna()


# =============================================================================
# Levers Off
# =============================================================================

baseline_pol = Policy("baseline", **{lv.name: 0 for lv in ema_model.levers})

with SequentialEvaluator(ema_model) as evaluator:
    experiments_policy, outcomes_policy = evaluator.perform_experiments(
        scenarios=scenarios, 
        policies=baseline_pol  # Explore baseline 
    )

scenarios = [Scenario(f"scen_{i}", **row.drop(['policy']).to_dict())
    for i, row in experiments_policy.iterrows()] # Saving Xs 

experiments_policy_df = pd.DataFrame.from_dict(experiments_policy)
outcomes_policy_df = pd.DataFrame.from_dict(outcomes_policy)
uncertainty_base_df = pd.concat([experiments_policy_df, outcomes_policy_df], axis=1)

base_df_numeric = uncertainty_base_df.select_dtypes(include=[float, int, bool])
base_df_numeric = base_df_numeric.replace([np.inf, -np.inf], np.nan)

base_df_numeric.to_csv('EMA_Output_BaseRun_Climate.csv', index=False)

# =============================================================================
# Levers On
# =============================================================================

with SequentialEvaluator(ema_model) as evaluator:
    experiments_unc, outcomes_unc = evaluator.perform_experiments(scenarios=scenarios, policies=30)

results_unc_df = pd.DataFrame.from_dict(experiments_unc)
outcomes_unc_df = pd.DataFrame.from_dict(outcomes_unc)
uncertainty_df = pd.concat([results_unc_df, outcomes_unc_df], axis=1)

uncertainty_df_numeric = uncertainty_df.select_dtypes(include=[float, int, bool])
uncertainty_df_numeric = uncertainty_df_numeric.replace([np.inf, -np.inf], np.nan)
uncertainty_df_numeric = uncertainty_df_numeric.dropna()
uncertainty_df_numeric.to_csv('EMA_Output_LeverRun.csv', index=False)

# Read after closing
uncertainty_df_numeric = pd.read_csv('EMA_Output_LeverRun.csv')
base_df_numeric = pd.read_csv('EMA_Output_BaseRun_Climate.csv')

# =============================================================================
# RENAMING AND STANDARDIZATION
# =============================================================================

# Create human-readable variable names following XLRM framework
rename_map = {
    # ═══════════════════ UNCERTAINTIES (X) ═══════════════════
    "a_0": "Base Radicalization",
    "a_inequality": "Inequality Effect",
    "tau": "Belief Lag (yr)",
    "gamma": "Moderation Power",
    "eta": "Climate Damage",
    "kappa": "Adaptation",
    "omega": "Trust Erosion",
    "g": "Growth Rate",
    "r": "Capital Return",
    
    # ═══════════════════ POLICY LEVERS (L) ═══════════════════
    "carbon_tax": "Carbon Tax",
    "wealth_tax": "Wealth Tax",
    "social_programs": "Social Programs",
    "policy_start": "Policy Delay (yr)",
    
    # ═══════════════════ OUTCOME METRICS (M) ═══════════════════
    "time_to_crisis": "Time to PSI>1",
    "max_radicalization": "Peak Radicalization",
    "cumulative_damage": "Avg Climate Loss",
    "final_temperature": "Final Temp (°C)",
    "final_gini": "Final Gini",
    "trust_collapse": "Trust <20%",
    "youth_adult_ratio": "Age Radicalization Bias",
    "policy_effectiveness": "Cost-Benefit Ratio"
}

# Apply renaming and reordering
ordered_cols = list(rename_map.values())

baseline_numeric = base_df_numeric.rename(columns=rename_map)
baseline_numeric = baseline_numeric[ordered_cols]

policy_numeric = uncertainty_df_numeric.rename(columns=rename_map)
policy_numeric = policy_numeric[ordered_cols]

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_highlighted_experiments(
        df: pd.DataFrame,
        outcome_col: str,
        global_limits=None,
        n: int = 20,                 # how many runs to highlight
        mode: str = "low",           # 'low' or 'high'
        color: str = "blue",
        decimals: int = 1,           # rounding
        fig_width: float = 16,
        fig_height: float = None,
        label_fontsize: int = 8,
        background_sample: int = 1000,
        line_width_bg: float = 0.4,
        line_width_fg: float = 1.2):
    """
    Parallel-coordinates plot highlighting best/worst experiments.
    Adapted for climate-inequality XLRM framework.
    """

    # Subset to highlight
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

    # Random background sample
    bg = df.sample(background_sample, random_state=0) if len(df) > background_sample else df

    # Create plot
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

    # Color-code by XLRM component
    n_uncert = 9    # First 9 columns = uncertainties (X)
    n_levers = 4    # Next 5 columns = levers (L)

    colors_xlrm = {
        'uncertainty': '#fff2dc',   # Warm beige for uncertainties (X)
        'lever':       '#dcf2ff',   # Cool blue for levers (L)  
        'outcome':     '#e6f7e6'    # Green for outcomes/metrics (M)
    }

    for i, ax in enumerate(paraxes.axes):
        if i < n_uncert:
            ax.patch.set_facecolor(colors_xlrm['uncertainty'])
        elif i < n_uncert + n_levers:
            ax.patch.set_facecolor(colors_xlrm['lever'])
        else:
            ax.patch.set_facecolor(colors_xlrm['outcome'])
        ax.patch.set_zorder(0)          

    # Add XLRM legend
    handles = [
        mpatches.Patch(color=colors_xlrm['uncertainty'], label='Uncertainties (X)'),
        mpatches.Patch(color=colors_xlrm['lever'], label='Levers (L)'),
        mpatches.Patch(color=colors_xlrm['outcome'], label='Metrics (M)'),
    ]
    plt.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.45), ncol=3,
               frameon=False, fontsize=label_fontsize)

    plt.suptitle(f"{n} Experiments with {sel} '{outcome_col}'",
                 x=0.02, ha='left', fontsize=16, weight='bold') 
    plt.tight_layout()
    plt.show()

    return limits

# =============================================================================
# PARALLEL COORDINATES ANALYSIS
# =============================================================================

print("Creating parallel coordinates plots...")

# Get global limits for consistent plotting
global_limits = parcoords.get_limits(policy_numeric)

# Baseline Analysis Plots
print("Baseline scenario analysis:")

plot_highlighted_experiments(baseline_numeric, 'Peak Radicalization', 
                            global_limits=global_limits, n=50, mode="low", color="green")
plot_highlighted_experiments(baseline_numeric, "Peak Radicalization", 
                            global_limits=global_limits, n=50, mode="high", color="red")

plot_highlighted_experiments(baseline_numeric, "Avg Climate Loss", 
                            global_limits=global_limits, n=30, mode="high", color="green")
plot_highlighted_experiments(baseline_numeric, "Avg Climate Loss", 
                            global_limits=global_limits, n=30, mode="low", color="red")

plot_highlighted_experiments(baseline_numeric, "Final Temp (°C)", 
                            global_limits=global_limits, n=30, mode="low", color="green")
plot_highlighted_experiments(baseline_numeric, "Final Temp (°C)", 
                            global_limits=global_limits, n=30, mode="high", color="red")

# Policy Exploration Plots
print("Policy intervention analysis:")

plot_highlighted_experiments(policy_numeric, "Peak Radicalization", 
                            global_limits=global_limits, n=100, mode="low", color="blue")
plot_highlighted_experiments(policy_numeric, "Peak Radicalization", 
                            global_limits=global_limits, n=100, mode="high", color="red")

plot_highlighted_experiments(policy_numeric, "Avg Climate Loss", 
                            global_limits=global_limits, n=50, mode="high", color="blue")
plot_highlighted_experiments(policy_numeric, "Avg Climate Loss", 
                            global_limits=global_limits, n=50, mode="low", color="red")

plot_highlighted_experiments(policy_numeric, "Final Temp (°C)", 
                            global_limits=global_limits, n=50, mode="low", color="blue")
plot_highlighted_experiments(policy_numeric, "Final Temp (°C)", 
                            global_limits=global_limits, n=50, mode="high", color="red")



# =============================================================================
# PRIM ANALYSIS - CHARACTERIZING SUCCESS AND FAILURE
# =============================================================================

print("Running PRIM analysis to identify success and failure patterns...")

# Define uncertainty columns for PRIM analysis
uncertainty_cols = [
    "Base Radicalization",
    "Inequality Effect",
    "Belief Lag (yr)",
    "Moderation Power",
    "Climate Damage",
    "Adaptation",
    "Trust Erosion",
    "Growth Rate",
    "Capital Return"
]

# Baseline Success/Failure Analysis
print("Analyzing baseline scenarios...")
X_baseline = baseline_numeric[uncertainty_cols]

# Define success: 
success_baseline = (
    (baseline_numeric["Final Temp (°C)"] < 1.95) & 
    (baseline_numeric["Peak Radicalization"] < 0.3)
).values
np.sum(success_baseline)

# Define failure: high radicalization OR early crisis
failure_baseline = (
    (baseline_numeric["Final Temp (°C)"] >= 1.95) |
    (baseline_numeric["Peak Radicalization"] >= 0.3)
).values
np.sum(failure_baseline)

succcess_base = prim_analysis(X_baseline, success_baseline)
fail_base = prim_analysis(X_baseline, failure_baseline)


# Include both uncertainties and levers for policy analysis
lever_cols = [
    "Carbon Tax",
    "Wealth Tax",
    "Social Programs",
    "Policy Delay (yr)"
]
  
X_policy = policy_numeric[uncertainty_cols + lever_cols]



# Define success: 
success_policy= (
    (policy_numeric["Final Temp (°C)"] < 1.75) & 
    (policy_numeric["Peak Radicalization"] < 0.3)
).values
np.sum(success_policy)

# Define failure: high radicalization OR early crisis
failure_policy = (
    (policy_numeric["Final Temp (°C)"] >= 1.7) |
    (policy_numeric["Peak Radicalization"] >= 0.3)
).values
np.sum(failure_policy)

success_policy_analysis = prim_analysis(X_policy, success_policy)
failure_policy_analysis = prim_analysis(X_policy, failure_policy)





# Define success: 
success_ineq= (
    (policy_numeric["Final Gini"] < 0.49) 
).values
np.sum(success_ineq)

# Define failure: high inequality
failure_ineq = (
    (policy_numeric["Final Gini"] >= 0.49)
).values
np.sum(failure_ineq)

success_ineq_analysis = prim_analysis(X_policy, success_ineq)
failure_ineq_analysis = prim_analysis(X_policy, failure_ineq)





# Define success: 
success_loss = (
    (policy_numeric["Avg Climate Loss"] < 0.004127) 
).values
np.sum(success_loss)
\
# Define failure: high climate loss 
failure_loss = (
    (policy_numeric["Avg Climate Loss"] >= 0.004127)
).values
np.sum(failure_loss)

success_loss_analysis = prim_analysis(X_policy, success_loss)
failure_loss_analysis = prim_analysis(X_policy, failure_loss)