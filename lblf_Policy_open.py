from lblf_SIR import SIRModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ema_workbench import (Model, RealParameter, ScalarOutcome, ema_logging, Constant)
from ema_workbench.em_framework import SequentialEvaluator
from ema_workbench.em_framework.optimization import (EpsilonProgress, HyperVolume)
from ema_workbench.analysis import parcoords, prim, pairs_plotting, feature_scoring, dimensional_stacking
import pandas as pd
import seaborn as sns

#  logging
ema_logging.log_to_stderr(ema_logging.INFO)

class PolicySIRModel(SIRModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize policy parameters
        self.eta_w = 0.0  # Sensitivity of the controller to w_T - w(t)
        self.eta_a = np.zeros(self.T_ages)  # Sensitivity to I_a(t) - I_Ta
        self.w_T = 0.0  # Target level for income available to workers
        self.I_Ta = np.zeros(self.T_ages)  # Target level for tolerable radicalization in cohort a
        self.policy_cost = 0.0  # Initialize policy cost
        self.delta_w = np.zeros(len(self.period))  # Policy adjustment over time

    def set_policy_parameters(self, eta_w, eta_a, w_T, I_Ta):
        """Policy parameters."""
        self.eta_w = eta_w
        self.eta_a = np.array(eta_a)
        self.w_T = w_T
        self.I_Ta = np.array(I_Ta)

    def simulate(self, S0, I0, R0, w):
        """Run the SIR model simulation with policy adjustments."""
        S = np.zeros((self.T_ages, len(self.period)))
        I = np.zeros((self.T_ages, len(self.period)))
        R = np.zeros((self.T_ages, len(self.period)))

        S[:, 0], I[:, 0], R[:, 0] = S0, I0, R0

        S_sum, I_sum, R_sum, alpha_rec = [np.full(len(self.period), np.nan) for _ in range(4)]
        S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:, 0]), np.sum(I[:, 0]), np.sum(R[:, 0])

        # Initialize local copies of elit and epsln
        elit = np.full(len(self.period), np.nan)
        elit[0] = self.e_0  # Initial proportion of elite
        epsln = np.full(len(self.period), np.nan)
        epsln[0] = self.eps_factor * (1 - w[0]) / elit[0]

        delta_w = np.zeros(len(self.period))
        policy_cost = 0.0

        for t in range(len(self.period) - 1):
            t1 = t + 1
            # Update elit[t1]
            if self.pt_original:
                elit[t1] = elit[t] + self.mu_0 * (self.w_0 - w[t]) / w[t]
                elit[t1] -= (elit[t1] - self.e_0) * R_sum[t1]
            else:
                elit[t1] = elit[t] + self.mu_0 * (self.w_0 - w[t]) / w[t] - (elit[t] - self.e_0) * R_sum[t]
            # Update epsln
            epsln[t1] = self.eps_factor * (1 - w[t1]) / elit[t1]

            # Implement the policy adjustment Δw(t)
            delta_w_t = self.eta_w * max(self.w_T - w[t], 0)
            delta_w_t += np.sum(self.eta_a * np.maximum(I[:, t] - self.I_Ta, 0))
            delta_w[t] = delta_w_t
            w[t] += delta_w_t  # Adjust w[t] with the policy adjustment
            policy_cost += abs(delta_w_t)  # Accumulate policy cost (absolute value)

            # Calculate alpha
            alpha = np.clip(self.a_0 + self.a_w * (self.w_0 - w[t]) + self.a_e * (elit[t] - self.e_0) + self.YB_A20[t], self.a_0, self.a_max)
            # Calculate sigma and rho
            sigma = np.clip((alpha - self.gamma * np.sum(R[:, t])) * np.sum(I[:, t]) + self.sigma_0, 0, 1)
            # Use fixed delta value (delta = 0.5 as per initial setup)
            delta_fixed = 0.5
            rho = np.clip(delta_fixed * np.sum(I[:, t - self.tau]) if t > self.tau else 0, 0, 1)
            alpha_rec[t1] = alpha

            # Update S, I, R
            S[0, t1] = 1 / self.T_ages  # Birth of new susceptibles
            for age in range(self.T_ages - 1):
                age1 = age + 1
                S[age1, t1] = (1 - sigma) * S[age, t]
                I[age1, t1] = (1 - rho) * I[age, t] + sigma * S[age, t]
                R[age1, t1] = R[age, t] + rho * I[age, t]
            S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(S[:, t1]), np.sum(I[:, t1]), np.sum(R[:, t1])
            if self.pt_original:
                elit[t1] -= (elit[t1] - self.e_0) * R_sum[t1]

        self.policy_cost = policy_cost  # Store total policy cost
        self.delta_w = delta_w  # Store policy adjustments
        return S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln

def policySIRModel(**kwargs):
    """Model function for ema_workbench."""
    # Extract parameters from kwargs
    a_w = kwargs['a_w']
    a_e = kwargs['a_e']
    eta_w = kwargs['eta_w']
    eta_a = kwargs['eta_a']
    w_T = kwargs['w_T']
    I_Ta = kwargs['I_Ta']

    # Initialize the model
    model = PolicySIRModel(initialize_SIR=False, show_SIR_variation=False, enable_SDT=True, verbose=False)
    model.a_w = a_w
    model.a_e = a_e
    # delta is fixed within the simulate method (delta = 0.5)

    # Set policy parameters
    eta_a_array = np.full(model.T_ages, eta_a)
    I_Ta_array = np.full(model.T_ages, I_Ta)
    model.set_policy_parameters(eta_w, eta_a_array, w_T, I_Ta_array)

    # Run the model
    model.run_model()
    # Get the results
    result = model.results[0]  # Assuming only one result
    I_sum = result['I_sum']

    # Compute outcomes
    max_radicalized = np.max(I_sum)
    final_radicalized = I_sum[-1]
    policy_cost = model.policy_cost

    return {'max_radicalized': max_radicalized,
            'final_radicalized': final_radicalized,
            'policy_cost': policy_cost}

# Excample Run 

ema_model = Model('PolicySIR', function=policySIRModel)

# Uncertainties 
ema_model.uncertainties = [
    RealParameter('a_w', 0.5, 1.5),
    RealParameter('a_e', 25.0, 75.0)
 ]

# levers
ema_model.levers = [
    RealParameter('eta_w', 0.0, 1.0),
    RealParameter('eta_a', 0.0, 1.0),  # Assuming same eta_a for all ages
    RealParameter('w_T', 0.8, 1.0),
    RealParameter('I_Ta', 0.0, 0.1),  # Assuming same I_Ta for all ages
]

# Outcomes
ema_model.outcomes = [
    ScalarOutcome('max_radicalized'), # kind=ScalarOutcome.MINIMIZE),
    ScalarOutcome('final_radicalized'), # kind=ScalarOutcome.MINIMIZE),
    ScalarOutcome('policy_cost') #, kind=ScalarOutcome.MINIMIZE),
]



# Use SequentialEvaluator
with SequentialEvaluator(ema_model) as evaluator:
    results_uncertainty = evaluator.perform_experiments(scenarios=1000, policies=10)

# Analyze and visualize the uncertainty results
experiments_unc, outcomes_unc = results_uncertainty
results_unc_df = pd.DataFrame.from_dict(experiments_unc)
outcomes_unc_df = pd.DataFrame.from_dict(outcomes_unc)
uncertainty_df = pd.concat([results_unc_df, outcomes_unc_df], axis=1)

# Exclude the 'model' column and ensure only numeric columns are included
uncertainty_df_numeric = uncertainty_df.select_dtypes(include=[float, int])



# Parallel Coords Plot
parcoords_lims = parcoords.get_limits(uncertainty_df_numeric)
paraxes_unc = parcoords.ParallelAxes(parcoords_lims)
paraxes_unc.plot(uncertainty_df_numeric)
plt.title('Uncertainty Impact on Outcomes')
plt.show()


# Parallel Coords Plot v2
top_lowest_radicalized = uncertainty_df_numeric.nsmallest(20, 'max_radicalized')

# Parallel Coords Plot
parcoords_lims = parcoords.get_limits(uncertainty_df_numeric)
paraxes_unc = parcoords.ParallelAxes(parcoords_lims)

# Plot all lines in light gray
paraxes_unc.plot(uncertainty_df_numeric, color='lightgray')

# Overlay the top 5 lines with the lowest 'max_radicalized' in blue
paraxes_unc.plot(top_lowest_radicalized, color='blue')

# Set plot title
plt.title('Uncertainty Impact on Outcomes')
plt.show()


# Parallel Coords Plot v3
top_lowest_cost = uncertainty_df_numeric.nsmallest(20, 'policy_cost')

# Parallel Coords Plot
parcoords_lims = parcoords.get_limits(uncertainty_df_numeric)
paraxes_unc = parcoords.ParallelAxes(parcoords_lims)

# Plot all lines in light gray
paraxes_unc.plot(uncertainty_df_numeric, color='lightgray')

# Overlay the top 5 lines with the lowest 'max_radicalized' in blue
paraxes_unc.plot(top_lowest_cost, color='blue')

# Set plot title
plt.title('Uncertainty Impact on Outcomes')
plt.show()






# Pair Plot 0
fig, axes = pairs_plotting.pairs_scatter(experiments_unc, outcomes_unc)

# Adjust figure size for better visualization
fig.set_size_inches(8, 8)
plt.show()


# Pair Plot 1
fig, axes = pairs_plotting.pairs_scatter(experiments_unc, outcomes_unc, group_by='policy', legend=False)

# Adjust figure size for better visualization
fig.set_size_inches(8, 8)

# Adjust layout to make space for the custom legend on the right
fig.subplots_adjust(right=0.8)

# Define custom legend elements
policy_colors = ['blue', 'orange', 'green', 'purple', 'red', 'gray', 'black', 'pink', 'yellow', 'cyan']
policy_labels = [str(i) for i in range(1001, 1011)]
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, 
                          markerfacecolor=color, markersize=8) 
                   for label, color in zip(policy_labels, policy_colors)]

# Create the custom legend outside the plot
fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.85, 0.5), title="Policy")
fig.suptitle('Policies and Outcomes', fontsize=16)
plt.show()









# Prim 
x = uncertainty_df_numeric.drop(columns=['max_radicalized', 'final_radicalized', 'policy_cost'])  # Use only input parameters
y = uncertainty_df_numeric['max_radicalized'] < 0.8  # Define y based on threshold for max_radicalized outcome


# Initialize and run PRIM algorithm
prim_alg = prim.Prim(x, y, threshold=0.8)
box1 = prim_alg.find_box()

# Display the results
box1.show_tradeoff()
box1.inspect(10, style='table')  # Inspecting the first box

box1.inspect(10)
box1.inspect(10, style="graph")








# Feature Scoring 
outcome_columns = ['max_radicalized', 'final_radicalized', 'policy_cost']

x = uncertainty_df_numeric.drop(columns = outcome_columns) 
y = {col: uncertainty_df_numeric[col] for col in outcome_columns}


fs = feature_scoring.get_feature_scores_all(x, y)
sns.heatmap(fs, cmap="viridis", annot=True)
plt.show()








#Dimensional Stacking 

x = uncertainty_df_numeric.drop(columns=['max_radicalized', 'final_radicalized', 'policy_cost'])
y = uncertainty_df_numeric['max_radicalized'] < 0.8   # Define the condition for dimensional stacking

y_array = y.values  # Converts the Series to a numpy array

# Create the dimensional stacking plot with y as a numpy array

dimensional_stacking.create_pivot_plot(x, y_array, 3, nbins = 3)
plt.show()

