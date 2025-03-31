import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from ema_workbench import Model, RealParameter, ScalarOutcome , ema_logging
from ema_workbench.em_framework import SequentialEvaluator, Samplers
from ema_workbench.analysis import parcoords, pairs_plotting

from lblf_SIR_natural_resources import SIRModel

ema_logging.log_to_stderr(ema_logging.INFO)

class PolicySIRModel(SIRModel):
    """
    Extends the resource-based SIRModel from lblf_SIR_natural_resources.py
    to add: wage targeting, radicalization thresholds, minimum wage and conservation costs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.eta_w = 0.0 # Sensitivity to wage shortfalls
        self.eta_a = np.zeros(self.T_ages) # Sensitivity to radicalization thresholds by cohort
        self.w_T = 0.0  # Target wage
        self.I_Ta = np.zeros(self.T_ages) # Tolerable radicalization fraction by cohort

        self.delta_w = np.zeros(len(self.period)) # Time-series of wage adjustments
        self.wage_cost = 0.0
        self.conservation_cost = 0.0
        self.conservation_investment = 0.0 # Effort
        self.conservation_unit_cost = 5.0  # Cost

    def set_policy_parameters(
            self, 
            eta_w, 
            eta_a, 
            w_T, 
            I_Ta, 
            conservation_investment=0.0,
            conservation_unit_cost=5.0
        ):
        self.eta_w = eta_w
        self.eta_a = np.array(eta_a)
        self.w_T = w_T
        self.I_Ta = np.array(I_Ta)
        self.conservation_investment = conservation_investment
        self.conservation_unit_cost = conservation_unit_cost

    def simulate(self, S0, I0, R0):
        """
        Changing method for the expanded class
        """
        S = np.zeros((self.T_ages, len(self.period)))
        I = np.zeros((self.T_ages, len(self.period)))
        R = np.zeros((self.T_ages, len(self.period)))
        S[:, 0], I[:, 0], R[:, 0] = S0, I0, R0

        S_sum = np.full(len(self.period), np.nan)
        I_sum = np.full(len(self.period), np.nan)
        R_sum = np.full(len(self.period), np.nan)
        alpha_rec = np.full(len(self.period), np.nan)

        S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:, 0]), np.sum(I[:, 0]), np.sum(R[:, 0])

        elit = np.full(len(self.period), np.nan)
        elit[0] = self.e_0
        epsln = np.full(len(self.period), np.nan)

        nat_res_local = self.nat_res_array.copy()

        # Wages and Conservation
        w = np.full(len(self.period), np.nan)
        w[0] = self.wage_function(nat_res_local[0], elit[0])
        epsln[0] = self.eps_factor * (1 - w[0]) / elit[0]

        self.wage_cost = 0.0
        self.conservation_cost = 0.0
        self.delta_w = np.zeros(len(self.period))

        for t in range(len(self.period) - 1):
            t1 = t + 1

            # Minimum wage
            delta_w_t = self.eta_w * max(self.w_T - w[t], 0)
            # Radicalization threshold
            over_threshold = np.maximum(I[:, t] - self.I_Ta, 0)
            delta_w_t += np.sum(self.eta_a * over_threshold)

            self.delta_w[t] = delta_w_t

            w[t] += delta_w_t
            self.wage_cost += abs(delta_w_t)

            # Conservation cost
            self.conservation_cost += self.conservation_investment * self.conservation_unit_cost

            if self.pt_original:
                elit[t1] = elit[t] + self.mu_0 * (self.w_0 - w[t]) / w[t]
                elit[t1] -= (elit[t1] - self.e_0) * R_sum[t]
            else:
                elit[t1] = elit[t] + self.mu_0*(self.w_0 - w[t])/w[t] - (elit[t] - self.e_0)*R_sum[t]
            elit[t1] = np.clip(elit[t1], 0, 1)

            w[t1] = self.wage_function(nat_res_local[t], elit[t1])
            epsln[t1] = self.eps_factor * (1 - w[t1]) / elit[t1]

            alpha = np.clip(
                self.a_0
                + self.a_w * (self.w_0 - w[t1])
                + self.a_e * (elit[t1] - self.e_0)
                + self.YB_A20[t1],
                self.a_0, self.a_max
            )
            alpha_rec[t1] = alpha

            sigma = np.clip((alpha - self.gamma*np.sum(R[:, t])) * np.sum(I[:, t]) + self.sigma_0, 0, 1)
            rho = np.clip(self.delta * (I_sum[t - self.tau] if t > self.tau else 0), 0, 1)

            S[0, t1] = 1 / self.T_ages
            for age in range(self.T_ages - 1):
                S[age + 1, t1] = (1 - sigma) * S[age, t]
                I[age + 1, t1] = (1 - rho) * I[age, t] + sigma * S[age, t]
                R[age + 1, t1] = R[age, t] + rho * I[age, t]

            S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(S[:, t1]), np.sum(I[:, t1]), np.sum(R[:, t1])

            if self.pt_original:
                elit[t1] -= (elit[t1] - self.e_0) * R_sum[t1]
                elit[t1] = np.clip(elit[t1], 0, 1)

            # Resource depletion and partial conservation
            dep_t = self.depletion_function(w[t], elit[t])
            dep_t *= (1 - 0.5 * self.conservation_investment)
            nat_res_local[t1] = self.resource_update(nat_res_local[t], dep_t)

        return (
            S, I, R, S_sum, I_sum, R_sum, 
            alpha_rec, elit, epsln, w, nat_res_local
        )

    def run_model(self):
        """
        Call simulate() in new class.
        """
        self.results = []
        for y_i in self.SIR_starts:
            y_i = int(y_i)
            S0, I0, R0 = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]

            (
                S, I, R,
                S_sum, I_sum, R_sum,
                alpha_rec, elit, epsln,
                w, nat_res_local
            ) = self.simulate(S0, I0, R0)

            self.results.append({
                'S': S,
                'I': I,
                'R': R,
                'S_sum': S_sum,
                'I_sum': I_sum,
                'R_sum': R_sum,
                'alpha_rec': alpha_rec,
                'elit': elit,
                'epsln': epsln,
                'w': w,
                'nat_res_array': nat_res_local,
                'wage_cost': self.wage_cost,
                'conservation_cost': self.conservation_cost
            })


def policySIRModel(**kwargs):
    """
    Wrapper for the EMA Workbench calls. Instantiates PolicySIRModel, 
    sets parameters, runs it, and returns the outcomes
    """

    a_w = kwargs.get('a_w', 1.0)
    a_e = kwargs.get('a_e', 50.0)
    eta_w = kwargs.get('eta_w', 0.0)
    eta_a = kwargs.get('eta_a', 0.0)
    w_T = kwargs.get('w_T', 0.8)
    I_Ta = kwargs.get('I_Ta', 0.1)
    conservation_investment = kwargs.get('conservation_investment', 0.0)
    conservation_unit_cost = kwargs.get('conservation_unit_cost', 5.0)

    # Resource uncertainties
    nat_res_regen = kwargs.get('nat_res_regen', 0.05)
    delta_extract = kwargs.get('delta_extract', 0.03)

    model = PolicySIRModel(
        initialize_SIR=False,
        show_SIR_variation=False,
        enable_SDT=True,
        verbose=False
    )

    model.a_w = a_w
    model.a_e = a_e

    # Resource
    model.nat_res_regen = nat_res_regen
    model.delta_extract = delta_extract

    eta_a_array = np.full(model.T_ages, eta_a)
    I_Ta_array = np.full(model.T_ages, I_Ta)

    # Policy Parameters
    model.set_policy_parameters(
        eta_w,
        eta_a_array,
        w_T,
        I_Ta_array,
        conservation_investment=conservation_investment,
        conservation_unit_cost=conservation_unit_cost
    )

    model.run_model()
    result = model.results[0]

    I_sum = result['I_sum']
    max_radicalized = np.max(I_sum)
    final_radicalized = I_sum[-1]
    wage_cost = result['wage_cost']
    conservation_cost = result['conservation_cost']
    final_resource = result['nat_res_array'][-1]

    return {
        'max_radicalized': max_radicalized,
        'final_radicalized': final_radicalized,
        'wage_cost': wage_cost,
        'conservation_cost': conservation_cost,
        'final_resource': final_resource
    }


### Test Run
ema_model = Model('PolicySIR', function=policySIRModel)

# Uncertainties
ema_model.uncertainties = [
    RealParameter('a_w', 0.01, 3.0), # how wage deviation affects radicalization
    RealParameter('a_e', 10.0, 150.0), # how elite fraction affects radicalization
    RealParameter('nat_res_regen', 0.01,  0.2), # resource regeneration
    RealParameter('delta_extract', 0.005, 0.1), # resource extraction
    RealParameter('delta', 0.1, 1.0)
]

# Levers
ema_model.levers = [
    RealParameter('eta_w', 0.01, 2.0), # wage sensitivity
    RealParameter('eta_a', 0.01, 2.0),  # radicalization threshold policy
    RealParameter('w_T', 0.5, 1.0), # wage target
    RealParameter('I_Ta', 0.01, 0.3), # radicalization threshold for each cohort
    RealParameter('conservation_investment', 0.0, 1.0),
    RealParameter('conservation_unit_cost', 1.0, 15.0)
]

# Putcomes
ema_model.outcomes = [
    ScalarOutcome('max_radicalized'),
    ScalarOutcome('final_radicalized'),
    ScalarOutcome('wage_cost'),
    ScalarOutcome('conservation_cost'),
    ScalarOutcome('final_resource'),
]

### Exploration 

with SequentialEvaluator(ema_model) as evaluator:
    results_uncertainty = evaluator.perform_experiments(scenarios=500, policies=5)

experiments_unc, outcomes_unc = results_uncertainty
results_unc_df = pd.DataFrame.from_dict(experiments_unc)
outcomes_unc_df = pd.DataFrame.from_dict(outcomes_unc)
uncertainty_df = pd.concat([results_unc_df, outcomes_unc_df], axis=1)

uncertainty_df_numeric = uncertainty_df.select_dtypes(include=[float, int])
uncertainty_df_numeric.to_csv('EMA_Output.csv')


## Parallel plots 
parcoords_lims = parcoords.get_limits(uncertainty_df_numeric)
paraxes_unc = parcoords.ParallelAxes(parcoords_lims)
paraxes_unc.plot(uncertainty_df_numeric)
plt.title('Uncertainty Impact on Outcomes')
plt.show()

# Low Radicalization
top_lowest_radicalized = uncertainty_df_numeric.nsmallest(10, 'max_radicalized')

parcoords_lims = parcoords.get_limits(uncertainty_df_numeric)
paraxes_unc = parcoords.ParallelAxes(parcoords_lims, fontsize = 8)
paraxes_unc.plot(uncertainty_df_numeric, color='lightgray')
paraxes_unc.plot(top_lowest_radicalized, color='blue')

plt.title('Experiments with Lowest max_radicalized Highlighted')
plt.show()

# Low Wage Compensation Cost
top_lowest_wage = uncertainty_df_numeric.nsmallest(20, 'wage_cost')

parcoords_lims = parcoords.get_limits(uncertainty_df_numeric)
paraxes_unc = parcoords.ParallelAxes(parcoords_lims, fontsize = 8)

paraxes_unc.plot(uncertainty_df_numeric, color='lightgray')
paraxes_unc.plot(top_lowest_wage, color='blue')

plt.title('Experiments with Lowest Wage Cost')
plt.show()

# Lowest Conservation Cost 
lowest_conservation_cost = uncertainty_df_numeric.nsmallest(10, 'conservation_cost')
parcoords_lims = parcoords.get_limits(uncertainty_df_numeric)
paraxes_unc = parcoords.ParallelAxes(parcoords_lims, fontsize = 8)
paraxes_unc.plot(uncertainty_df_numeric, color='lightgray')
paraxes_unc.plot(lowest_conservation_cost, color='blue')

plt.title('Experiments with Lowest Conservation Cost Highlighted')
plt.show()

# Highest Final Resources
largest_final_resource = uncertainty_df_numeric.nlargest(10, 'final_resource')
parcoords_lims = parcoords.get_limits(uncertainty_df_numeric)
paraxes_unc = parcoords.ParallelAxes(parcoords_lims, fontsize = 8)
paraxes_unc.plot(uncertainty_df_numeric, color='lightgray')
paraxes_unc.plot(largest_final_resource, color='blue')

plt.title('Experiments with Largest final_resource Highlighted')
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