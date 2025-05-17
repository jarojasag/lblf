import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import parcoords   

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
        self.conservation_effort = 0.0 # Effort
        self.conservation_unit_cost = 5.0  # Cost
        self.conservation_effectiveness = 0.5 # Maximum reduction in depletion due to conservation

    def set_policy_parameters(
            self, 
            eta_w, 
            eta_a, 
            w_T, 
            I_Ta, 
            conservation_effort=0.0,
            conservation_unit_cost=5.0,
            conservation_effectiveness=0.5
        ):
        self.eta_w = eta_w
        self.eta_a = np.array(eta_a)
        self.w_T = w_T
        self.I_Ta = np.array(I_Ta)
        self.conservation_effort = conservation_effort
        self.conservation_unit_cost = conservation_unit_cost
        self.conservation_effectiveness = conservation_effectiveness

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
            self.conservation_cost += self.conservation_effort * self.conservation_unit_cost

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
            dep_t *= (1 - self.conservation_effectiveness * self.conservation_effort)
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

    ## RADICALIZATION DRIVERS
    a_w = kwargs.get('a_w', 1.0)      # α_w : effect size of wage deviation on radicalization
    a_e = kwargs.get('a_e', 50.0)     # α_e : effect size of elite-fraction deviation on radicalization

    ## RESPONSE CURVATURE
    eta_w = kwargs.get('eta_w', 0.0)  # η_w : exponent for wage sensitivity in radicalization
    eta_a = kwargs.get('eta_a', 0.0)  # η_a : exponent for elite-fraction sensitivity in radicalization

    ## BASELINES & THRESHOLDS
    w_T  = kwargs.get('w_T', 0.8)     # w_T  : “normal” (target) wage level
    I_Ta = kwargs.get('I_Ta', 0.1)    # I_Ta : radicalization threshold share

    ## CONSERVATION POLICY LEVERS
    conservation_effort = kwargs.get('conservation_effort', 0.0)  # 0 to 1
    conservation_unit_cost = kwargs.get('conservation_unit_cost', 5.0)   # $ per unit of conservation action
    conservation_effectiveness = kwargs.get('conservation_effectiveness', 0.5)  # depletion reduction per unit invested

    ## SOCIO-ECOLOGICAL COUPLINGS
    eta_deplet    = kwargs.get('eta_deplet', 1.0)  # exponent linking wage to resource depletion pressure
    mu_elite_extr = kwargs.get('mu_elite_extr', 0.5)  # extraction multiplier driven by elites

    ## SOCIAL-STRUCTURE DYNAMICS
    mu_0 = kwargs.get('mu_0', 0.003)  # μ₀ : annual upward-mobility rate into the elite class
    e_0  = kwargs.get('e_0', 0.01)    # e₀ : long-run expected elite fraction

    ## RESOURCE SYSTEM
    nat_res_regen = kwargs.get('nat_res_regen', 0.05)  # r : intrinsic regeneration rate of the resource
    delta_extract = kwargs.get('delta_extract', 0.03)  # δ : baseline extraction rate (fraction of stock removed per year)

    model = PolicySIRModel(
        initialize_SIR=False,
        show_SIR_variation=False,
        enable_SDT=True,
        verbose=False
    )

    model.a_w, model.a_e = a_w, a_e
    model.nat_res_regen, model.delta_extract = nat_res_regen, delta_extract
    model.eta_deplet, model.mu_elite_extr = eta_deplet, mu_elite_extr
    model.mu_0, model.e_0 = mu_0, e_0

    # refresh derived surplus factor after changing e₀
    model.eps_factor = (1 - model.w_0) / model.e_0

    # policy levers by cohort
    eta_a_array = np.full(model.T_ages, eta_a)
    I_Ta_array  = np.full(model.T_ages, I_Ta)

    # Policy Parameters
    model.set_policy_parameters(
        eta_w, eta_a_array, w_T, I_Ta_array,
        conservation_effort=conservation_effort,
        conservation_unit_cost=conservation_unit_cost,
        conservation_effectiveness=conservation_effectiveness
    )

    model.run_model()
    res = model.results[0]

    I_sum = res['I_sum']
    return dict(
        max_radicalized   = np.max(I_sum),
        final_radicalized = I_sum[-1],
        wage_cost         = res['wage_cost'],
        conservation_cost = res['conservation_cost'],
        final_resource    = res['nat_res_array'][-1]
    )