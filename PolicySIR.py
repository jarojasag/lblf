from lblf_SIR import SIRModel
import numpy as np
import matplotlib.pyplot as plt
from ema_workbench import (Model, RealParameter, ScalarOutcome, ema_logging, Constant)
from ema_workbench.em_framework import SequentialEvaluator
from ema_workbench.em_framework.optimization import (EpsilonProgress, HyperVolume)
from ema_workbench.analysis import parcoords
import pandas as pd

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