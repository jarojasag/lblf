import numpy as np
from lblf_SIR_natural_resources import SIRModel

class PolicySIRModel(SIRModel):
    """
    Extends the resource-based SIRModel from lblf_SIR_natural_resources.py
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Wage and radicalization parameters
        self.eta_w = 0.0                         # Sensitivity to wage shortfalls
        self.eta_a = np.zeros(self.T_ages)       # Sensitivity to radicalization thresholds by cohort
        self.w_T = 0.0                           # Target wage
        self.I_Ta = np.zeros(self.T_ages)        # Tolerable radicalization fraction by cohort

        self.delta_w = np.zeros(len(self.period))  # Time-series of wage adjustments
        self.wage_cost = 0.0

        # Conservation parameters
        self.conservation_effort = 0.0           # Initial (fixed) conservation effort c₀
        self.conservation_unit_cost = 5.0        # Cost per unit of conservation action
        self.conservation_effectiveness = 0.5    # Max reduction in depletion due to conservation

        # Adaptive conservation settings
        self.phi_c = 0.0                         # Sensitivity of conservation to shortfall φ_c
        self.R_star = None                       # Target resource floor R*
        self.adaptive_conservation = False       # Flag to switch to adaptive mode
        self.conservation_cost = 0.0             # Total cost of conservation actions

    def set_policy_parameters(
        self,
        eta_w,
        eta_a,
        w_T,
        I_Ta,
        conservation_effort=0.0,
        conservation_unit_cost=5.0,
        conservation_effectiveness=0.5,
        phi_c=0.0,
        R_star=None,
        adaptive_conservation=False
    ):
        # Wage and radicalization
        self.eta_w = eta_w
        self.eta_a = np.array(eta_a)
        self.w_T = w_T
        self.I_Ta = np.array(I_Ta)

        # Conservation (fixed)
        self.conservation_effort = conservation_effort
        self.conservation_unit_cost = conservation_unit_cost
        self.conservation_effectiveness = conservation_effectiveness

        # Adaptive settings
        self.phi_c = phi_c
        self.R_star = R_star
        self.adaptive_conservation = adaptive_conservation

        if self.adaptive_conservation and self.R_star is None:
            raise ValueError("Must provide R_star for adaptive conservation mode")

    def simulate(self, S0, I0, R0, shock_type=None, shock_year=None, shock_magnitude=0.0):
        """
        Expanded simulate supporting optional adaptive conservation
        """
        # Initialize compartments
        S = np.zeros((self.T_ages, len(self.period)))
        I = np.zeros((self.T_ages, len(self.period)))
        R = np.zeros((self.T_ages, len(self.period)))
        S[:, 0], I[:, 0], R[:, 0] = S0, I0, R0

        S_sum = np.full(len(self.period), np.nan)
        I_sum = np.full(len(self.period), np.nan)
        R_sum = np.full(len(self.period), np.nan)
        alpha_rec = np.full(len(self.period), np.nan)
        S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:, 0]), np.sum(I[:, 0]), np.sum(R[:, 0])

        # Elites
        elit = np.full(len(self.period), np.nan)
        elit[0] = self.e_0
        epsln = np.full(len(self.period), np.nan)

        # Resources
        nat_res_local = self.nat_res_array.copy()

        # Wage series
        w = np.full(len(self.period), np.nan)
        w[0] = self.wage_function(nat_res_local[0], elit[0])
        epsln[0] = self.eps_factor * (1 - w[0]) / elit[0]

        # Reset costs and deltas
        self.wage_cost = 0.0
        self.conservation_cost = 0.0
        self.delta_w = np.zeros(len(self.period))

        # Conservation effort series (fixed or adaptive)
        c_series = np.full(len(self.period), self.conservation_effort)

        for t in range(len(self.period) - 1):
            t1 = t + 1

            # Wage adjustments
            delta_w_t = self.eta_w * max(self.w_T - w[t], 0)
            over_threshold = np.maximum(I[:, t] - self.I_Ta, 0)
            delta_w_t += np.sum(self.eta_a * over_threshold)
            self.delta_w[t] = delta_w_t
            w[t] += delta_w_t
            self.wage_cost += abs(delta_w_t)

            # Adaptive conservation update
            if self.adaptive_conservation:
                shortfall = max(self.R_star - nat_res_local[t], 0)
                c_series[t1] = c_series[t] + self.phi_c * shortfall

            # Conservation cost
            self.conservation_cost += c_series[t1] * self.conservation_unit_cost

            # Elite update
            if self.pt_original:
                elit[t1] = elit[t] + self.mu_0 * (self.w_0 - w[t]) / w[t]
                elit[t1] -= (elit[t1] - self.e_0) * R_sum[t]
            else:
                elit[t1] = (
                    elit[t]
                    + self.mu_0 * (self.w_0 - w[t]) / w[t]
                    - (elit[t] - self.e_0) * R_sum[t]
                )
            elit[t1] = np.clip(elit[t1], 0, 1)

            # Wage and elite surplus update
            w[t1] = self.wage_function(nat_res_local[t], elit[t1])
            epsln[t1] = self.eps_factor * (1 - w[t1]) / elit[t1]

            # Political Stress Index
            alpha = np.clip(
                self.a_0
                + self.a_w * (self.w_0 - w[t1])
                + self.a_e * (elit[t1] - self.e_0)
                + self.YB_A20[t1],
                self.a_0, self.a_max
            )
            alpha_rec[t1] = alpha

            sigma = np.clip((alpha - self.gamma * np.sum(R[:, t])) * np.sum(I[:, t]) + self.sigma_0, 0, 1)
            rho = np.clip(self.delta * (I_sum[t - self.tau] if t > self.tau else 0), 0, 1)

            # Age cohort updates
            S[0, t1] = 1 / self.T_ages
            for age in range(self.T_ages - 1):
                S[age + 1, t1] = (1 - sigma) * S[age, t]
                I[age + 1, t1] = (1 - rho) * I[age, t] + sigma * S[age, t]
                R[age + 1, t1] = R[age, t] + rho * I[age, t]

            S_sum[t1], I_sum[t1], R_sum[t1] = (
                np.sum(S[:, t1]), np.sum(I[:, t1]), np.sum(R[:, t1])
            )

            if self.pt_original:
                elit[t1] -= (elit[t1] - self.e_0) * R_sum[t1]
                elit[t1] = np.clip(elit[t1], 0, 1)

            # Resource depletion with conservation
            dep_t = self.depletion_function(w[t], elit[t])
            dep_t *= (1 - self.conservation_effectiveness * c_series[t1])
            nat_res_local[t1] = self.resource_update(nat_res_local[t], dep_t)

        return (
            S, I, R, S_sum, I_sum, R_sum,
            alpha_rec, elit, epsln, w,
            nat_res_local, c_series
        )

    def run_model(self, shock_type=None, shock_year=None, shock_magnitude=0.0):
        """
        Call simulate() in new class.
        """
        self.results = []
        for y_i in self.SIR_starts:
            y_i = int(y_i)
            S0, I0, R0 = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]

            (S, I, R,
                S_sum, I_sum, R_sum,
                alpha_rec, elit, epsln,
                w, nat_res_local, c_series
            ) = self.simulate(
                S0, I0, R0,
                shock_type=shock_type,
                shock_year=shock_year,
                shock_magnitude=shock_magnitude
            )

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
                'YB_A20': self.YB_A20,
                'conservation_series': c_series,
                'wage_cost': self.wage_cost,
                'conservation_cost': self.conservation_cost
            })

""" 
# Test Run
model = PolicySIRModel(
    pt_original=False,
    quick_adjust=False,
    initialize_SIR=False,
    show_SIR_variation=False,
    enable_SDT=True,
    verbose=False
)

# Policy parameters
model.set_policy_parameters(
    eta_w=0.1,
    eta_a=[0.05] * model.T_ages,
    w_T=1.0,
    I_Ta=[0.1] * model.T_ages,
    conservation_effort=0.2,
    conservation_unit_cost=5.0,
    conservation_effectiveness=0.5,
    phi_c=0.01,
    R_star=0.5,
    adaptive_conservation=True
)

model.run_model()
model.plot_results() 
"""