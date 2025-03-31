import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import pickle
import sys

class SIRModel:
    def __init__(self, pt_original=False, initialize_SIR=False, show_SIR_variation=True, enable_SDT=True, verbose=True):
        self.pt_original = pt_original
        self.initialize_SIR = initialize_SIR
        self.show_SIR_variation = show_SIR_variation
        self.enable_SDT = enable_SDT
        self.verbose = verbose

        # Model parameters
        self.T_ages = 45  # Number of age classes
        self.sigma_0 = 0.0005
        self.a_0 = 0.1
        self.a_max = 1
        self.gamma = 1
        self.delta = 0.5
        self.tau = 10
        self.period_n = 1
        self.I_0 = 0.1
        self.R_0 = 0.0
        self.S0 = np.zeros((self.T_ages, self.period_n))
        self.I0 = np.zeros((self.T_ages, self.period_n))
        self.R0 = np.zeros((self.T_ages, self.period_n))
        self.SIR_starts = [0]

        if self.initialize_SIR:
            self.initialize_parameters()
        else:
            self.load_initial_state()

        if self.show_SIR_variation:
            self.SIR_increment = 5
            self.SIR_starts = np.arange(self.T_ages, self.S0.shape[1], self.SIR_increment)
        else:
            self.SIR_increment = 0
            self.SIR_starts = [min(self.T_ages, self.S0.shape[1] - 1)]

        # SDT parameters
        self.mu_0 = 0.3 / 100
        self.a_w = 1.0
        self.a_e = 0.5 * 100
        self.YB_year = 0
        self.age_factor = 0.2
        self.w_0 = 0.90  # Base reference wage
        self.w_m = 0.75
        self.e_0 = 0.01
        self.show_year = 2020

        # Fixed period range
        self.period = np.arange(1810, 2101)

        # If SDT disabled, remove w and YB effects
        if not self.enable_SDT:
            self.a_w, self.a_e, self.YB_year = 0.0, 0.0, 0

        # Youth Bulge (YB)
        self.YB_A20 = np.zeros(len(self.period))
        if self.YB_year:
            self.YB_A20 = self.age_factor * np.exp(-((self.period - self.YB_year) ** 2) / 10**2)
        else:
            self.YB_year = '(None)'

        # Production function parameters
        # Introduce productivity parameters for each group:
        self.phi_S = 1.0
        self.phi_I = 0.8
        self.phi_R = 1.2

        # Cobb-Douglas parameters
        self.alpha = 0.4   # Capital share
        self.beta = 0.4    # Labor share (so 1 - alpha - beta = 0.2 is resource share)
        self.A = 1.0       # Total Factor Productivity (constant for now)
        self.R_resource = 1.0  # Fixed natural resource endowment

        # Capital accumulation parameters
        self.K_0 = 0.5  # Initial capital
        self.s = 0.2    # Savings (reinvestment) rate out of capital income
        self.delta = 0.05  # Depreciation rate

        # Initialize capital and arrays
        self.K = np.full(len(self.period), np.nan)  # Storing capital over time
        self.K[0] = self.K_0

        # Initialize results storage
        self.results = []

        if self.verbose:
            print("Model initialization complete.")

    def initialize_parameters(self):
        self.S0[0, :] = (1 - self.I_0 - self.R_0) / self.T_ages
        self.I0[0, :] = self.I_0 / self.T_ages
        self.R0[0, :] = self.R_0 / self.T_ages
        self.SIR_starts = [0]

    def load_initial_state(self):
        try:
            with open('initial_SIR.pkl', 'rb') as fh:
                if sys.version_info[:3] >= (3, 0):
                    self.T_ages, self.a_0, self.a_max, self.gamma, self.sigma_0, self.delta, self.tau, self.S0, self.I0, self.R0 = pickle.load(fh, encoding='latin1')
                else:
                    self.T_ages, self.a_0, self.a_max, self.gamma, self.sigma_0, self.delta, self.tau, self.S0, self.I0, self.R0 = pickle.load(fh)
        except:
            raise RuntimeError("Unable to load initial_SIR.pkl!")

    def simulate(self, S0, I0, R0):
        # Initialize compartments for full simulation period
        S = np.zeros((self.T_ages, len(self.period)))
        I = np.zeros((self.T_ages, len(self.period)))
        R = np.zeros((self.T_ages, len(self.period)))

        S[:, 0], I[:, 0], R[:, 0] = S0, I0, R0

        S_sum, I_sum, R_sum, alpha_rec = [np.full(len(self.period), np.nan) for _ in range(4)]
        S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:, 0]), np.sum(I[:, 0]), np.sum(R[:, 0])

        # Elite fraction and epsilon arrays
        elit = np.full(len(self.period), np.nan)
        elit[0] = self.e_0  # Start from baseline elite fraction
        epsln = np.full(len(self.period), np.nan)
        epsln[0] = (1 - self.w_0)/elit[0]  # Just as a placeholder from old logic

        # Arrays for storing wages, output
        w_array = np.full(len(self.period), np.nan)
        Y_array = np.full(len(self.period), np.nan)

        # Start with given K from self.K (capital)
        K = self.K.copy()

        for t in range(len(self.period) - 1):
            t1 = t + 1

            # Compute effective labor L(t)
            L_t = self.phi_S * S_sum[t] + self.phi_I * I_sum[t] + self.phi_R * R_sum[t]

            # Production this period
            Y_t = self.A * (K[t]**self.alpha) * (L_t**self.beta) * (self.R_resource**(1 - self.alpha - self.beta))
            Y_array[t] = Y_t

            # Wage per effective labor unit (marginal product of labor)
            # dY/dL = A * K^alpha * beta * L^{beta-1} * R^{1-alpha-beta}
            w_t = self.A * (K[t]**self.alpha) * self.beta * (L_t**(self.beta - 1)) * (self.R_resource**(1 - self.alpha - self.beta))
            w_array[t] = w_t

            # Update elite fraction
            if self.pt_original:
                elit[t1] = elit[t] + self.mu_0*(self.w_0 - w_t)/w_t
                elit[t1] -= (elit[t1]-self.e_0)*R_sum[t]  # old logic
            else:
                elit[t1] = elit[t] + self.mu_0*(self.w_0 - w_t)/w_t - (elit[t]-self.e_0)*R_sum[t]

            # Update epsln (just for compatibility with old code)
            epsln[t1] = (1 - w_t)/elit[t1]

            # Compute alpha (radicalization rate) now using endogenous w_t
            # Keep old logic but replace w[t1] with w_t:
            alpha = np.clip(self.a_0 + self.a_w*(self.w_0 - w_t) + self.a_e*(elit[t1]-self.e_0) + self.YB_A20[t1],
                            self.a_0, self.a_max)

            # sigma and rho as before
            sigma = np.clip((alpha - self.gamma * R_sum[t]) * I_sum[t] + self.sigma_0, 0, 1)
            rho = np.clip(self.delta * I_sum[t - self.tau] if t > self.tau else 0, 0, 1)
            alpha_rec[t1] = alpha

            # Age-structured updates (like in original code)
            S[0, t1] = 1 / self.T_ages
            for age in range(self.T_ages - 1):
                age1 = age + 1
                S[age1, t1] = (1 - sigma)*S[age, t]
                I[age1, t1] = (1 - rho)*I[age, t] + sigma*S[age, t]
                R[age1, t1] = R[age, t] + rho*I[age, t]

            S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(S[:, t1]), np.sum(I[:, t1]), np.sum(R[:, t1])

            if self.pt_original:
                elit[t1] -= (elit[t1]-self.e_0)*R_sum[t1]

            # Capital accumulation
            # Elites earn alpha*Y_t
            # They save fraction s of it
            # Capital next period:
            K[t1] = (1 - self.delta)*K[t] + self.s * self.alpha * Y_t

        # Compute last step output/wages
        # (for completeness if needed)
        L_t_end = self.phi_S * S_sum[-1] + self.phi_I * I_sum[-1] + self.phi_R * R_sum[-1]
        Y_t_end = self.A * (K[-1]**self.alpha) * (L_t_end**self.beta) * (self.R_resource**(1 - self.alpha - self.beta))
        w_t_end = self.A * (K[-1]**self.alpha) * self.beta * (L_t_end**(self.beta - 1)) * (self.R_resource**(1 - self.alpha - self.beta))
        Y_array[-1] = Y_t_end
        w_array[-1] = w_t_end

        return S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln, w_array, Y_array, K

    def run_model(self):
        self.results = []
        for y_i in self.SIR_starts:
            y_i = int(y_i)
            S0, I0, R0 = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]
            S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln, w_array, Y_array, K_array = self.simulate(S0, I0, R0)
            # Update self.K with the final K_array from the simulation
            self.K = K_array.copy()
            self.results.append({
                'S': S, 'I': I, 'R': R, 'S_sum': S_sum, 'I_sum': I_sum,
                'R_sum': R_sum, 'alpha_rec': alpha_rec, 'elit': elit, 'epsln': epsln, 
                'w': w_array, 'Y': Y_array, 'K': K_array
            })

    def plot_results(self):
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        brown, nice_green = '#A52A2A', '#00BFFF'

        # We no longer have exogenous w array. We'll plot w_array from results.
        for result in self.results:
            S_sum = result['S_sum']
            I_sum = result['I_sum']
            R_sum = result['R_sum']
            alpha_rec = result['alpha_rec']
            elit = result['elit']
            w_array = result['w']
            # YB_A20 was kept for logic
            YB_A20 = self.YB_A20

            if self.enable_SDT:
                w_ax1, = ax1.plot(self.period, w_array, color=brown, linestyle='-', linewidth=2, label='(w) Endogenous wage')
                YB_ax1, = ax1.plot(self.period, YB_A20, color='k', linestyle='-.', linewidth=2, label=f'Youth bulge {self.YB_year}')
                elit_ax1, = ax1.plot(self.period, elit * 100, color='b', linestyle='-', label='(e*100) Elite fraction')
                alpha_ax1, = ax1.plot(self.period, alpha_rec, color=nice_green, linestyle='--', linewidth=2, label=r'($\alpha$) radicalization rate')
                if self.show_year in self.period:
                    ax1.axvline(x=self.show_year, color='k', linestyle='--')

            I_ax1, = ax1.plot(self.period, I_sum, color='r', linestyle='-', label='(I) Radical fraction')
            R_ax1, = ax1.plot(self.period, R_sum, color='k', linestyle='-', label='(R) Moderate fraction')

        if self.enable_SDT:
            handles = [w_ax1, YB_ax1, elit_ax1, alpha_ax1, I_ax1, R_ax1]
        else:
            handles = [I_ax1, R_ax1]
        ax1.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        ax1.set_xlabel('Year', fontsize=12)
        ax1.grid(True)
        ax1.set_ylim([-0.1, 2.5])
        ax1.set_xlim([self.period[0], self.period[-1]])
        ax1.set_title(f'Model with Endogenous Wages, Capital Accumulation, and Resource Constraint')
        plt.show()

    def introduce_shock(self, shock_type, shock_year, shock_magnitude):
        """
        Introduce a shock to capital, resources, elite numbers, or productivity.

        :param shock_type: Type of shock ('capital', 'resource', 'elite', 'productivity')
        :param shock_year: Year when the shock occurs
        :param shock_magnitude: Magnitude of the shock
        """
        shock_index = np.where(self.period == shock_year)[0]
        if len(shock_index) == 0:
            raise ValueError("Invalid shock_year. Year not found in the period range.")
        shock_index = shock_index[0]

        # Apply shock to the appropriate variable
        if shock_type == 'capital':
            self.K[shock_index:] += shock_magnitude
        elif shock_type == 'resource':
            # Ensure resource does not go negative
            new_val = self.R_resource + shock_magnitude
            self.R_resource = max(new_val, 0)
        elif shock_type == 'elite':
            # Increase/decrease baseline elite fraction
            self.e_0 += shock_magnitude
        elif shock_type == 'productivity':
            # Increase/decrease baseline productivity
            self.A += shock_magnitude
        else:
            raise ValueError("Invalid shock_type. Must be 'capital', 'resource', 'elite', or 'productivity'.")

    def compare_with_shock(self, shock_type, shock_year, shock_magnitude):
        """
        Compare the model trajectories with and without a shock.

        :param shock_type: Type of shock ('capital', 'resource', 'elite', 'productivity')
        :param shock_year: Year when the shock occurs
        :param shock_magnitude: Magnitude of the shock
        """

        # Store original parameters so we can restore them after running baseline
        original_A = self.A
        original_e0 = self.e_0
        original_R_resource = self.R_resource
        original_K = self.K.copy()

        y_i = int(self.SIR_starts[0])
        S0, I0, R0 = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]

        # 1. Run baseline model
        self.A = original_A
        self.e_0 = original_e0
        self.R_resource = original_R_resource
        self.K = original_K.copy()

        self.run_model()
        baseline_results = self.results[0]

        # 2. Introduce shock and re-run model
        self.A = original_A
        self.e_0 = original_e0
        self.R_resource = original_R_resource
        self.K = original_K.copy()

        self.introduce_shock(shock_type, shock_year, shock_magnitude)
        self.run_model()
        shocked_results = self.results[0]

        # Extract results for plotting
        period = self.period
        baseline_w = baseline_results['w']
        shocked_w = shocked_results['w']
        baseline_elit = baseline_results['elit']
        shocked_elit = shocked_results['elit']

        baseline_I = baseline_results['I_sum']
        shocked_I = shocked_results['I_sum']

        # Plot comparison
        fig, ax = plt.subplots(3, 1, figsize=(12, 16))

        # Plot wages
        ax[0].plot(period, baseline_w, label='Wages (Baseline)', color='b')
        ax[0].plot(period, shocked_w, label='Wages (Shock)', color='r', linestyle='--')
        ax[0].axvline(shock_year, color='k', linestyle=':', label='Shock Year')
        ax[0].set_title(f'Impact of {shock_type.capitalize()} Shock on Wages (Magnitude: {shock_magnitude})')
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Wage Level')
        ax[0].legend()
        ax[0].grid(True)

        # Plot elite dynamics
        ax[1].plot(period, baseline_elit, label='Elite Fraction (Baseline)', color='g')
        ax[1].plot(period, shocked_elit, label='Elite Fraction (Shock)', color='orange', linestyle='--')
        ax[1].axvline(shock_year, color='k', linestyle=':', label='Shock Year')
        ax[1].set_title(f'Impact of {shock_type.capitalize()} Shock on Elite Fraction (Magnitude: {shock_magnitude})')
        ax[1].set_xlabel('Year')
        ax[1].set_ylabel('Elite Fraction')
        ax[1].legend()
        ax[1].grid(True)

        # Plot radicalization (I_sum)
        ax[2].plot(period, baseline_I, label='Radical Fraction (Baseline)', color='r')
        ax[2].plot(period, shocked_I, label='Radical Fraction (Shock)', color='r', linestyle='--')
        ax[2].axvline(shock_year, color='k', linestyle=':', label='Shock Year')
        ax[2].set_title(f'Impact of {shock_type.capitalize()} Shock on Radical Fraction (Magnitude: {shock_magnitude})')
        ax[2].set_xlabel('Year')
        ax[2].set_ylabel('Radical Fraction')
        ax[2].legend()
        ax[2].grid(True)

        fig.suptitle(f'Comparison of Model Outcomes with and without {shock_type.capitalize()} Shock')
        plt.tight_layout()
        plt.show()

        # Restore parameters after plotting if you plan to use the model again
        self.A = original_A
        self.e_0 = original_e0
        self.R_resource = original_R_resource
        self.K = original_K.copy()

    @staticmethod
    def save_pkl_file(filename, data_tuple):
        with open(filename, "wb") as fh:
            pickle.dump(data_tuple, fh)
        return True

    @staticmethod
    def load_pkl_file(filename):
        try:
            with open(filename, "rb") as fh:
                if sys.version_info[:3] >= (3, 0):
                    data_tuple = pickle.load(fh, encoding='latin1')
                else:
                    data_tuple = pickle.load(fh)
        except:
            raise RuntimeError(f"Unable to load {filename}!")
        return data_tuple

### Example usage:
model = SIRModel(initialize_SIR=False, show_SIR_variation=False, enable_SDT=True)
model.run_model()
model.plot_results()

### Shock Analysis
model.compare_with_shock('productivity', 1900, 0.2)
model.compare_with_shock('resource', 1950, -0.5)
model.compare_with_shock('capital', 1840, 0.8)
model.compare_with_shock('elite', 1860, 0.02)