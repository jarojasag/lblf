import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import pickle
import sys

class SIRModel:
    def __init__(self, pt_original=False, initialize_SIR=False, show_SIR_variation=True, enable_SDT=True, verbose=True):
        # Model configuration flags:
        # pt_original: If True, uses original parameter logic for elites, wages, etc.
        # initialize_SIR: If True, calls initialize_parameters() to set up initial S, I, R from scratch.
        # show_SIR_variation: If True, introduces variation in SIR starting points over time.
        # enable_SDT: If True, enables structural demographic theory dynamics (affecting radicalization, elites, wages).
        # verbose: If True, prints status messages during initialization and runs.

        self.pt_original = pt_original
        self.initialize_SIR = initialize_SIR
        self.show_SIR_variation = show_SIR_variation
        self.enable_SDT = enable_SDT
        self.verbose = verbose

        # Demographic and epidemiological parameters:
        self.T_ages = 45     # Number of distinct age classes or cohorts in the population.
        self.sigma_0 = 0.0005 # Base spontaneous radicalization rate (small baseline rate with no other influences).
        self.a_0 = 0.1       # Base parameter influencing how radicalization responds to wage and elite fraction deviations.
        self.a_max = 1       # Maximum limit on the radicalization rate (alpha) after clipping.
        self.gamma = 1        # Influences how recovered (moderate) population affects radicalization rate.
        self.delta = 0.5      # Recovery/moderation rate fraction, how quickly infected (radicalized) become moderate.
        self.tau = 10         # Time delay parameter influencing how past radicalization affects current moderation.
        self.period_n = 1     # Number of periods to simulate per age class step (often 1 means yearly steps).
        self.I_0 = 0.1        # Initial fraction of infected (radicalized) at the start.
        self.R_0 = 0.0        # Initial fraction of recovered (moderate) at the start.

        # Initial state arrays:
        # S0, I0, R0 are distributions across T_ages for the initial period.
        self.S0 = np.zeros((self.T_ages, self.period_n))
        self.I0 = np.zeros((self.T_ages, self.period_n))
        self.R0 = np.zeros((self.T_ages, self.period_n))
        self.SIR_starts = [0] # Index of periods from which to start SIR simulations.

        if self.initialize_SIR:
            self.initialize_parameters()
        else:
            self.load_initial_state()

        # If show_SIR_variation is True, vary SIR starting index every few years:
        if self.show_SIR_variation:
            self.SIR_increment = 5
            self.SIR_starts = np.arange(self.T_ages, self.S0.shape[1], self.SIR_increment)
        else:
            self.SIR_increment = 0
            self.SIR_starts = [min(self.T_ages, self.S0.shape[1] - 1)]

        # Structural Demographic Theory (SDT) parameters:
        self.mu_0 = 0.3 / 100  # Baseline mobility or elite change rate parameter.
        self.a_w = 1.0         # Sensitivity of radicalization rate to wage deviations.
        self.a_e = 0.5 * 100   # Sensitivity of radicalization rate to elite fraction deviations.
        self.YB_year = 0       # Year of a hypothetical Youth Bulge peak (0 = none).
        self.age_factor = 0.2  # Scaling factor for youth bulge influence if present.
        self.w_0 = 0.90        # Base reference wage level (used as a benchmark).
        self.w_m = 0.75        # Another reference wage level used for comparisons.
        self.e_0 = 0.01        # Baseline elite fraction (starting point).
        self.show_year = 2020  # Year for a vertical reference line in plots.

        # Define simulation period (years):
        self.period = np.arange(1810, 2101)

        # If SDT is disabled, neutralize wage and youth bulge impacts on radicalization:
        if not self.enable_SDT:
            self.a_w, self.a_e, self.YB_year = 0.0, 0.0, 0

        # Youth Bulge (YB) array over time:
        # If YB_year != 0, creates a time-varying factor (e.g. Gaussian peak at YB_year).
        self.YB_A20 = np.zeros(len(self.period))
        if self.YB_year:
            self.YB_A20 = self.age_factor * np.exp(-((self.period - self.YB_year) ** 2) / 10**2)
        else:
            self.YB_year = '(None)'

        # Productivity parameters for each group:
        self.phi_S = 1.0  # Productivity factor for Susceptibles
        self.phi_I = 0.8  # Productivity factor for Infected/Radicalized (lower than S)
        self.phi_R = 1.2  # Productivity factor for Recovered/Moderate (slightly higher than S)

        # Cobb-Douglas parameters for production:
        self.alpha = 0.2   # Capital share of output
        self.beta = 0.7    # Labor share of output
        # The remaining share (1 - alpha - beta = 0.1) is resource share.
        self.A = 1.0       # Total Factor Productivity (baseline scaling factor)

        # Resource dynamics parameters:
        self.R_resource = 1.0  # Initial resource stock level
        self.R_max = 2.0       # Carrying capacity (maximum sustainable resource level)
        self.r_regen = 0.05    # Natural regeneration rate of the resource toward R_max
        self.c_extract = 0.03  # Fraction of output extracted as resource usage (exploitation rate per unit of output)

        # Capital accumulation parameters:
        self.K_0 = 0.5   # Initial capital stock
        self.s = 0.1    # Savings rate (fraction of capital income reinvested)
        self.delta = 0.06 # Capital depreciation rate per period

        # Initialize capital and resource arrays:
        self.K = np.full(len(self.period), np.nan)
        self.K[0] = self.K_0         # Capital at start of simulation
        self.R_array = np.full(len(self.period), np.nan)
        self.R_array[0] = self.R_resource # Resource stock at start of simulation

        # Results storage for simulation runs:
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
        S = np.zeros((self.T_ages, len(self.period)))
        I = np.zeros((self.T_ages, len(self.period)))
        R = np.zeros((self.T_ages, len(self.period)))

        S[:, 0], I[:, 0], R[:, 0] = S0, I0, R0

        S_sum, I_sum, R_sum, alpha_rec = [np.full(len(self.period), np.nan) for _ in range(4)]
        S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:, 0]), np.sum(I[:, 0]), np.sum(R[:, 0])

        elit = np.full(len(self.period), np.nan)
        elit[0] = self.e_0
        epsln = np.full(len(self.period), np.nan)
        epsln[0] = (1 - self.w_0)/elit[0]

        w_array = np.full(len(self.period), np.nan)
        Y_array = np.full(len(self.period), np.nan)

        K = self.K.copy()
        R_nat = self.R_array.copy()  # Resource stock array

        for t in range(len(self.period) - 1):
            t1 = t + 1
            L_t = self.phi_S * S_sum[t] + self.phi_I * I_sum[t] + self.phi_R * R_sum[t]

            # Production with current resource stock R_nat[t]
            Y_t = self.A * (K[t]**self.alpha) * (L_t**self.beta) * (R_nat[t]**(1 - self.alpha - self.beta))
            Y_array[t] = Y_t

            # Marginal product of labor (wage)
            w_t = self.A * (K[t]**self.alpha) * self.beta * (L_t**(self.beta - 1)) * (R_nat[t]**(1 - self.alpha - self.beta))
            w_array[t] = w_t

            # Update elite fraction
            if self.pt_original:
                elit[t1] = elit[t] + self.mu_0*(self.w_0 - w_t)/w_t
                elit[t1] -= (elit[t1]-self.e_0)*R_sum[t]
            else:
                elit[t1] = elit[t] + self.mu_0*(self.w_0 - w_t)/w_t - (elit[t]-self.e_0)*R_sum[t]

            epsln[t1] = (1 - w_t)/elit[t1]

            # Radicalization rate alpha
            alpha = np.clip(self.a_0 + self.a_w*(self.w_0 - w_t) + self.a_e*(elit[t1]-self.e_0) + self.YB_A20[t1],
                            self.a_0, self.a_max)

            sigma = np.clip((alpha - self.gamma * R_sum[t]) * I_sum[t] + self.sigma_0, 0, 1)
            rho = np.clip(self.delta * I_sum[t - self.tau] if t > self.tau else 0, 0, 1)
            alpha_rec[t1] = alpha

            # Age structure
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
            K[t1] = (1 - self.delta)*K[t] + self.s * self.alpha * Y_t

            # Resource dynamics
            # R_{t+1} = R_t + r*(R_max - R_t) - c_extract * Y_t
            R_nat[t1] = R_nat[t] + self.r_regen*(self.R_max - R_nat[t]) - self.c_extract*Y_t
            # Ensure resource does not go negative
            R_nat[t1] = max(R_nat[t1], 0.0)

        # Last step output/wages
        L_t_end = self.phi_S * S_sum[-1] + self.phi_I * I_sum[-1] + self.phi_R * R_sum[-1]
        Y_t_end = self.A * (K[-1]**self.alpha) * (L_t_end**self.beta) * (R_nat[-1]**(1 - self.alpha - self.beta))
        w_t_end = self.A * (K[-1]**self.alpha) * self.beta * (L_t_end**(self.beta - 1)) * (R_nat[-1]**(1 - self.alpha - self.beta))
        Y_array[-1] = Y_t_end
        w_array[-1] = w_t_end

        return S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln, w_array, Y_array, K, R_nat

    def run_model(self):
        self.results = []
        for y_i in self.SIR_starts:
            y_i = int(y_i)
            S0, I0, R0 = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]
            (S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln, 
             w_array, Y_array, K_array, R_nat) = self.simulate(S0, I0, R0)
            # Update self.K and self.R_array with the final arrays from simulation
            self.K = K_array.copy()
            self.R_array = R_nat.copy()
            self.results.append({
                'S': S, 'I': I, 'R': R, 'S_sum': S_sum, 'I_sum': I_sum,
                'R_sum': R_sum, 'alpha_rec': alpha_rec, 'elit': elit, 'epsln': epsln,
                'w': w_array, 'Y': Y_array, 'K': K_array, 'R_nat': R_nat
            })

    def plot_results(self):
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        brown, nice_green = '#A52A2A', '#00BFFF'

        for result in self.results:
            S_sum = result['S_sum']
            I_sum = result['I_sum']
            R_sum = result['R_sum']
            alpha_rec = result['alpha_rec']
            elit = result['elit']
            w_array = result['w']
            R_nat = result['R_nat']
            YB_A20 = self.YB_A20

            # Top plot: Wages, elites, alpha, I, R
            if self.enable_SDT:
                ax[0].plot(self.period, w_array, color=brown, linestyle='-', linewidth=2, label='(w) Endogenous wage')
                ax[0].plot(self.period, YB_A20, color='k', linestyle='-.', linewidth=2, label=f'Youth bulge {self.YB_year}')
                ax[0].plot(self.period, elit * 100, color='b', linestyle='-', label='(e*100) Elite fraction')
                ax[0].plot(self.period, alpha_rec, color=nice_green, linestyle='--', linewidth=2, label=r'($\alpha$) radicalization rate')
                if self.show_year in self.period:
                    ax[0].axvline(x=self.show_year, color='k', linestyle='--')

            ax[0].plot(self.period, I_sum, color='r', linestyle='-', label='(I) Radical fraction')
            ax[0].plot(self.period, R_sum, color='k', linestyle='-', label='(R) Moderate fraction')
            ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
            ax[0].set_xlabel('Year', fontsize=12)
            ax[0].grid(True)
            ax[0].set_ylim([-0.1, 2.5])
            ax[0].set_xlim([self.period[0], self.period[-1]])
            ax[0].set_title('Social and Economic Variables')

            # Bottom plot: Resource dynamics
            ax[1].plot(self.period, R_nat, color='green', linestyle='-', linewidth=2, label='Resource Stock')
            ax[1].axhline(self.R_max, color='gray', linestyle='--', label='R_max')
            ax[1].set_title('Natural Resource Dynamics')
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Resource Stock')
            ax[1].legend()
            ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    def introduce_shock(self, shock_type, shock_year, shock_magnitude):
        shock_index = np.where(self.period == shock_year)[0]
        if len(shock_index) == 0:
            raise ValueError("Invalid shock_year. Year not found in the period range.")
        shock_index = shock_index[0]

        if shock_type == 'capital':
            self.K[shock_index:] += shock_magnitude
        elif shock_type == 'resource':
            # Add or remove a chunk of resource stock at shock_year
            self.R_array[shock_index:] = np.maximum(self.R_array[shock_index:] + shock_magnitude, 0)
        elif shock_type == 'elite':
            self.e_0 += shock_magnitude
        elif shock_type == 'productivity':
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
        original_R_array = self.R_array.copy()

        y_i = int(self.SIR_starts[0])
        S0, I0, R0 = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]

        # Baseline scenario
        self.A = original_A
        self.e_0 = original_e0
        self.R_resource = original_R_resource
        self.K = original_K.copy()
        self.R_array = original_R_array.copy()

        self.run_model()
        baseline_results = self.results[0]

        # Shock scenario
        self.A = original_A
        self.e_0 = original_e0
        self.R_resource = original_R_resource
        self.K = original_K.copy()
        self.R_array = original_R_array.copy()

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
        baseline_R_nat = baseline_results['R_nat']
        shocked_R_nat = shocked_results['R_nat']

        fig, ax = plt.subplots(4, 1, figsize=(12, 20))

        # Plot wages
        ax[0].plot(period, baseline_w, label='Wages (Baseline)', color='b')
        ax[0].plot(period, shocked_w, label='Wages (Shock)', color='r', linestyle='--')
        ax[0].axvline(shock_year, color='k', linestyle=':', label='Shock Year')
        ax[0].set_title(f'Impact of {shock_type.capitalize()} Shock on Wages (Magnitude: {shock_magnitude})')
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Wage Level')
        ax[0].legend()
        ax[0].grid(True)

        # Elite dynamics
        ax[1].plot(period, baseline_elit, label='Elite Fraction (Baseline)', color='g')
        ax[1].plot(period, shocked_elit, label='Elite Fraction (Shock)', color='orange', linestyle='--')
        ax[1].axvline(shock_year, color='k', linestyle=':', label='Shock Year')
        ax[1].set_title(f'Impact of {shock_type.capitalize()} Shock on Elite Fraction (Magnitude: {shock_magnitude})')
        ax[1].set_xlabel('Year')
        ax[1].set_ylabel('Elite Fraction')
        ax[1].legend()
        ax[1].grid(True)

        # Radical fraction
        ax[2].plot(period, baseline_I, label='Radical Fraction (Baseline)', color='r')
        ax[2].plot(period, shocked_I, label='Radical Fraction (Shock)', color='r', linestyle='--')
        ax[2].axvline(shock_year, color='k', linestyle=':', label='Shock Year')
        ax[2].set_title(f'Impact of {shock_type.capitalize()} Shock on Radical Fraction (Magnitude: {shock_magnitude})')
        ax[2].set_xlabel('Year')
        ax[2].set_ylabel('Radical Fraction')
        ax[2].legend()
        ax[2].grid(True)

        # Resource stock
        ax[3].plot(period, baseline_R_nat, label='Resource (Baseline)', color='green')
        ax[3].plot(period, shocked_R_nat, label='Resource (Shock)', color='green', linestyle='--')
        ax[3].axvline(shock_year, color='k', linestyle=':', label='Shock Year')
        ax[3].set_title(f'Impact of {shock_type.capitalize()} Shock on Resource Stock (Magnitude: {shock_magnitude})')
        ax[3].set_xlabel('Year')
        ax[3].set_ylabel('Resource Stock')
        ax[3].legend()
        ax[3].grid(True)

        # Move suptitle upward by specifying y-coordinate and then call tight_layout
        fig.suptitle(f'Comparison of Model Outcomes with and without {shock_type.capitalize()} Shock', y=0.98, fontsize=16)
        
        # Adjust layout to make room for the suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space at top by reducing top margin

        plt.show()

        # Restore parameters
        self.A = original_A
        self.e_0 = original_e0
        self.R_resource = original_R_resource
        self.K = original_K.copy()
        self.R_array = original_R_array.copy()


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


# Example usage
model = SIRModel(initialize_SIR=False, show_SIR_variation=False, enable_SDT=True)
model.run_model()
model.plot_results()

# Example shocks
model.compare_with_shock('productivity', 1900, 0.2)
model.compare_with_shock('resource', 1950, -0.5)
model.compare_with_shock('capital', 1840, 0.8)
model.compare_with_shock('elite', 1860, 0.02)
