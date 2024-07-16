import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import pickle
import sys

class SIRModel:
    def __init__(self, pt_original=False, initialize_SIR=False, show_SIR_variation=True, enable_SDT=True):
        """
        Initialize the SIRModel with configuration settings.

        :param pt_original: Whether to use Peter's original parameters and initialization (True) or Jim's variation (False)
        :param initialize_SIR: Whether to initialize the SIR model with original parameters and states
        :param show_SIR_variation: Whether to show variation in the SIR model starting state
        :param enable_SDT: Whether to enable SDT (Social Dominance Theory) forcing on the SIR model
        """
        self.pt_original = pt_original
        self.initialize_SIR = initialize_SIR
        self.show_SIR_variation = show_SIR_variation
        self.enable_SDT = enable_SDT

        # Model parameters
        self.T_ages = 45  # Number of age classes
        self.sigma_0 = 0.0005  # Base spontaneous radicalization rate/fraction per year
        self.a_0 = 0.1  # Modulate down how many radicals form in the presence of radicals
        self.a_max = 1  # Maximum fraction of susceptibles that can radicalize
        self.gamma = 1  # Fraction of all recovered that influences conversion to radicals
        self.delta = 0.5  # Fraction of radicalized converting to moderate
        self.tau = 10  # Time delay for infected (radicalized) people to recover
        self.period_n = 1  # Number of periods to simulate
        self.I_0 = 0.1  # Initial fraction of infected
        self.R_0 = 0.0  # Initial fraction of recovered
        self.S0 = np.zeros((self.T_ages, self.period_n))  # Initial susceptible state
        self.I0 = np.zeros((self.T_ages, self.period_n))  # Initial infected state
        self.R0 = np.zeros((self.T_ages, self.period_n))  # Initial recovered state
        self.SIR_starts = [0]  # Initial SIR state based on starting parameters

        if self.initialize_SIR:
            self.initialize_parameters()
        else:
            self.load_initial_state()

        if self.show_SIR_variation:
            self.SIR_increment = 5  # Sample variation in response in the SIR model starting distribution every 5 years
            self.SIR_starts = np.arange(self.T_ages, self.S0.shape[1], self.SIR_increment)
        else:
            self.SIR_increment = 0  # No variation
            self.SIR_starts = [min(self.T_ages, self.S0.shape[1] - 1)]  # Avoid strange initialization in startup sequence

        # SDT parameters
        self.mu_0 = 0.3 / 100  # Rate of upward mobility to elites
        self.a_w = 1.0  # How radicalization changes as relative worker income changes from expected
        self.a_e = 0.5 * 100  # How radicalization changes as relative elite numbers change from expected
        self.YB_year = 0  # Assume no youth bulge initially
        self.age_factor = 0.2
        self.w_0 = 0.90
        self.w_m = 0.75
        self.e_0 = 0.01  # Expected proportion of elite
        self.show_year = 2020

        # Exogenous drivers
        self.transitions = np.array([
            [1810, self.w_0], [1840, self.w_0], [1850, self.w_m],
            [1910, self.w_m], [1940, self.w_0], [1970, self.w_0], [2000, self.w_m],
            [2020, self.w_m], [2025, self.w_0], [2100, self.w_0]
        ])

        if not self.enable_SDT:
            # Eliminate impact of w and YB on elite and popular radicalization
            self.a_w, self.a_e, self.YB_year = 0.0, 0.0, 0

        # Interpolating relative income
        self.period = np.arange(self.transitions[0, 0], self.transitions[-1, 0] + 1)
        self.interp_fn = PchipInterpolator(self.transitions[:, 0], self.transitions[:, 1])
        self.w = self.interp_fn(self.period)
        self.YB_A20 = np.zeros(len(self.period))
        if self.YB_year:
            self.YB_A20 = self.age_factor * np.exp(-((self.period - self.YB_year) ** 2) / 10**2)
        else:
            self.YB_year = '(None)'  # For legend display

        # Reporting the transitions
        for i in range(1, len(self.transitions)):
            eyr, syr = self.transitions[i, 0], self.transitions[i - 1, 0]
            delta_t = eyr - syr
            change = 100 * (self.transitions[i, 1] - self.transitions[i - 1, 1]) / delta_t
            print(f'{int(syr)}-{int(eyr)}: {int(delta_t)} years {change:.1f}%')

        # Expectation wealth per elite when relative wage is w_0
        self.eps_factor = (1 - self.w_0) / self.e_0

        # Track the relative proportion of population that are elite
        self.elit = np.full(len(self.period), np.nan)
        self.elit[0] = self.e_0  # Assume initial proportion of elite
        self.epsln = np.full(len(self.period), np.nan)
        self.epsln[0] = self.eps_factor * (1 - self.w[0]) / self.elit[0]

        # Create the SIR matrices to track proportions per age class over all the time periods
        self.S, self.I, self.R = np.zeros((self.T_ages, len(self.period))), np.zeros((self.T_ages, len(self.period))), np.zeros((self.T_ages, len(self.period)))

    def initialize_parameters(self):
        """Initialize SIR constants and parameters."""
        self.S0[0, :] = (1 - self.I_0 - self.R_0) / self.T_ages
        self.I0[0, :] = self.I_0 / self.T_ages
        self.R0[0, :] = self.R_0 / self.T_ages
        self.SIR_starts = [0]

    def load_initial_state(self):
        """Load previous SIR state with its creation parameters."""
        try:
            with open('initial_SIR.pkl', 'rb') as fh:
                if sys.version_info[:3] >= (3, 0):
                    self.T_ages, self.a_0, self.a_max, self.gamma, self.sigma_0, self.delta, self.tau, self.S0, self.I0, self.R0 = pickle.load(fh, encoding='latin1')
                else:
                    self.T_ages, self.a_0, self.a_max, self.gamma, self.sigma_0, self.delta, self.tau, self.S0, self.I0, self.R0 = pickle.load(fh)
        except:
            raise RuntimeError("Unable to load initial_SIR.pkl!")

    def run_model(self):
        """Run the SIR model simulation."""
        for y_i in self.SIR_starts:
            y_i = int(y_i)  # Ensure index
            self.S[:, 0], self.I[:, 0], self.R[:, 0] = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]

            S_sum, I_sum, R_sum, alpha_rec = [np.full(len(self.period), np.nan) for _ in range(4)]
            S_sum[0], I_sum[0], R_sum[0] = np.sum(self.S[:, 0]), np.sum(self.I[:, 0]), np.sum(self.R[:, 0])

            for t in range(len(self.period) - 1):
                t1 = t + 1
                if self.pt_original:
                    self.elit[t] + self.mu_0 * (self.w_0 - self.w[t]) / self.w[t]  # Update relative elite numbers
                else:
                    self.elit[t1] = self.elit[t] + self.mu_0 * (self.w_0 - self.w[t]) / self.w[t] - (self.elit[t] - self.e_0) * R_sum[t]  # Update relative elite numbers

                self.epsln[t1] = self.eps_factor * (1 - self.w[t1]) / self.elit[t1]
                alpha = np.clip(self.a_0 + self.a_w * (self.w_0 - self.w[t1]) + self.a_e * (self.elit[t1] - self.e_0) + self.YB_A20[t1], self.a_0, self.a_max)
                sigma = np.clip((alpha - self.gamma * np.sum(self.R[:, t])) * np.sum(self.I[:, t]) + self.sigma_0, 0, 1)
                rho = np.clip(self.delta * np.sum(self.I[:, t - self.tau]) if t > self.tau else 0, 0, 1)
                alpha_rec[t1] = alpha  # Record alpha

                self.S[0, t1] = 1 / self.T_ages
                for age in range(self.T_ages - 1):
                    age1 = age + 1
                    self.S[age1, t1] = (1 - sigma) * self.S[age, t]
                    self.I[age1, t1] = (1 - rho) * self.I[age, t] + sigma * self.S[age, t]
                    self.R[age1, t1] = self.R[age, t] + rho * self.I[age, t]

                S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(self.S[:, t1]), np.sum(self.I[:, t1]), np.sum(self.R[:, t1])
                if self.pt_original:
                    self.elit[t1] -= (self.elit[t1] - self.e_0) * R_sum[t1]

    def plot_results(self):
        """Plot the results of the SIR model simulation."""
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        brown, nice_green = '#A52A2A', '#00BFFF'

        for y_i in self.SIR_starts:
            y_i = int(y_i)  # Ensure index
            self.S[:, 0], self.I[:, 0], self.R[:, 0] = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]

            S_sum, I_sum, R_sum, alpha_rec = [np.full(len(self.period), np.nan) for _ in range(4)]
            S_sum[0], I_sum[0], R_sum[0] = np.sum(self.S[:, 0]), np.sum(self.I[:, 0]), np.sum(self.R[:, 0])

            for t in range(len(self.period) - 1):
                t1 = t + 1
                if self.pt_original:
                    self.elit[t] + self.mu_0 * (self.w_0 - self.w[t]) / self.w[t]
                else:
                    self.elit[t1] = self.elit[t] + self.mu_0 * (self.w_0 - self.w[t]) / self.w[t] - (self.elit[t] - self.e_0) * R_sum[t]

                self.epsln[t1] = self.eps_factor * (1 - self.w[t1]) / self.elit[t1]
                alpha = np.clip(self.a_0 + self.a_w * (self.w_0 - self.w[t1]) + self.a_e * (self.elit[t1] - self.e_0) + self.YB_A20[t1], self.a_0, self.a_max)
                sigma = np.clip((alpha - self.gamma * np.sum(self.R[:, t])) * np.sum(self.I[:, t]) + self.sigma_0, 0, 1)
                rho = np.clip(self.delta * np.sum(self.I[:, t - self.tau]) if t > self.tau else 0, 0, 1)
                alpha_rec[t1] = alpha

                self.S[0, t1] = 1 / self.T_ages
                for age in range(self.T_ages - 1):
                    age1 = age + 1
                    self.S[age1, t1] = (1 - sigma) * self.S[age, t]
                    self.I[age1, t1] = (1 - rho) * self.I[age, t] + sigma * self.S[age, t]
                    self.R[age1, t1] = self.R[age, t] + rho * self.I[age, t]

                S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(self.S[:, t1]), np.sum(self.I[:, t1]), np.sum(self.R[:, t1])
                if self.pt_original:
                    self.elit[t1] -= (self.elit[t1] - self.e_0) * R_sum[t1]

            if self.enable_SDT:
                w_ax1, = ax1.plot(self.period, self.w, color=brown, linestyle='-', linewidth=2, label='(w) Relative income')
                YB_ax1, = ax1.plot(self.period, self.YB_A20, color='k', linestyle='-.', linewidth=2, label=f'Youth bulge {self.YB_year}')
                elit_ax1, = ax1.plot(self.period, self.elit * 100, color='b', linestyle='-', label=f'(e*100) Elite fraction of population')
                alpha_ax1, = ax1.plot(self.period, alpha_rec, color=nice_green, linestyle='--', linewidth=2, label=r'($\alpha$) radicalization rate')

                if self.show_year in self.period:
                    ax1.axvline(x=self.show_year, color='k', linestyle='--')

            I_ax1, = ax1.plot(self.period, I_sum, color='r', linestyle='-', label='(I) Radical fraction')
            R_ax1, = ax1.plot(self.period, R_sum, color='k', linestyle='-', label='(R) Moderate fraction')

        handles = [w_ax1, YB_ax1, elit_ax1, alpha_ax1, I_ax1, R_ax1] if self.enable_SDT else [I_ax1, R_ax1]
        ax1.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        ax1.set_xlabel('Year', fontsize=12)
        ax1.grid(True)
        ax1.set_ylim([-0.1, 2.5])
        ax1.set_xlim([self.period[0], self.period[-1]])
        ax1.set_title(f'US: Mockup historical w; fast recovery 2025 {self.show_SIR_variation} incr: {self.SIR_increment}')
        plt.show()


    def introduce_shock(self, shock_type, shock_year, shock_magnitude):
        """
        Introduce a shock to either elite numbers or wages.

        :param shock_type: Type of shock ('elite' or 'wage')
        :param shock_year: Year when the shock occurs
        :param shock_magnitude: Magnitude of the shock
        """
        if shock_type not in ['elite', 'wage']:
            raise ValueError("Invalid shock_type. Must be 'elite' or 'wage'.")

        shock_index = np.where(self.period == shock_year)[0]
        if len(shock_index) == 0:
            raise ValueError("Invalid shock_year. Year not found in the period range.")
        shock_index = shock_index[0]

        if shock_type == 'elite':
            self.elit[shock_index:] += shock_magnitude
        elif shock_type == 'wage':
            self.w[shock_index:] += shock_magnitude

    def compare_with_shock(self, shock_type, shock_year, shock_magnitude):
        """
        Compare the trajectories of variables with and without the shock.

        :param shock_type: Type of shock ('elite' or 'wage')
        :param shock_year: Year when the shock occurs
        :param shock_magnitude: Magnitude of the shock
        """
        # Run the model without the shock
        self.run_model()
        original_S = self.S.copy()
        original_I = self.I.copy()
        original_R = self.R.copy()
        original_elit = self.elit.copy()
        original_w = self.w.copy()

        # Introduce the shock and run the model again
        self.introduce_shock(shock_type, shock_year, shock_magnitude)
        self.run_model()

        # Plot the results for comparison
        fig, ax = plt.subplots(2, 1, figsize=(12, 16))
        brown, nice_green = '#A52A2A', '#00BFFF'

        ax[0].plot(self.period, original_I.sum(axis=0), color='r', linestyle='-', label='Radical fraction (no shock)')
        ax[0].plot(self.period, self.I.sum(axis=0), color='r', linestyle='--', label='Radical fraction (with shock)')
        ax[0].plot(self.period, original_R.sum(axis=0), color='k', linestyle='-', label='Moderate fraction (no shock)')
        ax[0].plot(self.period, self.R.sum(axis=0), color='k', linestyle='--', label='Moderate fraction (with shock)')
        ax[0].legend()
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Fraction')
        ax[0].grid(True)

        if self.enable_SDT:
            ax[1].plot(self.period, original_w, color=brown, linestyle='-', linewidth=2, label='Relative income (no shock)')
            ax[1].plot(self.period, self.w, color=brown, linestyle='--', linewidth=2, label='Relative income (with shock)')
            ax[1].plot(self.period, original_elit * 100, color='b', linestyle='-', label='Elite fraction (no shock)')
            ax[1].plot(self.period, self.elit * 100, color='b', linestyle='--', label='Elite fraction (with shock)')
            ax[1].legend()
            ax[1].set_xlabel('Year')
            ax[1].set_ylabel('Values')
            ax[1].grid(True)

        fig.suptitle(f'Comparison of variables with and without shock ({shock_type} shock in {shock_year} of magnitude {shock_magnitude})')
        plt.show()

    @staticmethod
    def save_pkl_file(filename, data_tuple):
        """Save data to a pickle file."""
        with open(filename, "wb") as fh:
            pickle.dump(data_tuple, fh)
        return True

    @staticmethod
    def load_pkl_file(filename):
        """Load data from a pickle file."""
        try:
            with open(filename, "rb") as fh:
                if sys.version_info[:3] >= (3, 0):
                    data_tuple = pickle.load(fh, encoding='latin1')
                else:
                    data_tuple = pickle.load(fh)
        except:
            raise RuntimeError(f"Unable to load {filename}!")
        return data_tuple


### Example Run
model = SIRModel(initialize_SIR=False, show_SIR_variation=False, enable_SDT=True)
model.run_model()
model.plot_results()

### Example Shock
shock_type = 'wage'  # or 'elite'
shock_year = 1960
shock_magnitude = -0.1
model.compare_with_shock(shock_type, shock_year, shock_magnitude)