import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import pickle
import sys

class SIRModel:
    def __init__(self, pt_original=False, quick_adjust=False, initialize_SIR=False, show_SIR_variation=True, enable_SDT=True, verbose=True):
        """
        Initialize the SIRModel with configuration settings.

        :param pt_original: 
            - If True, uses Peter's original elite fraction dynamics.
            - If False, uses Jim's variation of elite fraction dynamics.
        :param quick_adjust:
            - If True, enables rapid adjustments in resource and wage dynamics.
            - If False, uses standard adjustment rates.
        :param initialize_SIR: Whether to initialize the SIR model with original parameters and states.
        :param show_SIR_variation: Whether to show variation in the SIR model starting state.
        :param enable_SDT: Whether to enable SDT (Structural Demographic Theory) forcing on the SIR model.
        :param verbose: If True, print additional information during initialization.
        """
        # Configuration flags
        self.pt_original = pt_original
        self.quick_adjust = quick_adjust
        self.initialize_SIR = initialize_SIR
        self.show_SIR_variation = show_SIR_variation
        self.enable_SDT = enable_SDT
        self.verbose = verbose

        # SIR model parameters
        self.T_ages = 45                # Number of age classes
        self.sigma_0 = 0.0005           # Base spontaneous radicalization rate/fraction per year
        self.a_0 = 0.1                  # Base modulation factor for radical formation
        self.a_max = 1                  # Maximum fraction of susceptibles that can radicalize
        self.gamma = 1                  # Influence of the recovered population on radicalization
        self.delta = 0.5                # Fraction of radicalized converting to moderate
        self.tau = 10                   # Time delay for infected (radicalized) people to recover
        self.period_n = 1               # Number of periods to simulate
        self.I_0 = 0.1                  # Initial fraction of infected
        self.R_0 = 0.0                  # Initial fraction of recovered
        self.S0 = np.zeros((self.T_ages, self.period_n))  # Initial susceptible state
        self.I0 = np.zeros((self.T_ages, self.period_n))  # Initial infected state
        self.R0 = np.zeros((self.T_ages, self.period_n))  # Initial recovered state
        self.SIR_starts = [0]           # Initial SIR state based on starting parameters

        # Initialize SIR states
        if self.initialize_SIR:
            self.initialize_parameters()
        else:
            self.load_initial_state()

        # Determine starting points for multiple simulations if variations are shown
        if self.show_SIR_variation:
            self.SIR_increment = 5      # Sample variation every 5 years
            self.SIR_starts = np.arange(self.T_ages, self.S0.shape[1], self.SIR_increment)
        else:
            self.SIR_increment = 0
            self.SIR_starts = [min(self.T_ages, self.S0.shape[1] - 1)]

        # Structural Demographic Theory (SDT) parameters
        self.mu_0 = 0.3 / 100          # Rate of upward mobility to elites
        self.a_w = 1.0                 # Sensitivity of radicalization to worker income changes
        self.a_e = 0.5 * 100           # Sensitivity of radicalization to elite fraction changes
        self.YB_year = 0               # Year of youth bulge (0 means no youth bulge)
        self.age_factor = 0.2          # Scaling factor for youth bulge effect
        self.w_0 = 0.90                # Baseline wage factor
        self.w_m = 0.75                # Alternate wage factor
        self.e_0 = 0.01                # Expected baseline proportion of elites
        self.show_year = 2020          # Specific year to highlight in plots

        # Exogenous drivers for wages (original approach, now replaced by endogenous wages)
        self.transitions = np.array([
            [1810, self.w_0], [1840, self.w_0], [1850, self.w_m],
            [1910, self.w_m], [1940, self.w_0], [1970, self.w_0],
            [2000, self.w_m], [2020, self.w_m], [2025, self.w_0], [2100, self.w_0]
        ])

        # Disable SDT influences if not enabled
        if not self.enable_SDT:
            self.a_w, self.a_e, self.YB_year = 0.0, 0.0, 0

        # Define the simulation period based on transitions
        self.period = np.arange(self.transitions[0, 0], self.transitions[-1, 0] + 1)

        # Youth bulge array
        self.YB_A20 = np.zeros(len(self.period))
        if self.YB_year:
            # Define youth bulge effect if applicable
            self.YB_A20 = self.age_factor * np.exp(-((self.period - self.YB_year) ** 2) / 10**2)
        else:
            self.YB_year = '(None)'  # For legend display

        # Reporting the transitions (for verbosity)
        for i in range(1, len(self.transitions)):
            eyr, syr = self.transitions[i, 0], self.transitions[i - 1, 0]
            delta_t = eyr - syr
            change = 100 * (self.transitions[i, 1] - self.transitions[i - 1, 1]) / delta_t
            if self.verbose:
                print(f'{int(syr)}-{int(eyr)}: {int(delta_t)} years {change:.1f}%')

        # Expectation wealth per elite when relative wage is w_0
        self.eps_factor = (1 - self.w_0) / self.e_0

        # Initialize results storage
        self.results = []

        # Resource dynamics parameters
        self.R_max = 1              # Carrying capacity for natural resources
        self.r_regen = 0.05            # Natural regeneration rate toward R_max
        self.delta_extract = 0.03      # Base depletion scaling factor for resource extraction
        self.mu_elite_extr = 0.5       # Additional depletion factor per unit of elite fraction
        self.alpha_w = 1.5             # Elasticity parameter for wages with respect to resources
        self.eta = 1.0                 # Depletion sensitivity to wage levels

        # Playing with Resource Sensitivity & Regeneration 

        self.r_regen = 0.01  # Slower resource replenishment (was 0.05)
        self.mu_elite_extr = 1.0  # Elites have a stronger effect on resource use (was 0.5)
        self.eta = 3.0  # Wage increases cause even more resource depletion (was 1.0)

        # Playing with Wage Elasticity & Economic Sensitivity
        self.alpha_w = 2.0  # Wages are more sensitive to resource depletion (was 1.0)
        self.a_w = 2.0  # Higher sensitivity of radicalization to wage fluctuations (was 1.0)
        self.a_e = 1.0 * 100  # Make radicalization more responsive to elite fraction (was 0.5 * 100)

        # Initialize resource stock array
        self.R_nat = np.full(len(self.period), np.nan)  # Natural resource stock over time
        self.R_nat[0] = 1.0                             # Initial resource stock

        if self.verbose:
            print("Model initialized with endogenous wage and resource dynamics.")

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

    def wage_function(self, R_t, elit_t):
        """
        Compute endogenous wage based on resource stock and elite fraction.

        :param R_t: Current resource stock
        :param elit_t: Current elite fraction
        :return: Endogenous wage level
        """
        R0 = self.R_nat[0]  # Initial resource stock
        numerator = (1 - elit_t) * R_t
        denominator = (1 - self.e_0) * R0
        return self.w_0 * (numerator / denominator)**self.alpha_w

    def depletion_function(self, w_t, elit_t):
        """
        Compute resource depletion based on wage and elite fraction.

        :param w_t: Current wage level
        :param elit_t: Current elite fraction
        :return: Depletion amount
        """
        return self.delta_extract * (1 + self.mu_elite_extr * elit_t) * (w_t**self.eta)

    def resource_update(self, R_t, D_t):
        """
        Update resource stock with logistic regeneration and depletion.

        :param R_t: Current resource stock
        :param D_t: Current depletion amount
        :return: Updated resource stock
        """
        growth = self.r_regen * R_t * (1 - R_t / self.R_max)
        if self.quick_adjust:
            # Faster regeneration and/or different depletion response
            growth *= 1.5  # Example: increase regeneration rate by 50%
            depletion_factor = 1.5  # Example: increase depletion effect by 50%
            D_t *= depletion_factor
        return R_t + growth - D_t

    def simulate(self, S0, I0, R0):
        """
        Run the SIR model simulation with endogenous wages and resource dynamics.

        :param S0: Initial susceptible state array
        :param I0: Initial infected state array
        :param R0: Initial recovered state array
        :return: Tuple containing S, I, R matrices and their sums, alpha_rec, elite fractions, and epsilon values
        """
        # Initialize SIR matrices
        S = np.zeros((self.T_ages, len(self.period)))
        I = np.zeros((self.T_ages, len(self.period)))
        R = np.zeros((self.T_ages, len(self.period)))

        # Set initial states
        S[:, 0], I[:, 0], R[:, 0] = S0, I0, R0

        # Initialize sums and alpha_rec
        S_sum = np.full(len(self.period), np.nan)
        I_sum = np.full(len(self.period), np.nan)
        R_sum = np.full(len(self.period), np.nan)
        alpha_rec = np.full(len(self.period), np.nan)
        S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:, 0]), np.sum(I[:, 0]), np.sum(R[:, 0])

        # Initialize elite fraction and epsilon arrays
        elit = np.full(len(self.period), np.nan)
        elit[0] = self.e_0  # Initial elite fraction
        epsln = np.full(len(self.period), np.nan)
        epsln[0] = self.eps_factor * (1 - self.wage_function(self.R_nat[0], elit[0])) / elit[0]

        # Temporary copy of resource stock to update during simulation
        R_nat_local = self.R_nat.copy()

        # Run simulation over each time period
        for t in range(len(self.period)-1):
            t1 = t + 1

            # Compute wage at current time
            w_t = self.wage_function(R_nat_local[t], elit[t])

            # Update elite fraction based on wage dynamics
            if self.pt_original:
                elit[t1] = elit[t] + self.mu_0 * (self.w_0 - w_t) / w_t
                elit[t1] -= (elit[t1] - self.e_0) * R_sum[t]
            else:
                elit[t1] = elit[t] + self.mu_0 * (self.w_0 - w_t) / w_t - (elit[t] - self.e_0) * R_sum[t]

            # Ensure elite fraction remains between 0 and 1
            elit[t1] = np.clip(elit[t1], 0.0, 1.0)

            # Recompute wage after elite update for epsilon calculation
            w_t1 = self.wage_function(R_nat_local[t], elit[t1])
            epsln[t1] = self.eps_factor * (1 - w_t1) / elit[t1]

            # Calculate radicalization rate alpha
            alpha = np.clip(
                self.a_0 + self.a_w * (self.w_0 - w_t1) + self.a_e * (elit[t1] - self.e_0) + self.YB_A20[t1],
                self.a_0, self.a_max
            )
            alpha_rec[t1] = alpha

            # Calculate sigma and rho for SIR transitions
            sigma = np.clip((alpha - self.gamma * np.sum(R[:, t])) * np.sum(I[:, t]) + self.sigma_0, 0, 1)
            # Corrected rho calculation using I_sum
            rho = np.clip(self.delta * (I_sum[t - self.tau] if t > self.tau else 0), 0, 1)

            # Update S, I, R with age-structured transitions
            S[0, t1] = 1 / self.T_ages
            for age in range(self.T_ages - 1):
                S[age + 1, t1] = (1 - sigma) * S[age, t]
                I[age + 1, t1] = (1 - rho) * I[age, t] + sigma * S[age, t]
                R[age + 1, t1] = R[age, t] + rho * I[age, t]

            # Update sums
            S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(S[:, t1]), np.sum(I[:, t1]), np.sum(R[:, t1])

            # Additional elite adjustment if using original parameters
            if self.pt_original:
                elit[t1] -= (elit[t1] - self.e_0) * R_sum[t1]

            # Compute depletion based on current wage and elite fraction
            D_t = self.depletion_function(w_t, elit[t])

            # Update resource stock for next period
            R_nat_local[t1] = self.resource_update(R_nat_local[t], D_t)

        return S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln, R_nat_local

    def run_model(self):
        """
        Run the SIR model simulation.

        This method iterates over all starting points (if variations are enabled),
        runs the simulation, and stores the results.
        """
        self.results = []
        for y_i in self.SIR_starts:
            y_i = int(y_i)  # Ensure index is integer
            S0, I0, R0 = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]
            # Compute simulation with current SIR state
            S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln, R_nat = self.simulate(S0, I0, R0)
            # Compute wage array based on resource stock and elite fraction
            w_array = np.array([self.wage_function(R_nat[t], elit[t]) for t in range(len(self.period))])
            # Store simulation results
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
                'w': w_array,
                'R_nat': R_nat,
                'YB_A20': self.YB_A20
            })

    def plot_results(self):
        """
        Plot the results of the SIR model simulation with a polished style.
        
        Generates two subplots:
        - Top: Wages, Youth Bulge, Elite Fraction, Radicalization Rate, Radical Fraction (I), Moderate Fraction (R)
        - Bottom: Natural Resource Stock over time, 
        with y-axis limited to the max resource value encountered.
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        
        # Define some custom colors
        brown = '#A52A2A'       # For wages
        nice_green = '#00BFFF'  # For radicalization rate (alpha)

        # We'll collect plot handles for a consolidated legend
        handles_collector = []

        # First, find the maximum resource value across all results for dynamic scaling
        max_resource = 0
        for result in self.results:
            max_resource = max(max_resource, np.max(result['R_nat']))
        
        # Plot each result set if multiple runs exist
        for idx, result in enumerate(self.results):
            S_sum = result['S_sum']
            I_sum = result['I_sum']
            R_sum = result['R_sum']
            alpha_rec = result['alpha_rec']
            elit = result['elit']
            w_array = result['w']
            R_nat = result['R_nat']
            YB_A20 = result['YB_A20']

            # ===== TOP SUBPLOT: Social and Economic Variables =====
            # Plot wage
            if self.enable_SDT:
                l1, = ax[0].plot(self.period, w_array, color=brown, linestyle='-', linewidth=2,
                                label='(w) Endogenous wage' if idx == 0 else "_nolegend_")

                # Youth bulge
                l2, = ax[0].plot(self.period, YB_A20, color='k', linestyle='-.', linewidth=2,
                                label=f'Youth bulge {self.YB_year}' if idx == 0 else "_nolegend_")

                # Elite fraction (scaled by 100 to make it percentage-like)
                l3, = ax[0].plot(self.period, elit * 100, color='b', linestyle='-',
                                label='(e*100) Elite fraction' if idx == 0 else "_nolegend_")

                # Radicalization rate alpha
                l4, = ax[0].plot(self.period, alpha_rec, color=nice_green, linestyle='--', linewidth=2,
                                label=r'($\alpha$) radicalization rate' if idx == 0 else "_nolegend_")

                # Keep references for the legend if first run
                if idx == 0:
                    handles_collector += [l1, l2, l3, l4]

            # Radical (I) fraction
            l5, = ax[0].plot(self.period, I_sum, color='r', linestyle='-',
                            label='(I) Radical fraction' if idx == 0 else "_nolegend_")
            # Moderate (R) fraction
            l6, = ax[0].plot(self.period, R_sum, color='k', linestyle='-',
                            label='(R) Moderate fraction' if idx == 0 else "_nolegend_")

            if idx == 0:
                handles_collector += [l5, l6]

            # Optionally add a vertical line for a highlight year
            if (self.show_year in self.period) and (idx == 0):
                ax[0].axvline(x=self.show_year, color='k', linestyle='--')

            # ===== BOTTOM SUBPLOT: Resource Dynamics =====
            l7, = ax[1].plot(self.period, R_nat, color='green', linestyle='-', linewidth=2,
                            label='Resource Stock' if idx == 0 else "_nolegend_")

            if idx == 0:
                handles_collector.append(l7)

        # Add horizontal line for R_max in bottom subplot
        ax[1].axhline(self.R_max, color='gray', linestyle='--', label='R_max')

        # ===== Final configuration of subplots =====
        # Top subplot settings
        ax[0].set_xlabel('Year', fontsize=12)
        ax[0].set_ylabel('Fraction / Rate / Level', fontsize=12)
        ax[0].set_title('Social and Economic Variables', fontsize=14)
        ax[0].set_xlim([self.period[0], self.period[-1]])
        ax[0].grid(True)

        # Bottom subplot settings
        ax[1].set_title('Natural Resource Dynamics', fontsize=14)
        ax[1].set_xlabel('Year', fontsize=12)
        ax[1].set_ylabel('Resource Stock', fontsize=12)
        ax[1].grid(True)

        # Dynamically limit the y-axis to show detail of resource fluctuations
        # Add a small buffer on top so lines are not flush with the upper boundary
        # Consolidated legend for all lines
        fig.legend(handles=handles_collector,
                loc='center left',
                bbox_to_anchor=(0.95, 0.5),
                fontsize=11)

        # Adjust the layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.93, 1])
        plt.show()

    def introduce_shock(self, shock_type, shock_year, shock_magnitude):
        """
        Introduce a shock to the model by altering resource stock or elite fraction.

        :param shock_type: Type of shock ('resource' or 'elite')
        :param shock_year: Year when the shock occurs
        :param shock_magnitude: Magnitude of the shock (positive or negative)
        """
        shock_index = np.where(self.period == shock_year)[0]
        if len(shock_index) == 0:
            raise ValueError("Invalid shock_year. Year not found in the period range.")
        shock_index = shock_index[0]

        if shock_type == 'resource':
            # Apply shock to resource stock from shock_year onward
            self.R_nat[shock_index:] += shock_magnitude
            # Ensure resource stock does not drop below zero
            self.R_nat = np.maximum(self.R_nat, 0.0)
        elif shock_type == 'elite':
            # Adjust the baseline elite fraction
            self.e_0 += shock_magnitude
            # Ensure elite fraction remains between 0 and 1
            self.e_0 = np.clip(self.e_0, 0.0, 1.0)
        else:
            raise ValueError("Invalid shock_type. Must be 'resource' or 'elite'.")

    def compare_with_shock(self, shock_type, shock_year, shock_magnitude):
        """
        Compare the model trajectories with and without a specified shock.

        :param shock_type: Type of shock ('resource' or 'elite')
        :param shock_year: Year when the shock occurs
        :param shock_magnitude: Magnitude of the shock
        """
        # Store original parameters to restore after comparison
        original_e0 = self.e_0
        original_R_nat = self.R_nat.copy()

        # Run baseline simulation without shock
        self.e_0 = original_e0
        self.R_nat = original_R_nat.copy()
        self.run_model()
        baseline_result = self.results[-1]

        # Apply shock
        self.introduce_shock(shock_type, shock_year, shock_magnitude)

        # Run shocked simulation
        self.run_model()
        shocked_result = self.results[-1]

        # Extract data for plotting
        S_sum_base = baseline_result['S_sum']
        I_sum_base = baseline_result['I_sum']
        R_sum_base = baseline_result['R_sum']
        elit_base = baseline_result['elit']
        w_base = baseline_result['w']
        R_nat_base = baseline_result['R_nat']
        alpha_rec_base = baseline_result['alpha_rec']

        S_sum_shock = shocked_result['S_sum']
        I_sum_shock = shocked_result['I_sum']
        R_sum_shock = shocked_result['R_sum']
        elit_shock = shocked_result['elit']
        w_shock = shocked_result['w']
        R_nat_shock = shocked_result['R_nat']
        alpha_rec_shock = shocked_result['alpha_rec']

        # Create plots for comparison
        fig, ax = plt.subplots(2, 1, figsize=(14, 16))
        brown, nice_green = '#A52A2A', '#00BFFF'

        # Top subplot: Social and Economic Variables
        if self.enable_SDT:
            ax[0].plot(self.period, w_base, color=brown, linestyle='-', linewidth=2, label='(w) Endogenous wage (Baseline)')
            ax[0].plot(self.period, w_shock, color=brown, linestyle='--', linewidth=2, label='(w) Endogenous wage (Shock)')

            ax[0].plot(self.period, self.YB_A20, color='k', linestyle='-.', linewidth=2, label=f'Youth bulge {self.YB_year} (Baseline)')
            ax[0].plot(self.period, self.YB_A20, color='k', linestyle=':', linewidth=2, label=f'Youth bulge {self.YB_year} (Shock)')

            ax[0].plot(self.period, elit_base * 100, color='b', linestyle='-', label='(e*100) Elite fraction (Baseline)')
            ax[0].plot(self.period, elit_shock * 100, color='b', linestyle='--', label='(e*100) Elite fraction (Shock)')

            ax[0].plot(self.period, alpha_rec_base, color=nice_green, linestyle='-', linewidth=2, label=r'($\alpha$) Radicalization rate (Baseline)')
            ax[0].plot(self.period, alpha_rec_shock, color=nice_green, linestyle='--', linewidth=2, label=r'($\alpha$) Radicalization rate (Shock)')

            if self.show_year in self.period:
                ax[0].axvline(x=self.show_year, color='k', linestyle='--')

        # Plot Radical and Moderate fractions
        ax[0].plot(self.period, I_sum_base, color='r', linestyle='-', label='(I) Radical fraction (Baseline)')
        ax[0].plot(self.period, I_sum_shock, color='r', linestyle='--', label='(I) Radical fraction (Shock)')
        ax[0].plot(self.period, R_sum_base, color='k', linestyle='-', label='(R) Moderate fraction (Baseline)')
        ax[0].plot(self.period, R_sum_shock, color='k', linestyle='--', label='(R) Moderate fraction (Shock)')

        # Configure top subplot
        handles = []
        if self.enable_SDT:
            handles += [
                plt.Line2D([0], [0], color=brown, lw=2, label='(w) Endogenous wage (Baseline)'),
                plt.Line2D([0], [0], color=brown, lw=2, linestyle='--', label='(w) Endogenous wage (Shock)'),
                plt.Line2D([0], [0], color='k', lw=2, linestyle='-.', label=f'Youth bulge {self.YB_year} (Baseline)'),
                plt.Line2D([0], [0], color='k', lw=2, linestyle=':', label=f'Youth bulge {self.YB_year} (Shock)'),
                plt.Line2D([0], [0], color='b', lw=2, label='(e*100) Elite fraction (Baseline)'),
                plt.Line2D([0], [0], color='b', lw=2, linestyle='--', label='(e*100) Elite fraction (Shock)'),
                plt.Line2D([0], [0], color=nice_green, lw=2, linestyle='-', label=r'($\alpha$) Radicalization rate (Baseline)'),
                plt.Line2D([0], [0], color=nice_green, lw=2, linestyle='--', label=r'($\alpha$) Radicalization rate (Shock)')
            ]
        handles += [
            plt.Line2D([0], [0], color='r', lw=2, label='(I) Radical fraction (Baseline)'),
            plt.Line2D([0], [0], color='r', lw=2, linestyle='--', label='(I) Radical fraction (Shock)'),
            plt.Line2D([0], [0], color='k', lw=2, label='(R) Moderate fraction (Baseline)'),
            plt.Line2D([0], [0], color='k', lw=2, linestyle='--', label='(R) Moderate fraction (Shock)')
        ]

        ax[0].legend(handles=handles, loc='upper left', fontsize=12)
        ax[0].set_xlabel('Year', fontsize=14)
        ax[0].set_ylabel('Fraction / Percentage', fontsize=14)
        ax[0].grid(True)
        ax[0].set_xlim([self.period[0], self.period[-1]])
        ax[0].set_title('Social and Economic Variables', fontsize=16)

        # Bottom subplot: Natural Resource Dynamics
        for result in self.results:
            R_nat = result['R_nat']
            ax[1].plot(self.period, R_nat, color='green', linestyle='-', linewidth=2, label='Resource Stock')

        ax[1].axhline(self.R_max, color='gray', linestyle='--', label='R_max')
        ax[1].set_title('Natural Resource Dynamics', fontsize=16)
        ax[1].set_xlabel('Year', fontsize=14)
        ax[1].set_ylabel('Resource Stock', fontsize=14)
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    def introduce_shock(self, shock_type, shock_year, shock_magnitude):
        """
        Introduce a shock to the model by altering resource stock or elite fraction.

        :param shock_type: Type of shock ('resource' or 'elite')
        :param shock_year: Year when the shock occurs
        :param shock_magnitude: Magnitude of the shock (positive or negative)
        """
        shock_index = np.where(self.period == shock_year)[0]
        if len(shock_index) == 0:
            raise ValueError("Invalid shock_year. Year not found in the period range.")
        shock_index = shock_index[0]

        if shock_type == 'resource':
            # Apply shock to resource stock from shock_year onward
            self.R_nat[shock_index:] += shock_magnitude
            # Ensure resource stock does not drop below zero
            self.R_nat = np.maximum(self.R_nat, 0.0)
        elif shock_type == 'elite':
            # Adjust the baseline elite fraction
            self.e_0 += shock_magnitude
            # Ensure elite fraction remains between 0 and 1
            self.e_0 = np.clip(self.e_0, 0.0, 1.0)        
        elif shock_type == 'reduce_regen':
            new_r_regen = self.r_regen - shock_magnitude
            # Ensure it doesn't go below zero
            self.r_regen = max(new_r_regen, 0.0)
        else:
            raise ValueError("Invalid shock_type. Must be 'resource' or 'elite'.")

    def compare_with_shock(self, shock_type, shock_year, shock_magnitude):
        """
        Compare the model trajectories with and without a specified shock.

        :param shock_type: Type of shock ('resource' or 'elite')
        :param shock_year: Year when the shock occurs
        :param shock_magnitude: Magnitude of the shock
        """
        # Store original parameters to restore after comparison
        original_e0 = self.e_0
        original_R_nat = self.R_nat.copy()

        # Run baseline simulation without shock
        self.e_0 = original_e0
        self.R_nat = original_R_nat.copy()
        self.run_model()
        baseline_result = self.results[-1]

        # Apply shock
        self.introduce_shock(shock_type, shock_year, shock_magnitude)

        # Run shocked simulation
        self.run_model()
        shocked_result = self.results[-1]

        # Extract data for plotting
        S_sum_base = baseline_result['S_sum']
        I_sum_base = baseline_result['I_sum']
        R_sum_base = baseline_result['R_sum']
        elit_base = baseline_result['elit']
        w_base = baseline_result['w']
        R_nat_base = baseline_result['R_nat']
        alpha_rec_base = baseline_result['alpha_rec']

        S_sum_shock = shocked_result['S_sum']
        I_sum_shock = shocked_result['I_sum']
        R_sum_shock = shocked_result['R_sum']
        elit_shock = shocked_result['elit']
        w_shock = shocked_result['w']
        R_nat_shock = shocked_result['R_nat']
        alpha_rec_shock = shocked_result['alpha_rec']

        # Create plots for comparison
        fig, ax = plt.subplots(2, 1, figsize=(14, 16))
        brown, nice_green = '#A52A2A', '#00BFFF'

        # Top subplot: Social and Economic Variables
        if self.enable_SDT:
            ax[0].plot(self.period, w_base, color=brown, linestyle='-', linewidth=2, label='(w) Endogenous wage (Baseline)')
            ax[0].plot(self.period, w_shock, color=brown, linestyle='--', linewidth=2, label='(w) Endogenous wage (Shock)')

            ax[0].plot(self.period, self.YB_A20, color='k', linestyle='-.', linewidth=2, label=f'Youth bulge {self.YB_year} (Baseline)')
            ax[0].plot(self.period, self.YB_A20, color='k', linestyle=':', linewidth=2, label=f'Youth bulge {self.YB_year} (Shock)')

            ax[0].plot(self.period, elit_base * 100, color='b', linestyle='-', label='(e*100) Elite fraction (Baseline)')
            ax[0].plot(self.period, elit_shock * 100, color='b', linestyle='--', label='(e*100) Elite fraction (Shock)')

            ax[0].plot(self.period, alpha_rec_base, color=nice_green, linestyle='-', linewidth=2, label=r'($\alpha$) Radicalization rate (Baseline)')
            ax[0].plot(self.period, alpha_rec_shock, color=nice_green, linestyle='--', linewidth=2, label=r'($\alpha$) Radicalization rate (Shock)')

            if self.show_year in self.period:
                ax[0].axvline(x=self.show_year, color='k', linestyle='--')

        # Plot Radical and Moderate fractions
        ax[0].plot(self.period, I_sum_base, color='r', linestyle='-', label='(I) Radical fraction (Baseline)')
        ax[0].plot(self.period, I_sum_shock, color='r', linestyle='--', label='(I) Radical fraction (Shock)')
        ax[0].plot(self.period, R_sum_base, color='k', linestyle='-', label='(R) Moderate fraction (Baseline)')
        ax[0].plot(self.period, R_sum_shock, color='k', linestyle='--', label='(R) Moderate fraction (Shock)')

        # Configure top subplot
        handles = []
        if self.enable_SDT:
            handles += [
                plt.Line2D([0], [0], color=brown, lw=2, label='(w) Endogenous wage (Baseline)'),
                plt.Line2D([0], [0], color=brown, lw=2, linestyle='--', label='(w) Endogenous wage (Shock)'),
                plt.Line2D([0], [0], color='k', lw=2, linestyle='-.', label=f'Youth bulge {self.YB_year} (Baseline)'),
                plt.Line2D([0], [0], color='k', lw=2, linestyle=':', label=f'Youth bulge {self.YB_year} (Shock)'),
                plt.Line2D([0], [0], color='b', lw=2, label='(e*100) Elite fraction (Baseline)'),
                plt.Line2D([0], [0], color='b', lw=2, linestyle='--', label='(e*100) Elite fraction (Shock)'),
                plt.Line2D([0], [0], color=nice_green, lw=2, linestyle='-', label=r'($\alpha$) Radicalization rate (Baseline)'),
                plt.Line2D([0], [0], color=nice_green, lw=2, linestyle='--', label=r'($\alpha$) Radicalization rate (Shock)')
            ]
        handles += [
            plt.Line2D([0], [0], color='r', lw=2, label='(I) Radical fraction (Baseline)'),
            plt.Line2D([0], [0], color='r', lw=2, linestyle='--', label='(I) Radical fraction (Shock)'),
            plt.Line2D([0], [0], color='k', lw=2, label='(R) Moderate fraction (Baseline)'),
            plt.Line2D([0], [0], color='k', lw=2, linestyle='--', label='(R) Moderate fraction (Shock)')
        ]

        ax[0].legend(handles=handles, loc='upper left', fontsize=12)
        ax[0].set_xlabel('Year', fontsize=14)
        ax[0].set_ylabel('Fraction / Percentage', fontsize=14)
        ax[0].grid(True)
        ax[0].set_xlim([self.period[0], self.period[-1]])
        ax[0].set_title(f'Comparison of Variables with and without {shock_type.capitalize()} Shock (Magnitude: {shock_magnitude})', fontsize=16)

        # Bottom subplot: Natural Resource Dynamics
        ax[1].plot(self.period, R_nat_base, color='green', linestyle='-', linewidth=2, label='Resource Stock (Baseline)')
        ax[1].plot(self.period, R_nat_shock, color='green', linestyle='--', linewidth=2, label='Resource Stock (Shock)')
        ax[1].axhline(self.R_max, color='gray', linestyle='--', label='R_max')
        ax[1].set_title('Natural Resource Dynamics Comparison', fontsize=16)
        ax[1].set_xlabel('Year', fontsize=14)
        ax[1].set_ylabel('Resource Stock', fontsize=14)
        ax[1].legend()
        ax[1].grid(True)
        ax[1].autoscale_view()

        plt.tight_layout()
        plt.show()

        # Restore original parameters after comparison
        self.e_0 = original_e0
        self.R_nat = original_R_nat.copy()

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


# Example 

model = SIRModel(
    pt_original=False,          # Use Jim's variation for elite fraction dynamics
    quick_adjust=True,          # Enable rapid adjustments in resource and wage dynamics
    initialize_SIR=False,       # Load initial state from file
    show_SIR_variation=False,   # Do not show variations in starting states
    enable_SDT=True,            # Enable Structural Demographic Theory factors
    verbose=True                # Enable verbose output
)

model.run_model()
model.plot_results()

model.compare_with_shock('resource', 1950, -0.3)
model.compare_with_shock('elite', 1990, 0.05)
model.compare_with_shock('reduce_regen', 1990, 0.05)
