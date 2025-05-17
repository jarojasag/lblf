import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

class SIRModel:
    def __init__(self, 
                 pt_original=False, 
                 quick_adjust=False, 
                 initialize_SIR=False, 
                 show_SIR_variation=True, 
                 enable_SDT=True, 
                 verbose=True):
        """
        Model with endogenous wages and resource dynamics.
        Includes mid-run shocks and plots.

        :param pt_original: Whether to use Peter's original parameters and initialization (True) or Jim's variation (False)
        :param quick_adjust: If True, accelerates resource growth/depletion for quicker simulation adjustments
        :param initialize_SIR: Whether to initialize the SIR model with original parameters and states, or load from file
        :param show_SIR_variation: Whether to sample multiple SIR starting points to illustrate variation in trajectories
        :param enable_SDT: Whether to enable Structural Demographic Theory (SDT) influences on radicalization/wages
        :param verbose: If True, prints wage transition logs
        """
        # Model Flags
        self.pt_original = pt_original
        self.quick_adjust = quick_adjust
        self.initialize_SIR = initialize_SIR
        self.show_SIR_variation = show_SIR_variation
        self.enable_SDT = enable_SDT
        self.verbose = verbose

        # SIR Parameters
        self.T_ages = 45           # Number of age classes
        self.sigma_0 = 0.0005      # Base spontaneous radicalization rate/fraction per year
        self.a_0 = 0.1             # Modulates base radicalization level
        self.a_max = 1             # Maximum fraction of susceptibles that can radicalize
        self.gamma = 1             # Fraction of the "recovered" group that influences new radicalization
        self.delta = 0.5           # Fraction of radicals converting to moderate per time step
        self.tau = 10              # Time delay for infected (radicalized) people to transition to recovered
        self.period_n = 1          # Number of periods in the initialization dimension
        self.I_0 = 0.1             # Initial fraction of infected (radicalized)
        self.R_0 = 0.0             # Initial fraction of recovered (moderate)

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

        # SDT Parameters
        self.mu_0 = 0.3 / 100      # Rate of upward mobility to elites
        self.a_w = 1.0             # How radicalization changes with worker income deviation
        self.a_e = 0.5 * 100       # How radicalization changes with elite fraction deviation
        self.YB_year = 0           # If >0, the year in which a youth bulge is centered (0 means none)
        self.age_factor = 0.2      # Peak amplitude factor of youth bulge
        self.w_0 = 0.90            # "Normal" wage baseline
        self.w_m = 0.75            # Alternate wage level (used in transitions array below)
        self.e_0 = 0.01            # Expected proportion of elites
        self.show_year = 2020      # Vertical line for reference in plots

        # Some exogenous transitions for legacy wages
        self.transitions = np.array([
            [1810, self.w_0], [1840, self.w_0], [1850, self.w_m],
            [1910, self.w_m], [1940, self.w_0], [1970, self.w_0],
            [2000, self.w_m], [2020, self.w_m], [2025, self.w_0], [2100, self.w_0]
        ])

        # Disable SDT if not enabled
        if not self.enable_SDT:
            self.a_w, self.a_e, self.YB_year = 0.0, 0.0, 0

        # Define simulation period
        self.period = np.arange(self.transitions[0,0], self.transitions[-1,0] + 1)

        # Build youth bulge
        self.YB_A20 = np.zeros(len(self.period))
        if self.YB_year:
            self.YB_A20 = self.age_factor * np.exp(-((self.period - self.YB_year) ** 2) / 10**2)
        else:
            self.YB_year = '(None)'

        if self.verbose:
            print("Model initialized. Here are the wage transitions")

        for i in range(1, len(self.transitions)):
            eyr, syr = self.transitions[i, 0], self.transitions[i - 1, 0]
            delta_t = eyr - syr
            change = 100 * (self.transitions[i, 1] - self.transitions[i - 1, 1]) / delta_t
            if self.verbose:
                print(f'{int(syr)}-{int(eyr)}: {int(delta_t)} years {change:.1f}%')

        self.eps_factor = (1 - self.w_0) / self.e_0

        # Natural Resource Parameters
        self.nat_res_max = 2.0         # Maximum carrying capacity for natural resources
        self.nat_res_regen = 0.05      # Logistic-growth coefficient for resource regeneration
        self.delta_extract = 0.03      # Base rate of resource extraction
        self.mu_elite_extr = 0.5       # Multiplier for extraction due to elite fraction
        self.alpha_w = 1.0             # Exponent controlling wage response to resource ratio
        self.eta_deplet = 1.0          # Exponent controlling how wage influences depletion

        self.nat_res_array = np.full(len(self.period), np.nan)
        self.nat_res_array[0] = 1.0

        self.results = []



    def initialize_parameters(self):
        self.S0[0, :] = (1 - self.I_0 - self.R_0) / self.T_ages
        self.I0[0, :] = self.I_0 / self.T_ages
        self.R0[0, :] = self.R_0 / self.T_ages
        self.SIR_starts = [0]

    def load_initial_state(self):
        try:
            with open('initial_SIR.pkl', 'rb') as fh:
                if sys.version_info[:3] >= (3, 0):
                    (self.T_ages, self.a_0, self.a_max, self.gamma, self.sigma_0, 
                     self.delta, self.tau, self.S0, self.I0, self.R0) = pickle.load(fh, encoding='latin1')
                else:
                    (self.T_ages, self.a_0, self.a_max, self.gamma, self.sigma_0, 
                     self.delta, self.tau, self.S0, self.I0, self.R0) = pickle.load(fh)
        except:
            raise RuntimeError("Unable to load initial_SIR.pkl!")

    def wage_function(self, nat_res_t, elit_t):
        """
        Endogenized wage from the current natural resource level (nat_res_t) and elite fraction.
        """
        nat_res_0 = self.nat_res_array[0]
        numerator = (1 - elit_t) * nat_res_t
        denominator = (1 - self.e_0) * nat_res_0
        return self.w_0 * (numerator / denominator)**self.alpha_w

    def depletion_function(self, w_t, elit_t):
        """
        Compute how much resource is extracted/depleted this period, given wage and elite fraction.
        """
        return self.delta_extract * (1 + self.mu_elite_extr * elit_t) * (w_t**self.eta_deplet)

    def resource_update(self, nat_res_t, dep_t):
        """
        Update natural resource stock using logistic growth and depletion:
         - growth = nat_res_regen * nat_res_t * (1 - nat_res_t / nat_res_max)
        """
        growth = self.nat_res_regen * nat_res_t * (1 - nat_res_t / self.nat_res_max)
        if self.quick_adjust:
            growth *= 1.5
            dep_t *= 1.5
        return nat_res_t + growth - dep_t

    def simulate(self, S0, I0, R0, shock_type=None, shock_year=None, shock_magnitude=0.0):
        """
        Generates Simulation and incorporates shock in particular year.

        Args:
            S0 (_type_): 
            I0 (_type_):
            R0 (_type_): 
            shock_type (_type_, optional): _description_. Defaults to None.
            shock_year (_type_, optional): _description_. Defaults to None.
            shock_magnitude (float, optional): _description_. Defaults to 0.0.
        """

        S = np.zeros((self.T_ages, len(self.period)))
        I = np.zeros((self.T_ages, len(self.period)))
        R = np.zeros((self.T_ages, len(self.period)))
        S[:, 0], I[:, 0], R[:, 0] = S0, I0, R0

        S_sum = np.full(len(self.period), np.nan)
        I_sum = np.full(len(self.period), np.nan)
        R_sum = np.full(len(self.period), np.nan)
        alpha_rec = np.full(len(self.period), np.nan)

        S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:,0]), np.sum(I[:,0]), np.sum(R[:,0])

        elit = np.full(len(self.period), np.nan)
        elit[0] = self.e_0
        epsln = np.full(len(self.period), np.nan) # Surplus per Elite
        epsln[0] = self.eps_factor * (1 - self.wage_function(self.nat_res_array[0], elit[0])) / elit[0]

        nat_res_local = self.nat_res_array.copy()  # for reference in shocks

        for t in range(len(self.period)-1):
            current_year = self.period[t]
            t1 = t + 1

            if (shock_type is not None) and (shock_year is not None):
                if current_year == shock_year:
                    if shock_type == 'resource':
                        nat_res_local[t] += shock_magnitude
                        nat_res_local[t] = max(nat_res_local[t], 0.0)
                    elif shock_type == 'elite':
                        new_e = elit[t] + shock_magnitude
                        elit[t] = np.clip(new_e, 0.0, 1.0)
                    elif shock_type == 'reduce_regen':
                        new_regen = self.nat_res_regen + shock_magnitude
                        self.nat_res_regen = max(new_regen, 0.0)

            w_t = self.wage_function(nat_res_local[t], elit[t])

            # Elite fraction update
            if self.pt_original:
                elit[t1] = elit[t] + self.mu_0*(self.w_0 - w_t)/w_t
                elit[t1] -= (elit[t1] - self.e_0)*R_sum[t]
            else:
                elit[t1] = elit[t] + self.mu_0*(self.w_0 - w_t)/w_t - (elit[t] - self.e_0)*R_sum[t]
            elit[t1] = np.clip(elit[t1], 0.0, 1.0)

            w_t1 = self.wage_function(nat_res_local[t], elit[t1])
            epsln[t1] = self.eps_factor * (1 - w_t1) / elit[t1]

            alpha = np.clip(
                self.a_0 + self.a_w*(self.w_0 - w_t1) + self.a_e*(elit[t1] - self.e_0) + self.YB_A20[t1],
                self.a_0, self.a_max
            )
            alpha_rec[t1] = alpha

            sigma = np.clip((alpha - self.gamma*np.sum(R[:,t])) * np.sum(I[:,t]) + self.sigma_0, 0, 1)
            rho = np.clip(self.delta * (I_sum[t-self.tau] if t>self.tau else 0), 0, 1)

            # Age progression for S, I, R in the SIR model
            S[0, t1] = 1 / self.T_ages
            for age in range(self.T_ages - 1):
                S[age+1, t1] = (1 - sigma)*S[age, t]
                I[age+1, t1] = (1 - rho)*I[age, t] + sigma*S[age, t]
                R[age+1, t1] = R[age, t] + rho*I[age, t]

            S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(S[:,t1]), np.sum(I[:,t1]), np.sum(R[:,t1])

            if self.pt_original:
                elit[t1] -= (elit[t1] - self.e_0)*R_sum[t1]

            # Update natural resource stock
            dep_t = self.depletion_function(w_t, elit[t])
            nat_res_local[t1] = self.resource_update(nat_res_local[t], dep_t)

        return (S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln, nat_res_local)

    def run_model(self, shock_type=None, shock_year=None, shock_magnitude=0.0):
        self.results = []
        for y_i in self.SIR_starts:
            y_i = int(y_i)
            S0, I0, R0 = self.S0[:, y_i], self.I0[:, y_i], self.R0[:, y_i]
            (S, I, R, S_sum, I_sum, R_sum, alpha_rec, elit, epsln, nat_res_local) = \
                self.simulate(S0, I0, R0, shock_type, shock_year, shock_magnitude)
            
            w_array = np.array([
                self.wage_function(nat_res_local[t], elit[t]) 
                for t in range(len(self.period))
            ])
            
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
                'nat_res_array': nat_res_local,  # was 'R_nat'
                'YB_A20': self.YB_A20
            })

    def plot_results(self):
        """
        Plot the results from the last run_model() call
        For baseline vs. shock on the same figure, use compare_with_shock_on_one_plot().
        """
        if not self.results:
            print("No results to plot! Run `model.run_model()` first.")
            return

        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        
        brown = '#A52A2A'
        nice_green = '#00BFFF'
        handles_collector = []

        for idx, result in enumerate(self.results):
            S_sum = result['S_sum']
            I_sum = result['I_sum']
            R_sum = result['R_sum']
            alpha_rec = result['alpha_rec']
            elit = result['elit']
            w_array = result['w']
            nat_res_array = result['nat_res_array']
            YB_A20 = result['YB_A20']

            # TOP subplot
            if self.enable_SDT:
                l1, = ax[0].plot(self.period, w_array, color=brown, linestyle='-', linewidth=2,
                                 label='(w) Endogenous wage' if idx == 0 else "_nolegend_")
                l2, = ax[0].plot(self.period, YB_A20, color='k', linestyle='-.', linewidth=2,
                                 label=f'Youth bulge {self.YB_year}' if idx == 0 else "_nolegend_")
                l3, = ax[0].plot(self.period, elit*100, color='b', linestyle='-',
                                 label='(e*100) Elite fraction' if idx == 0 else "_nolegend_")
                l4, = ax[0].plot(self.period, alpha_rec, color=nice_green, linestyle='--', linewidth=2,
                                 label=r'($\alpha$) radicalization rate' if idx == 0 else "_nolegend_")

                if idx == 0:
                    handles_collector += [l1, l2, l3, l4]

            l5, = ax[0].plot(self.period, I_sum, color='r', linestyle='-',
                             label='(I) Radical fraction' if idx == 0 else "_nolegend_")
            l6, = ax[0].plot(self.period, R_sum, color='k', linestyle='-',
                             label='(R) Moderate fraction' if idx == 0 else "_nolegend_")

            if idx == 0:
                handles_collector += [l5, l6]

            if (self.show_year in self.period) and (idx == 0):
                ax[0].axvline(x=self.show_year, color='k', linestyle='--')

            # BOTTOM subplot: Natural Resource
            l7, = ax[1].plot(self.period, nat_res_array, color='green', linestyle='-', linewidth=2,
                             label='Natural Resource Stock' if idx == 0 else "_nolegend_")

            if idx == 0:
                handles_collector.append(l7)

        # Show nat_res_max as a dashed line
        # ax[1].axhline(self.nat_res_max, color='gray', linestyle='--', label='nat_res_max')

        # Format subplots
        ax[0].set_title('Social and Economic Variables')
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Fraction / Rate / Level')
        ax[0].grid(True)
        ax[0].set_xlim([self.period[0], self.period[-1]])

        ax[1].set_title('Natural Resource Dynamics')
        ax[1].set_xlabel('Year')
        ax[1].set_ylabel('Resource Stock')
        ax[1].grid(True)

        fig.legend(handles=handles_collector,
                   loc='center left',
                   bbox_to_anchor=(0.92, 0.5))
        plt.tight_layout(rect=[0,0,0.90,1])
        plt.show()

    def compare_with_shock_on_one_plot(self, shock_type, shock_year, shock_magnitude):
        """
        1) Run baseline (no shock)
        2) Run shock scenario
        3) Plot both on a single figure to see the difference
        """

        # Baseline (no shock)
        self.run_model(shock_type=None, shock_year=None, shock_magnitude=0.0)
        baseline_result = self.results[0]

        orig_nat_res_regen = self.nat_res_regen # Save if modified

        baseline_nat_res = baseline_result['nat_res_array'].copy()
        baseline_w = baseline_result['w'].copy()
        baseline_elit = baseline_result['elit'].copy()
        baseline_I = baseline_result['I_sum'].copy()
        baseline_R = baseline_result['R_sum'].copy()
        baseline_alpha = baseline_result['alpha_rec'].copy()

        # Shock Scenario
        self.nat_res_regen = orig_nat_res_regen  # restore original
        self.run_model(shock_type=shock_type, shock_year=shock_year, shock_magnitude=shock_magnitude)
        shock_result = self.results[0]

        shock_nat_res = shock_result['nat_res_array']
        shock_w = shock_result['w']
        shock_elit = shock_result['elit']
        shock_I = shock_result['I_sum']
        shock_R = shock_result['R_sum']
        shock_alpha = shock_result['alpha_rec']

        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        
        brown = '#A52A2A'
        nice_green = '#00BFFF'

        # TOP subplot
        ax[0].plot(self.period, baseline_w, color=brown, linestyle='-', linewidth=2, 
                   label='(w) Wage (Baseline)')
        ax[0].plot(self.period, shock_w, color=brown, linestyle='--', linewidth=2, 
                   label='(w) Wage (Shock)')

        ax[0].plot(self.period, baseline_elit*100, color='b', linestyle='-', 
                   label='Elite *100 (Baseline)')
        ax[0].plot(self.period, shock_elit*100, color='b', linestyle='--', 
                   label='Elite *100 (Shock)')

        ax[0].plot(self.period, baseline_I, color='r', linestyle='-', 
                   label='(I) Radical (Baseline)')
        ax[0].plot(self.period, shock_I, color='r', linestyle='--', 
                   label='(I) Radical (Shock)')

        ax[0].plot(self.period, baseline_R, color='k', linestyle='-', 
                   label='(R) Moderate (Baseline)')
        ax[0].plot(self.period, shock_R, color='k', linestyle='--', 
                   label='(R) Moderate (Shock)')

        # Radicalization rate
        ax[0].plot(self.period, baseline_alpha, color=nice_green, linestyle='-', 
                   label=r'$\alpha$ (Baseline)')
        ax[0].plot(self.period, shock_alpha, color=nice_green, linestyle='--', 
                   label=r'$\alpha$ (Shock)')

        ax[0].set_title(f'Counterfactual vs Shock: {shock_type}, year={shock_year}, mag={shock_magnitude}')
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Levels / Fractions')
        ax[0].grid(True)
        ax[0].legend(loc='best')

        # Resource Subplot
        ax[1].plot(self.period, baseline_nat_res, color='green', linestyle='-', linewidth=2,
                   label='Natural Resource (Baseline)')
        ax[1].plot(self.period, shock_nat_res,    color='green', linestyle='--', linewidth=2,
                   label='Natural Resource (Shock)')

        # ax[1].axhline(self.nat_res_max, color='gray', linestyle='--', label='nat_res_max')
        ax[1].set_title('Resource Stock Comparison')
        ax[1].set_xlabel('Year')
        ax[1].set_ylabel('Resource Stock')
        ax[1].grid(True)
        ax[1].legend(loc='best')

        plt.tight_layout()
        plt.show()

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


### Test Run

model = SIRModel(
    pt_original=False,
    quick_adjust=False,
    initialize_SIR=False,
    show_SIR_variation=False,
    enable_SDT=True,
    verbose=False
)

# Compare with a resource shock at year=1960, magnitude=-.25
model.run_model()
model.plot_results()
model.compare_with_shock_on_one_plot('reduce_regen', 1960, -0.01)
model.compare_with_shock_on_one_plot('resource', 1960, -0.25)
model.compare_with_shock_on_one_plot('elite', 1960, 0.05)

