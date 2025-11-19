import numpy as np
from typing import Dict, List, Tuple, Optional
from parameters import ModelParameters

class CoreClimateInequalityModel:
    """
    Core Climate-Inequality-Radicalization Model
    Integrates age-structured SIR, economic dynamics, climate damage, and political stress
    """
    
    def __init__(self, config_file: str = "config.yml"):
        """Initialize model with parameters from config file"""
        self.params = ModelParameters(config_file)
        self.config = self.params.config
        self.results = {}
        
        # Initialize from config
        self.T_ages = self.config['population']['T_ages']
        self._initialize_state_variables()
    
    def _initialize_state_variables(self):
        """Initialize state variables from config"""
        init = self.config['initial_conditions']
        
        # Age-structured arrays
        self.S0 = np.zeros((self.T_ages, 1))
        self.I0 = np.zeros((self.T_ages, 1))
        self.R0 = np.zeros((self.T_ages, 1))
        
        S_total = 1 - init['I_total'] - init['R_total']
        self.S0[0, 0] = S_total / self.T_ages
        self.I0[0, 0] = init['I_total'] / self.T_ages
        self.R0[0, 0] = init['R_total'] / self.T_ages
        
        # Other initial conditions
        self.initial_state = {
            'I_total': init['I_total'],        
            'R_total': init['R_total'],        
            'beta': init['beta'],
            'elit': init['elit'],
            'damage': init['damage'],
            'trust': init['trust'],
            'temperature': init['temperature'],
            'debt_ratio': init['debt_ratio']
        }
    
    def calculate_inequality_pressure(self, beta_t, elit_t):
        """Calculate inequality pressure and Gini coefficient"""
        ineq = self.config['inequality']
        
        beta_dev = (beta_t - self.initial_state['beta']) / self.initial_state['beta']
        elite_dev = (elit_t - self.config['elite']['e_0']) / self.config['elite']['e_0']
        
        ineq_index = ineq['beta_weight'] * beta_dev + ineq['elite_weight'] * elite_dev
        
        # Calculate Gini using logistic mapping
        g_min = ineq['gini_min']
        g_max = ineq['gini_max']
        g_base = ineq['gini_baseline']
        
        k = ineq['gini_logit_slope']
        g_base_prime = (g_base - g_min) / (g_max - g_min)
        g_base_prime = np.clip(g_base_prime, 1e-6, 1 - 1e-6) #safey
        logit_base = np.log(g_base_prime / (1 - g_base_prime))
        logit_gini = logit_base + k * ineq_index
        g_prime = 1 / (1 + np.exp(-logit_gini))
        gini_approx = g_min + (g_max - g_min) * g_prime
        
        # radicalization multiplier
        gini_excess = max(0.0, gini_approx - ineq['gini_threshold'])
        inequality_multiplier = 1.0 + self.config['radicalization']['a_inequality'] * \
                              (gini_excess ** ineq['inequality_exponent']) # larger non-linear effect 
        
        return inequality_multiplier, gini_approx
    
    def climate_damage_function(self, temperature, gdp, method='ensemble'):
        """Calculate climate damage using configured damage functions"""
        climate = self.config['climate']
        damage_funcs = climate['damage_functions']
        weights = climate['damage_weights']
        
        # Calculate damages
        damages = {
            'nordhaus': damage_funcs['nordhaus_coeff'] * (temperature**2) * gdp,
            'meta': damage_funcs['meta_coeff'] * (temperature**2) * gdp,
            'hsiang': (damage_funcs['hsiang_linear'] * temperature + 
                      damage_funcs['hsiang_nonlinear'] * temperature**damage_funcs['hsiang_exponent']) * gdp
        }
        
        if method in damages:
            return damages[method]
        
        # Ensemble average
        total_damage = sum(damages[k] * weights[k] for k in damages)
        return min(total_damage, climate['damage_cap'])
    
    def compute_political_stress_index(self, I_sum, elit, trust, gini, debt_ratio=None):
        """Calculate Political Stress Index and components"""
        psi_config = self.config['political_stress']
        
        if debt_ratio is None:
            debt_ratio = self.initial_state['debt_ratio']
        
        # Mass Mobilization Potential (MMP)
        # z_I = 0.0
        # if I_sum > psi_config['I_threshold']:
        #    z_I = (I_sum / 0.05) ** 2
        #    z_I = min(z_I * psi_config['mmp_radicals'], psi_config['psi_component_cap'])
        
        z_G = 0.0
        if gini > self.config['inequality']['gini_threshold']:
            gini_excess = (gini - self.config['inequality']['gini_threshold']) / 0.05
            z_G = gini_excess * psi_config['mmp_ineq']
        
        MMP = z_G 
        
        # Elite Mobilization Potential (EMP)
        z_e = 0.0
        if elit > psi_config['e_threshold']:
            z_e = ((elit - psi_config['e_threshold']) / psi_config['e_threshold']) ** 1.5 # make 1.5 uncertainty
            z_e = z_e * psi_config['emp_slope']
        
        EMP = z_e
        
        # State Fiscal Distress (SFD)
        z_T = 0.0
        if trust < self.initial_state['trust']:
            trust_deficit = (self.initial_state['trust'] - trust) / self.initial_state['trust']
            z_T = trust_deficit ** 1.5
            z_T = z_T * psi_config['sfd_trust']
        
        z_D = 0.0
        if debt_ratio > psi_config['debt_baseline']:
            debt_excess = (debt_ratio - psi_config['debt_baseline']) / (1.0 - psi_config['debt_baseline'])
            z_D = debt_excess * psi_config['sfd_debt']
        
        SFD = z_T + z_D
        
        # Combine components
        PSI = MMP * (1 + EMP) * (1 + SFD)
        
        return PSI, MMP, EMP, SFD
    
    def policy_effectiveness(self, PSI):
        """Calculate policy effectiveness based on political stress"""
        psi_effect = self.config['political_stress']['psi_effect']
        return 1 / (1 + psi_effect * PSI)
    


    def calculate_policy_effects(self, policies, effectiveness, beta_current):
        """
        Calculate policy effects for current timestep.
        
        Args:
            policies: List of active policy names (empty list for baseline)
            effectiveness: Policy effectiveness factor (0-1)
            beta_current: Current emissions level (for context-dependent effects)
        
        Returns:
            Dictionary of policy effects (multipliers and financial impacts)
        """
        # Start with base effects (no policy scenario)
        policy_effects = self.config['policy']['base_effects'].copy()
        
        # Return base effects if no policies are active
        if not policies:
            return policy_effects
        
        policy_cfg = self.config['policy']
        
        # Apply each active policy's effects
        for policy in policies:
            if policy == 'carbon_tax':
                policy_effects['emissions_mult'] *= (1 - policy_cfg['carbon_tax']['emissions_reduction'] * effectiveness)
                policy_effects['revenue'] += policy_cfg['carbon_tax']['revenue_rate'] * beta_current * effectiveness
                
            if policy == 'wealth_tax':
                policy_effects['savings_mult'] *= (1 - policy_cfg['wealth_tax']['savings_reduction'] * effectiveness)
                policy_effects['elite_lambda_mult'] *= (1 - policy_cfg['wealth_tax']['elite_lambda_reduction'] * effectiveness)
                policy_effects['revenue'] += policy_cfg['wealth_tax']['revenue_base'] * effectiveness
                
            if policy == 'social_programs':
                policy_effects['delta_mult'] *= (1 + policy_cfg['social_programs']['delta_boost'] * effectiveness)
                policy_effects['trust_theta_mult'] *= (1 + policy_cfg['social_programs']['trust_theta_boost'] * effectiveness)
                policy_effects['spending'] += policy_cfg['social_programs']['spending_rate'] * effectiveness
                
        policy_effects['net_budget'] = policy_effects['revenue'] - policy_effects['spending']
        
        return policy_effects

    def simulate_age_structured(self, time_horizon=46, start_year=2020, policies=None, policy_start_year=None, effectiveness=1.0):
        """
        Run age-structured simulation with radicalization dynamics (SIR) and policy interventions.
        
        Args:
            time_horizon: Number of years to simulate
            start_year: Starting year of simulation
            policies: List of policy names to implement (None or empty for baseline)
            policy_start_year: Year to start implementing policies
            effectiveness: Policy effectiveness/enforcement level (0-1)
        
        Returns:
            Dictionary containing time series of all state variables
        """
        years = np.arange(start_year, start_year + time_horizon + 1)
        n_years = len(years)
        
        # Initialize arrays for SIR (radicalization tracking)
        S = np.zeros((self.T_ages, n_years))  # Susceptible to radicalization
        I = np.zeros((self.T_ages, n_years))  # Radicalized
        R = np.zeros((self.T_ages, n_years))  # Recovered from radicalization
        
        # Set initial SIR conditions
        S[0, 0] = 1 - self.initial_state['I_total'] - self.initial_state['R_total']
        I[0, 0] = self.initial_state['I_total']
        R[0, 0] = self.initial_state['R_total']
        
        # Initialize state variables
        beta = np.full(n_years, self.initial_state['beta'])  # Wealth concentration
        elit = np.full(n_years, self.initial_state['elit'])
        damage = np.full(n_years, self.initial_state['damage'])
        trust = np.full(n_years, self.initial_state['trust'])
        temperature = np.full(n_years, self.initial_state['temperature'])
        debt_ratio = np.full(n_years, self.initial_state['debt_ratio'])
        gini = np.zeros(n_years)
        
        # Initialize policy tracking arrays
        policy_revenue = np.zeros(n_years)
        policy_spending = np.zeros(n_years)
        net_budget = np.zeros(n_years)
        emissions_reduction = np.zeros(n_years)
        
        # Calculate initial values
        S_sum = np.zeros(n_years)
        I_sum = np.zeros(n_years)
        R_sum = np.zeros(n_years)
        S_sum[0] = np.sum(S[:, 0])
        I_sum[0] = np.sum(I[:, 0])
        R_sum[0] = np.sum(R[:, 0])
        
        _, gini[0] = self.calculate_inequality_pressure(beta[0], elit[0])
        
        # Main simulation loop
        for t in range(n_years - 1):
            t1 = t + 1
            
            # Determine if policies are active
            active_policies = []
            if policies and policy_start_year and years[t1] >= policy_start_year:
                active_policies = policies
            
            # Calculate policy effects
            policy_effects = self.calculate_policy_effects(
                active_policies,
                effectiveness,
                beta[t]  # Using beta as proxy for current economic/emissions state
            )
            
            # Store policy metrics
            policy_revenue[t1] = policy_effects['revenue']
            policy_spending[t1] = policy_effects['spending']
            net_budget[t1] = policy_effects['net_budget']
            
            # Economic dynamics
            growth_rate_base = self.config['economic']['growth_rate_base']
            growth_penalty = damage[t] * self.config['economic']['damage_growth_multiplier']
            growth_rate = growth_rate_base - growth_penalty
            r_minus_g = self.config['economic']['return_on_capital'] - growth_rate
            
            # Beta (wealth concentration) dynamics with policy effects on savings
            if abs(growth_rate) > 1e-6:
                # Apply savings multiplier from policies (e.g., wealth tax reduces elite savings)
                adjusted_savings = self.config['economic']['savings_rate'] * policy_effects['savings_mult']
                beta_change = np.tanh(adjusted_savings / growth_rate - 1.0)
                trust_deficit = max(0, self.initial_state['trust'] - trust[t])
                concentration_effect = 1.0 + trust_deficit * 0.2
                beta[t1] = beta[t] * (1 + beta_change * concentration_effect * 0.1)
            else:
                beta[t1] = beta[t] * 1.02
            
            beta[t1] = np.clip(beta[t1], self.initial_state['beta'], 20.0)
            
            # Elite dynamics with policy effects
            elite_config = self.config['elite']
            elit_change = (elite_config['lambda_elit'] * r_minus_g * elit[t] - 
                        (elit[t] - elite_config['e_0']) * R_sum[t])
            elit[t1] = elit[t] + elit_change
            # Apply wealth tax effect on elite accumulation
            elit[t1] *= policy_effects['elite_lambda_mult']
            elit[t1] = np.clip(elit[t1], 0.0, 0.2)
            
            # Calculate inequality pressure
            inequality_multiplier, gini[t1] = self.calculate_inequality_pressure(beta[t1], elit[t1])
            
            # Radicalization dynamics (affected by inequality and climate damage)
            rad_config = self.config['radicalization']
            alpha = np.clip(
                rad_config['a_0'] * inequality_multiplier + 
                rad_config['a_e'] * max(0, elit[t1] - elite_config['e_0']) +
                rad_config['alpha_d'] * damage[t],
                rad_config['a_0'],
                rad_config['a_max']
            )
            
            # SIR dynamics for radicalization spread
            sigma = np.clip(
                (alpha - rad_config['gamma'] * R_sum[t]) * I_sum[t] + rad_config['sigma_0'],
                0, 1
            )
            
            if t >= rad_config['tau']:
                tau_idx = t - rad_config['tau']
                rho = np.clip(rad_config['delta'] * I_sum[tau_idx], 0, 1)
            else:
                rho = 0
            
            # Age progression
            if t >= self.T_ages:
                exit_pop = S[self.T_ages-1, t] + I[self.T_ages-1, t] + R[self.T_ages-1, t]
                entry_pop = exit_pop
            else:
                entry_pop = 1.0 / self.T_ages
            
            S[0, t1] = entry_pop
            
            for age in range(self.T_ages - 1):
                age1 = age + 1
                S[age1, t1] = (1 - sigma) * S[age, t]
                I[age1, t1] = (1 - rho) * I[age, t] + sigma * S[age, t]
                R[age1, t1] = R[age, t] + rho * I[age, t]
            
            S_sum[t1] = np.sum(S[:, t1])
            I_sum[t1] = np.sum(I[:, t1])
            R_sum[t1] = np.sum(R[:, t1])
            
            # Climate dynamics with policy effects
            climate = self.config['climate']
            
            # Calculate damage (with potential productivity boost from social programs)
            target_damage = self.climate_damage_function(temperature[t], 1.0)
            adapted_damage = target_damage * (1 - climate['kappa_adaptation'])
            # Social programs can help with adaptation
            damage_reduction = 1.0 / policy_effects['delta_mult']  # Higher productivity reduces effective damage
            damage[t1] = damage[t] + climate['damage_adjustment_rate'] * (adapted_damage * damage_reduction - damage[t])
            damage[t1] = np.clip(damage[t1], 0.0, climate['damage_cap'])
            
            # Emissions with policy effects (carbon tax reduces emissions)
            base_emissions = climate['emissions_rate'] * (1 + growth_rate)
            emissions = base_emissions * policy_effects['emissions_mult']
            emissions_reduction[t1] = (1 - policy_effects['emissions_mult']) * 100
            
            temperature[t1] = np.clip(temperature[t] + emissions, 0.0, 10.0)
            
            # Trust dynamics with policy effects
            trust_config = self.config['trust']
            trust_erosion = 0.0
            
            if beta[t1] > self.initial_state['beta']:
                beta_erosion = (beta[t1] - self.initial_state['beta']) / self.initial_state['beta']
                trust_erosion += trust_config['omega_beta'] * beta_erosion
            
            trust_erosion += trust_config['omega_damage'] * damage[t1]
            trust_erosion += trust_config['omega_I'] * I_sum[t1]  # Radicalization erodes trust
            
            PSI_current, _, _, _ = self.compute_political_stress_index(
                I_sum[t], elit[t], trust[t], gini[t], debt_ratio[t]
            )
            psi_effectiveness = self.policy_effectiveness(PSI_current)
            
            # Trust building with social program boost
            trust_building = (trust_config['theta'] * trust_config['baseline_trust_building'] * 
                            psi_effectiveness * policy_effects['trust_theta_mult'])
            
            trust[t1] = np.clip(
                trust[t] - trust_erosion + trust_building,
                trust_config['min_trust'],
                trust_config['max_trust']
            )
            
            # Update debt ratio considering policy fiscal impacts
            debt_growth = -net_budget[t1] * 0.01  # Policies affect debt
            debt_ratio[t1] = debt_ratio[t] + debt_growth
        
        # Store results
        self.results = {
            'years': years,
            'S': S, 'I': I, 'R': R,
            'S_sum': S_sum, 'I_sum': I_sum, 'R_sum': R_sum,
            'beta': beta, 'elit': elit,
            'damage': damage, 'trust': trust,
            'temperature': temperature, 'gini': gini,
            'debt_ratio': debt_ratio,
            'policy_revenue': policy_revenue,
            'policy_spending': policy_spending,
            'net_budget': net_budget,
            'emissions_reduction': emissions_reduction
        }
        
        return self.results
