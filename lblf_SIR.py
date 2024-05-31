import numpy as np
import matplotlib.pyplot as plt

class ExtendedSIRModel:
    def __init__(self, period, transitions, initial_states, w_0, elit_0, parameters):
        self.period = np.array(period)
        self.transitions = np.array(transitions)
        self.S = np.zeros_like(self.period, dtype=float)
        self.I = np.zeros_like(self.period, dtype=float)
        self.R = np.zeros_like(self.period, dtype=float)
        self.elit = np.zeros_like(self.period, dtype=float)
        self.alpha_rec = np.zeros_like(self.period, dtype=float)
        self.w = np.interp(self.period, self.transitions[:, 0], self.transitions[:, 1])
        self.w_0 = w_0  # Initialize w_0 as a class attribute
        
        # Parameters
        self.mu = parameters['mu']
        self.a_0 = parameters['a_0']
        self.a_w = parameters['a_w']
        self.a_e = parameters['a_e']
        self.sigma_0 = parameters['sigma_0']
        self.gamma = parameters['gamma']
        self.delta = parameters['delta']
        self.tau = parameters['tau']
        self.max_elit = parameters['max_elit']

        # Initial Conditions
        self.S[0], self.I[0], self.R[0], self.elit[0] = initial_states


    def simulate(self):
        for t in range(1, len(self.period)):
            # Update elites based on relative income and previous elite level
            self.elit[t] = max(0, min(self.max_elit, self.elit[t-1] + self.mu * (self.w_0 - self.w[t]) / self.w[t]))
            
            # Calculate radicalization forces
            alpha = self.a_0 + self.a_w * (self.w_0 - self.w[t]) + self.a_e * (self.elit[t] - self.elit[0])
            self.alpha_rec[t] = alpha
            
            # SIR dynamics with social influence
            sigma = max(0, min(1, (alpha - self.gamma * self.R[t-1]) * self.I[t-1] + self.sigma_0))
            rho = self.delta * self.I[t - self.tau] if t > self.tau else 0
            
            # Update SIR model
            self.S[t] = self.S[t-1] - sigma
            self.I[t] = self.I[t-1] + sigma - rho
            self.R[t] = self.R[t-1] + rho
            

    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.period, self.S, label='Susceptible')
        plt.plot(self.period, self.I, label='Infected')
        plt.plot(self.period, self.R, label='Recovered')
        plt.plot(self.period, self.elit, label='Elite fraction', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel('Fraction of Population')
        plt.title('Extended SIR Model with Socio-political Dynamics')
        plt.legend()
        plt.grid(True)
        plt.show()

# Parameters and initial conditions setup
period = np.arange(1800, 2101)  # From 1800 to 2100
transitions = [(1810, 0.9), (1840, 0.9), (1850, 0.75), (1990, 0.75), (2020, 0.7), (2050, 0.9)]
initial_states = (0.9, 0.01, 0.0, 0.01)  # Initial state of S, I, R, and initial elite fraction
parameters = {
    'mu': 0.01, 'a_0': 0.1, 'a_w': 1.0, 'a_e': 0.5, 
    'sigma_0': 0.0005, 'gamma': 1, 'delta': 0.5, 'tau': 10, 'max_elit': 0.03
}

# Create and run the model
model = ExtendedSIRModel(period, transitions, initial_states, 0.9, 0.01, parameters)
model.simulate()
model.plot()
