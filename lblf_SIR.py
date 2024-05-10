import numpy as np
import matplotlib.pyplot as plt

class SIRModel:
    def __init__(self, S0, I0, R0, transitions, period, w_0, elit_0, mu, a_0, a_max, gamma, delta, tau):
        # Initialize populations as proportions
        self.S = np.zeros((len(period),))
        self.I = np.zeros((len(period),))
        self.R = np.zeros((len(period),))
        self.elit = np.zeros((len(period),))
        self.epsln = np.zeros((len(period),))
        self.alpha_rec = np.zeros((len(period),))
        
        # Initial conditions ensuring they sum to 1
        total = S0 + I0 + R0
        self.S[0] = S0 / total
        self.I[0] = I0 / total
        self.R[0] = R0 / total

        self.transitions = transitions
        self.period = period
        self.w_0 = w_0
        self.elit_0 = elit_0
        self.mu = mu
        self.a_0 = a_0
        self.a_max = a_max
        self.gamma = gamma
        self.delta = delta
        self.tau = tau

    def simulate(self):
        for t in range(1, len(self.period)):
            # Interpolating the wage at this time point
            w = np.interp(self.period[t], self.transitions[:, 0], self.transitions[:, 1])
            alpha = self.a_0 + (self.w_0 - w) + 0.5 * (self.elit[t-1] - self.elit_0) # Radicalization factor

            # Support Validation
            delta_S = max(0, min(self.S[t-1], self.S[t-1] * alpha))
            delta_I = max(0, min(self.I[t-1], self.I[t-1] * (1 - self.delta)))
            delta_R = self.I[t-1] * self.delta
            
            self.S[t] = self.S[t-1] - delta_S
            self.I[t] = self.I[t-1] + delta_S - delta_R
            self.R[t] = self.R[t-1] + delta_R

            # Normalization
            total = self.S[t] + self.I[t] + self.R[t]
            self.S[t] /= total
            self.I[t] /= total
            self.R[t] /= total

            # Updating elite dynamics and economic factors
            self.elit[t] = max(0, min(1, self.elit[t-1] + self.mu * (self.w_0 - w) / w))
            self.epsln[t] = (1 - w) / self.elit[t]
            self.alpha_rec[t] = alpha

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.period, self.S, label='Susceptible')
        plt.plot(self.period, self.I, label='Infected')
        plt.plot(self.period, self.R, label='Recovered')
        plt.xlabel('Year')
        plt.ylabel('Population Fraction')
        plt.title('SIR Model Simulation with Socio-political Factors')
        plt.legend()
        plt.show()

# Example 
S0, I0, R0 = 0.9, 0.1, 0.0
transitions = np.array([[1810, 0.90], [1840, 0.90], [1850, 0.75], [1990, 0.75]])
period = np.arange(1800, 2000)
w_0 = 0.90
elit_0 = 0.01
mu = 0.3
a_0 = 0.1
a_max = 1
gamma = 1
delta = 0.5
tau = 10

model = SIRModel(S0, I0, R0, transitions, period, w_0, elit_0, mu, a_0, a_max, gamma, delta, tau)
model.simulate()
model.plot()