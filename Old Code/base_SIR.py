import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SIRModel:
    def __init__(self, beta, gamma, N, I0, R0):
        """
        Initializes the SIR Model.
        :param beta: The effective contact rate of the disease.
        :param gamma: The recovery rate.
        :param N: Total population.
        :param I0: Initial number of infected individuals.
        :param R0: Initial number of recovered individuals.
        """
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.I0 = I0
        self.R0 = R0
        self.S0 = N - I0 - R0  # Susceptible count

    def diff_eqs(self, y, t):
        """
        The differential equations of the SIR model.
        :param y: A tuple containing the current values of S, I, and R.
        :param t: Time step for simulation (not used here explicitly).
        """
        S, I, R = y
        N = self.N
        beta = self.beta
        gamma = self.gamma
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def run_simulation(self, days):
        """
        Run the SIR model simulation.
        :param days: Number of days to simulate.
        """
        t = np.linspace(0, days, days)
        initial_conditions = self.S0, self.I0, self.R0
        result = odeint(self.diff_eqs, initial_conditions, t)
        return t, result

    def plot_results(self, t, result):
        """
        Plot the results of the SIR model simulation.
        """
        S, I, R = result.T
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, 'b', label='Susceptible')
        plt.plot(t, I, 'r', label='Infected')
        plt.plot(t, R, 'g', label='Recovered')
        plt.xlabel('Time / days')
        plt.ylabel('Number')
        plt.title('SIR Model Simulation')
        plt.legend()
        plt.show()

# Parameters
N = 1000          # Total population
beta = 0.3        # Infection rate
gamma = 0.1       # Recovery rate
I0 = 1            # Initial number of infected individuals
R0 = 0            # Initial number of recovered individuals
days = 160        # Duration of simulation

# Initialize and run the model
model = SIRModel(beta, gamma, N, I0, R0)
t, result = model.run_simulation(days)
model.plot_results(t, result)