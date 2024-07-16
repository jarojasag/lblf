import numpy as np
from lblf import SIRModel
from ema_workbench import (Model, RealParameter, ScalarOutcome, SequentialEvaluator, Policy)
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)

class PolicySIR:
    def __init__(self, sir_model, nw, na, wr, Ir_a):
        """
        Initialize the PolicySIR class with the SIR model and policy parameters.

        :param sir_model: Instance of the SIRModel class
        :param nw: Coefficient for the income adjustment controller
        :param na: Coefficient for the radicalization adjustment controller
        :param wr: Target level for the income available to workers
        :param Ir_a: Target level for tolerable radicalization in cohort a
        """
        self.sir_model = sir_model
        self.nw = nw
        self.na = na
        self.wr = wr
        self.Ir_a = Ir_a
        self.Dw = np.zeros(len(self.sir_model.period))  # Policy adjustment array

    def pid_controller(self):
        """
        Apply the PID controller formalism to adjust relative income.
        """
        for t in range(len(self.sir_model.period)):
            Ia_t = np.sum(self.sir_model.I[:, t])  # Number of radicalized people in cohort at time t
            self.Dw[t] = self.nw * max(self.wr - self.sir_model.w[t], 0) + self.na * max(Ia_t - self.Ir_a, 0)
            self.sir_model.w[t] += self.Dw[t]

    def policy_cost(self):
        """
        Calculate the cost of the policy adjustments.

        :return: Total policy cost
        """
        return np.sum(self.Dw)

    def run_with_policy(self):
        """
        Run the SIR model with the policy adjustments.
        """
        self.pid_controller()
        self.sir_model.run_model()

    def plot_results_with_policy(self):
        """
        Plot the results of the SIR model simulation with policy adjustments.
        """
        self.sir_model.plot_results(title_suffix="with Policy")

def optimize_policy(sir_model):
    """
    Optimize the policy parameters using ema_workbench.

    :param sir_model: Instance of the SIRModel class
    """
    def policy_function(nw, na, wr, Ir_a):
        policy_sir = PolicySIR(sir_model, nw, na, wr, Ir_a)
        policy_sir.run_with_policy()
        cost = policy_sir.policy_cost()
        return {
            'policy_cost': cost,
            'radical_fraction': np.mean(policy_sir.sir_model.I.sum(axis=0))
        }

    model = Model('policySIR', function=policy_function)
    model.uncertainties = [
        RealParameter('nw', 0, 1),
        RealParameter('na', 0, 1),
        RealParameter('wr', 0, 2),
        RealParameter('Ir_a', 0, 1)
    ]

    model.outcomes = [
        ScalarOutcome('policy_cost', kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome('radical_fraction', kind=ScalarOutcome.MINIMIZE)
    ]

    policies = [Policy('no_policy', **{'nw': 0, 'na': 0, 'wr': 0, 'Ir_a': 0})]

    with SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(nfe=1000, searchover='levers', epsilons=[0.1, 0.1])

    return results


### TEst

sir_model = SIRModel(initialize_SIR=False, show_SIR_variation=False, enable_SDT=True)
sir_model.run_model()
sir_model.plot_results()

results = optimize_policy(sir_model)
print(results)