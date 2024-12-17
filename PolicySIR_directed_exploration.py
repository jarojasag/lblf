# Excample Run 

ema_model = Model('PolicySIR', function=policySIRModel)

# Uncertainties 
ema_model.uncertainties = [
    RealParameter('a_w', 0.1, 100),
    RealParameter('a_e', 0.1, 100)
 ]

# Constants
# ema_model.constants = [
#     Constant('a_w', 1.0),
#     Constant('a_e', 50.0)
# ]

# levers
ema_model.levers = [
    RealParameter('eta_w', 0.0, 1.0),
    RealParameter('eta_a', 0.0, 1.0),  # Assuming same eta_a for all ages
    RealParameter('w_T', 0.8, 1.0),
    RealParameter('I_Ta', 0.0, 0.1),  # Assuming same I_Ta for all ages
]

# Outcomes
ema_model.outcomes = [
    ScalarOutcome('max_radicalized'), # kind=ScalarOutcome.MINIMIZE),
    ScalarOutcome('final_radicalized'), # kind=ScalarOutcome.MINIMIZE),
    ScalarOutcome('policy_cost') #, kind=ScalarOutcome.MINIMIZE),
]

# Convergence metrics
convergence_metrics = [HyperVolume(minimum=[0, 0, 0], maximum=[1, 1, 10]),
                        EpsilonProgress()]

# Use SequentialEvaluator
with SequentialEvaluator(ema_model) as evaluator:
    results = evaluator.optimize(nfe=1000, searchover='levers',
                                    epsilons=[0.01, 0.01, 0.1],
                                    convergence=convergence_metrics)

# Analyze and plot the results
experiments, outcomes = results
results_df = pd.DataFrame.from_dict(experiments)
outcomes_df = pd.DataFrame.from_dict(outcomes)

# Combine experiments and outcomes
df = pd.concat([results_df, outcomes_df], axis=1)

# Plot parallel coordinates
limits = parcoords.get_limits(df)
paraxes = parcoords.ParallelAxes(limits)
paraxes.plot(df)
plt.show()
