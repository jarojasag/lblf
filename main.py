#!/usr/bin/env python3
"""
Climate-Inequality-Radicalization Model
Main execution script
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from core_model import CoreClimateInequalityModel
from visualization import (
    plot_results,
    plot_scenario_comparison
)
from analysis import calculate_metrics, run_policy_comparison
from utils import validate_config, save_results, export_metrics_to_csv

def main():
    """Main execution function"""
    print("="*60)
    print("Climate-Inequality-Radicalization Model")
    print("Age-Structured Population with Endogenous Inequality")
    print("="*60)

    # Validate configuration
    config_file = "config.yml"
    if not validate_config(config_file):
        print("ERROR: Invalid configuration file")
        exit(1)

    # 1. Run base simulation
    print("\n1. Running base simulation...")
    model = CoreClimateInequalityModel(config_file)
    results = model.simulate_age_structured(time_horizon=50, start_year=2020)

    # Calculate and display metrics
    metrics = calculate_metrics(model)

    print("\n" + "="*40)
    print("Base Simulation Metrics")
    print("="*40)
    print(f"Max radicalization: {metrics['max_radicalization']:.1%}")
    print(f"Final radicalization: {metrics['final_radicalization']:.1%}")
    print(f"Max PSI: {metrics['max_PSI']:.2f}")
    print(f"Final PSI: {metrics['final_PSI']:.2f}")
    print(f"Max Gini: {metrics['max_gini']:.3f}")
    print(f"Final temperature: {metrics['final_temperature']:.2f}°C")
    print(f"Trust collapse: {'Yes' if metrics['trust_collapse'] else 'No'}")

    # Plot results
    print("\nGenerating plots...")
    plot_results(model, save_path="base_simulation.png")

    # 2. Run policy comparison
    print("\n2. Running policy comparison...")
    policy_results = run_policy_comparison(config_file, use_adaptive=False)

    print("\n" + "="*40)
    print("Policy Comparison Results")
    print("="*40)
    for scenario_name, data in policy_results.items():
        metrics = data['metrics']
        print(f"\n{scenario_name}:")
        print(f"  Max radicalization: {metrics['max_radicalization']:.1%}")
        print(f"  Final temperature: {metrics['final_temperature']:.2f}°C")
        print(f"  Trust collapse: {'Yes' if metrics['trust_collapse'] else 'No'}")

    plot_scenario_comparison(policy_results, model.T_ages)

if __name__ == "__main__":
    main()