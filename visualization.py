import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec  # Add this import
import pandas as pd
from typing import Dict, List, Any

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec  # Add this import
import pandas as pd
from typing import Dict, List, Any

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_results(model, save_path=None):
    """Create comprehensive visualization of model results"""
    if not hasattr(model, 'results'):
        raise ValueError("Run simulation first!")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Climate-Inequality-Radicalization Model Results', fontsize=16, fontweight='bold')
    
    results = model.results
    years = results['years']
    time_points = years - years[0]
    
    # Calculate PSI timeline
    PSI_timeline = []
    for i in range(len(results['I_sum'])):
        PSI, _, _, _ = model.compute_political_stress_index(
            results['I_sum'][i], results['elit'][i],
            results['trust'][i], results['gini'][i],
            results.get('debt_ratio', [0.5]*len(years))[i]
        )
        PSI_timeline.append(PSI)
    
    # 1. Population dynamics
    axes[0, 0].plot(time_points, results['S_sum'], label='Susceptible', linewidth=2)
    axes[0, 0].plot(time_points, results['I_sum'], label='Radicalized', linewidth=2, color='red')
    axes[0, 0].plot(time_points, results['R_sum'], label='Moderate', linewidth=2, color='green')
    axes[0, 0].set_title('Population Dynamics')
    axes[0, 0].set_xlabel('Time (years)')
    axes[0, 0].set_ylabel('Fraction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PSI
    axes[0, 1].plot(time_points, PSI_timeline, linewidth=2, color='purple')
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='Critical')
    axes[0, 1].set_title('Political Stress Index')
    axes[0, 1].set_xlabel('Time (years)')
    axes[0, 1].set_ylabel('PSI')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Inequality
    axes[0, 2].plot(time_points, results['beta'], linewidth=2, color='orange')
    axes[0, 2].set_title('Capital/Income Ratio (β)')
    axes[0, 2].set_xlabel('Time (years)')
    axes[0, 2].set_ylabel('β')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Fiscal Indicators (replacing Elite Fraction)
    ax_fiscal = axes[1, 0]
    
    # Calculate fiscal metrics if not in results
    if 'gov_revenue' not in results:
        # Simulate fiscal dynamics based on other variables
        base_revenue = 0.35  # Base tax revenue as % of GDP
        base_expenditure = 0.33  # Base expenditure as % of GDP
        
        gov_revenue = []
        gov_expenditure = []
        debt_to_gdp = []
        debt = 0.5  # Initial debt/GDP ratio
        
        for i in range(len(time_points)):
            # Revenue decreases with declining trust and increasing damage
            revenue = base_revenue * results['trust'][i] * (1 - results['damage'][i])
            gov_revenue.append(revenue)
            
            # Expenditure increases with radicalization and climate damage
            crisis_spending = 0.1 * results['I_sum'][i] + 0.15 * results['damage'][i]
            expenditure = base_expenditure + crisis_spending
            gov_expenditure.append(expenditure)
            
            # Update debt
            deficit = expenditure - revenue
            debt += deficit
            debt_to_gdp.append(debt)
    else:
        gov_revenue = results['gov_revenue']
        gov_expenditure = results['gov_expenditure']
        debt_to_gdp = results.get('debt_to_gdp', results.get('debt_ratio', [0.5]*len(time_points)))
    
    # Plot fiscal indicators
    ax_fiscal.plot(time_points, np.array(gov_revenue)*100, label='Revenue', linewidth=2, color='green')
    ax_fiscal.plot(time_points, np.array(gov_expenditure)*100, label='Expenditure', linewidth=2, color='red')
    ax_fiscal2 = ax_fiscal.twinx()
    ax_fiscal2.plot(time_points, np.array(debt_to_gdp)*100, label='Debt/GDP', linewidth=2, color='darkblue', linestyle='--')
    
    ax_fiscal.set_title('Fiscal Indicators')
    ax_fiscal.set_xlabel('Time (years)')
    ax_fiscal.set_ylabel('% of GDP', color='black')
    ax_fiscal2.set_ylabel('Debt/GDP (%)', color='darkblue')
    ax_fiscal.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax_fiscal.get_legend_handles_labels()
    lines2, labels2 = ax_fiscal2.get_legend_handles_labels()
    ax_fiscal.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax_fiscal.grid(True, alpha=0.3)
    
    # 5. Climate Damage
    axes[1, 1].plot(time_points, results['damage'] * 100, linewidth=2, color='darkred')
    axes[1, 1].fill_between(time_points, 0, results['damage'] * 100, alpha=0.3, color='darkred')
    axes[1, 1].set_title('Climate Damage')
    axes[1, 1].set_xlabel('Time (years)')
    axes[1, 1].set_ylabel('GDP Loss (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Temperature
    axes[1, 2].plot(time_points, results['temperature'], linewidth=2, color='red')
    axes[1, 2].axhline(y=1.5, color='orange', linestyle='--', label='1.5°C Target')
    axes[1, 2].axhline(y=2.0, color='red', linestyle='--', label='2°C Threshold')
    axes[1, 2].set_title('Global Temperature Anomaly')
    axes[1, 2].set_xlabel('Time (years)')
    axes[1, 2].set_ylabel('Temperature (°C)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Trust
    axes[2, 0].plot(time_points, results['trust'], linewidth=2, color='blue')
    axes[2, 0].axhline(y=0.3, color='red', linestyle='--', label='Crisis Level')
    axes[2, 0].set_title('Institutional Trust')
    axes[2, 0].set_xlabel('Time (years)')
    axes[2, 0].set_ylabel('Trust Level')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Cohort Trajectory Tracking
    ax = axes[2, 1]

    # Select a few cohorts born at different times
    cohort_birth_years = [0, 10, 20, 30, 40]  # Birth years in simulation time
    colors = plt.cm.Set2(np.linspace(0, 1, len(cohort_birth_years)))

    for idx, birth_year in enumerate(cohort_birth_years):
        if birth_year < len(time_points) - model.T_ages:
            cohort_I = []
            cohort_ages = []
            
            # Track this cohort as it ages
            for age in range(min(model.T_ages, len(time_points) - birth_year)):
                time_idx = birth_year + age
                if time_idx < len(time_points) and age < model.T_ages:
                    cohort_I.append(results['I'][age, time_idx])
                    cohort_ages.append(age)
            
            ax.plot(cohort_ages, cohort_I, 
                label=f'Cohort Year {birth_year}',
                linewidth=2, color=colors[idx], marker='o', markersize=3)

    ax.set_title('Cohort Radicalization Trajectories')
    ax.set_xlabel('Age')
    ax.set_ylabel('Radicalization Rate')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 9. Gini coefficient
    axes[2, 2].plot(time_points, results['gini'], linewidth=2, color='purple')
    axes[2, 2].axhline(y=0.35, color='orange', linestyle='--', label='Unrest Threshold')
    axes[2, 2].set_title('Gini Coefficient')
    axes[2, 2].set_xlabel('Time (years)')
    axes[2, 2].set_ylabel('Gini')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_scenario_comparison(scenario_results: Dict, T_ages: int):
    """Plot comparison of different policy scenarios"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Policy Scenario Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_results)))
    
    for idx, (name, data) in enumerate(scenario_results.items()):
        sim = data['simulation']
        years = sim['years']
        
        # 1. Radicalization
        axes[0, 0].plot(years, sim['I_sum'], label=name, linewidth=2, color=colors[idx])
        
        # 2. PSI
        axes[0, 1].plot(years, data['PSI'], label=name, linewidth=2, color=colors[idx])
        
        # 3. Inequality
        axes[0, 2].plot(years, sim['beta'], label=name, linewidth=2, color=colors[idx])
        
        # 4. Fiscal health (replacing Elite fraction)
        if 'gov_revenue' in sim and 'gov_expenditure' in sim:
            fiscal_balance = np.array(sim['gov_revenue']) - np.array(sim['gov_expenditure'])
            axes[1, 0].plot(years, fiscal_balance * 100, label=name, linewidth=2, color=colors[idx])
        
        # 5. Climate damage
        axes[1, 1].plot(years, sim['damage'] * 100, label=name, linewidth=2, color=colors[idx])
        
        # 6. Temperature
        axes[1, 2].plot(years, sim['temperature'], label=name, linewidth=2, color=colors[idx])
        
        # 7. Trust
        axes[2, 0].plot(years, sim['trust'], label=name, linewidth=2, color=colors[idx])
    
        # 8. Debt ratio
        if 'debt_ratio' in sim:
            axes[2, 1].plot(years, sim['debt_ratio'], label=name, linewidth=2, color=colors[idx])
        
        # 9. Final age distribution
        axes[2, 2].plot(np.arange(T_ages), sim['I'][:, -1], 
                       label=name, linewidth=2, color=colors[idx])
    
    # Set titles and labels
    titles = [
        'Total Radicalization', 'Political Stress Index', 'Inequality (β)',
        'Fiscal Balance (% GDP)', 'Climate Damage (%GDP)', 'Temperature (°C)',
        'Institutional Trust', 'Debt/GDP Ratio', 'Final Age Distribution'
    ]
    
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        if title != 'Final Age Distribution':
            ax.set_xlabel('Year')
        else:
            ax.set_xlabel('Age')
    
    plt.tight_layout()
    plt.show()
    return fig