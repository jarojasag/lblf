import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core_model import CoreClimateInequalityModel

def calculate_metrics(model) -> Dict[str, Any]:
    """Calculate key metrics from simulation results"""
    if not hasattr(model, 'results'):
        raise ValueError("Run simulation first!")
    
    results = model.results
    config = model.config
    metrics = {}
    
    # Extract variables
    I_sum = results['I_sum']
    trust = results['trust']
    temperature = results['temperature']
    damage = results['damage']
    beta = results['beta']
    gini = results['gini']
    
    # Calculate PSI timeline
    PSI_timeline = []
    for i in range(len(I_sum)):
        # Fix: Check if 'elite' exists, use correct field name
        elite_value = results.get('elite', results.get('elit', [0]*len(I_sum)))[i]
        
        # Fix: Handle debt_ratio properly
        debt_ratio_value = results.get('debt_ratio', [0.5]*len(I_sum))[i]
        
        PSI, MMP, EMP, SFD = model.compute_political_stress_index(
            I_sum[i], 
            elite_value, 
            trust[i], 
            gini[i],
            debt_ratio_value
        )
        PSI_timeline.append(PSI)
    
    # Summary metrics
    metrics['max_radicalization'] = np.max(I_sum)
    metrics['final_radicalization'] = I_sum[-1]
    metrics['max_PSI'] = np.max(PSI_timeline)
    metrics['final_PSI'] = PSI_timeline[-1]
    metrics['max_inequality'] = np.max(beta)
    metrics['final_inequality'] = beta[-1]
    metrics['max_gini'] = np.max(gini)
    metrics['final_gini'] = gini[-1]
    metrics['total_climate_damage'] = damage[-1]
    metrics['final_temperature'] = temperature[-1]
    metrics['final_trust'] = trust[-1]
    
    # Threshold crossings
    thresholds = config['thresholds']
    metrics['trust_collapse'] = trust[-1] < thresholds['trust_collapse']
    metrics['radicalization_tipping'] = np.any(I_sum > thresholds['radicalization_tipping'])
    metrics['PSI_critical'] = np.any(np.array(PSI_timeline) > config['policy']['implementation_threshold'])
    metrics['temperature_2C'] = np.any(temperature > thresholds['temperature_2C'])
    
    # Age-specific metrics - Fix: Check if T_ages exists
    if hasattr(model, 'T_ages'):
        T_ages = model.T_ages
    else:
        T_ages = model.config.get('age_structure', {}).get('T_ages', 30)
    
    # Fix: Check if 'I' exists in results and has the right shape
    if 'I' in results and len(results['I'].shape) > 1:
        final_I_by_age = results['I'][:, -1]
        metrics['youth_radicalization'] = np.mean(final_I_by_age[:15]) if T_ages > 15 else np.mean(final_I_by_age)
        metrics['adult_radicalization'] = np.mean(final_I_by_age[15:30]) if T_ages > 30 else np.mean(final_I_by_age[15:])
        metrics['elder_radicalization'] = np.mean(final_I_by_age[30:]) if T_ages > 30 else 0
    else:
        # Fallback if age-structured data isn't available
        metrics['youth_radicalization'] = I_sum[-1]
        metrics['adult_radicalization'] = I_sum[-1]
        metrics['elder_radicalization'] = I_sum[-1]
    
    # Timing metrics
    if metrics['radicalization_tipping']:
        metrics['time_to_tipping'] = np.argmax(I_sum > thresholds['radicalization_tipping'])
    else:
        metrics['time_to_tipping'] = -1
    
    if metrics['temperature_2C']:
        metrics['time_to_2C'] = np.argmax(temperature > thresholds['temperature_2C'])
    else:
        metrics['time_to_2C'] = -1
    
    return metrics

def run_policy_comparison(config_file: str = "config.yml", time_horizon: int = 50, 
                         start_year: int = 2020, use_adaptive: bool = False) -> Dict:
    """Compare different policy scenarios"""
    
    scenarios = {
        'Business as Usual': None,
        'Early Intervention (Year 2)': {
            'start_year': start_year + 2,
            'policies': ['carbon_tax', 'wealth_tax', 'social_programs']
        },
        'Late Intervention (Year 10)': {
            'start_year': start_year + 10,
            'policies': ['carbon_tax', 'wealth_tax', 'social_programs']
        },
        'Climate Only': {
            'start_year': start_year + 2,
            'policies': ['carbon_tax']
        },
        'Inequality Only': {
            'start_year': start_year + 2,
            'policies': ['wealth_tax', 'social_programs']
        },
        'Combined Climate-Inequality': {
            'start_year': start_year + 2,
            'policies': ['carbon_tax', 'wealth_tax']
        },
        'Social Programs Only': {
            'start_year': start_year + 2,
            'policies': ['social_programs']
        }
    }
    
    results = {}
    
    for name, intervention in scenarios.items():
        print(f"Running scenario: {name}")
        
        try:
            # Choose model type
            if use_adaptive and intervention is not None:
                model = AdaptivePolicyModel(config_file)
            else:
                model = CoreClimateInequalityModel(config_file)
            
            # Prepare parameters for simulate_age_structured
            if intervention is None:
                # Business as usual - no policies
                sim_results = model.simulate_age_structured(
                    time_horizon=time_horizon,
                    start_year=start_year,
                    policies=None,
                    policy_start_year=None,
                    effectiveness=1.0
                )
            else:
                # Extract policies and start year from intervention
                sim_results = model.simulate_age_structured(
                    time_horizon=time_horizon,
                    start_year=start_year,
                    policies=intervention['policies'],
                    policy_start_year=intervention['start_year'],
                    effectiveness=1.0
                )
            
            # Store results in model for calculate_metrics
            model.results = sim_results
            
            # Calculate metrics
            metrics = calculate_metrics(model)
            
            # Calculate PSI timeline with proper error handling
            PSI_timeline = []
            I_sum = sim_results.get('I_sum', [])
            
            # Handle different field names for elite
            elite_values = sim_results.get('elite', sim_results.get('elit', [0]*len(I_sum)))
            trust_values = sim_results.get('trust', [0.5]*len(I_sum))
            gini_values = sim_results.get('gini', [0.35]*len(I_sum))
            debt_ratio_values = sim_results.get('debt_ratio', [0.5]*len(I_sum))
            
            for i in range(len(I_sum)):
                try:
                    PSI, _, _, _ = model.compute_political_stress_index(
                        I_sum[i],
                        elite_values[i] if i < len(elite_values) else 0,
                        trust_values[i] if i < len(trust_values) else 0.5,
                        gini_values[i] if i < len(gini_values) else 0.35,
                        debt_ratio_values[i] if i < len(debt_ratio_values) else 0.5
                    )
                    PSI_timeline.append(PSI)
                except Exception as e:
                    print(f"Warning: Error calculating PSI at time {i}: {e}")
                    PSI_timeline.append(0)
            
            results[name] = {
                'simulation': sim_results,
                'metrics': metrics,
                'PSI': PSI_timeline,
                'model': model
            }
            
        except Exception as e:
            print(f"Error in scenario '{name}': {e}")
            # Store partial results even if scenario fails
            results[name] = {
                'simulation': None,
                'metrics': {'error': str(e)},
                'PSI': [],
                'model': None
            }
    
    return results