import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Enable DST forcing on SIR model
enable_DST = True

# Initialize constants
T_ages = 45
initial_fraction = 1 / T_ages

# Initialize S0, I0, R0 arrays
S0 = np.tile(initial_fraction * 0.9, (T_ages, 60))
I0 = np.tile(initial_fraction * 0.1, (T_ages, 60))
R0 = np.tile(initial_fraction * 0.0, (T_ages, 60))
SIR_parms = np.array([0])  # Example starting parameter, because not in file

show_SIR_variation = False
SIR_increment = 5 if show_SIR_variation else 0
SIR_starts = np.arange(SIR_parms[0], S0.shape[1], SIR_increment) if show_SIR_variation else [SIR_parms[0]]

pt_original = False
force_one_percent_elites = not pt_original

qin_transitions = False
display_SIR_data = not enable_DST # Could delete, check with Jim

# US transitions and parameters
YB_year = 1965
age_factor = 0.2

w_0 = 0.90
w_m = 0.75

transitions = np.array([
    [1810, w_0], [1840, w_0], [1850, w_m], 
    [1910, w_m], [1940, w_0], [1970, w_0], [2000, w_m],
    [2020, w_m], [2025, w_0], [2100, w_0]
])

# Interpolating relative income
period = np.arange(transitions[0, 0], transitions[-1, 0] + 1)
interp_fn = PchipInterpolator(transitions[:, 0], transitions[:, 1])
w = interp_fn(period)

# Reporting the transitions
for i in range(1, len(transitions)):
    eyr, syr = transitions[i, 0], transitions[i - 1, 0]
    delta_t = eyr - syr
    change = 100 * (transitions[i, 1] - transitions[i - 1, 1]) / delta_t
    print(f'{int(syr)}-{int(eyr)}: {int(delta_t)} years {change:.1f}%')

# Radicalization parameters
elit_0, max_elit = (0.01, 0.03) if force_one_percent_elites else (0.5, 0.6)
elit_1, eps_factor = elit_0, (1 - w_0) / elit_0 if force_one_percent_elites else (0.5, 10 * elit_0)

mu, a_w, a_e = 0.3, 1.0, 0.5
scale = 100
mu /= scale
a_e *= scale

if not enable_DST:
    a_w, a_e, YB_year = 0.0, 0.0, 0.0

# Base SIR model parameters
sigma_0, a_0, a_max = 0.0005, 0.1, 1
gamma, delta, tau = 1, 0.5, 10

I_0, R_0 = 0.1, 0.0

YB_A20 = np.zeros(len(period))
if YB_year:
    YB_A20 = age_factor * np.exp(-((period - YB_year) ** 2) / 10**2)

elit = np.full(len(period), np.nan)
elit[0] = elit_1
epsln = np.full(len(period), np.nan)
epsln[0] = eps_factor * (1 - w[0]) / elit[0]

S, I, R = np.zeros((T_ages, len(period))), np.zeros((T_ages, len(period))), np.zeros((T_ages, len(period)))
S[:, 0], I[:, 0], R[:, 0] = (1 - I_0 - R_0) / T_ages, I_0 / T_ages, R_0 / T_ages

# Plot Data
tag = 'US: Mockup historical w; fast recovery 2025'
brown, nice_green = '#A52A2A', '#00BFFF'

# Adjust figure size
fig1, ax1 = plt.subplots(figsize=(12, 8))
fig2, ax2 = plt.subplots(figsize=(12, 8))

for y_i in SIR_starts:
    S[:, 0], I[:, 0], R[:, 0] = S0[:, y_i], I0[:, y_i], R0[:, y_i]

    S_sum, I_sum, R_sum, alpha_rec = [np.full(len(period), np.nan) for _ in range(4)]
    S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:, 0]), np.sum(I[:, 0]), np.sum(R[:, 0])

    for t in range(len(period) - 1):
        t1 = t + 1
        elit[t1] = elit[t] + mu * (w_0 - w[t]) / w[t] - (elit[t] - elit_0) * R_sum[t] if not pt_original else elit[t] + mu * (w_0 - w[t]) / w[t]
        epsln[t1] = eps_factor * (1 - w[t1]) / elit[t1]
        
        alpha = np.clip(a_0 + a_w * (w_0 - w[t1]) + a_e * (elit[t1] - elit_0) + YB_A20[t1], a_0, a_max)
        sigma = np.clip((alpha - gamma * np.sum(R[:, t])) * np.sum(I[:, t]) + sigma_0, 0, 1)
        rho = np.clip(delta * np.sum(I[:, t - tau]) if t > tau else 0, 0, 1)
        alpha_rec[t1] = alpha
        
        S[0, t1] = 1 / T_ages
        for age in range(T_ages - 1):
            age1 = age + 1
            S[age1, t1] = (1 - sigma) * S[age, t]
            I[age1, t1] = (1 - rho) * I[age, t] + sigma * S[age, t]
            R[age1, t1] = R[age, t] + rho * I[age, t]
        
        S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(S[:, t1]), np.sum(I[:, t1]), np.sum(R[:, t1])
        if pt_original:
            elit[t1] -= (elit[t1] - elit_0) * R_sum[t1]

    ax1.plot(period, w, color=brown, linestyle='-', linewidth=2, label='(w) Relative income')
    if YB_year:
        ax1.plot(period, YB_A20, color='k', linestyle='-.', linewidth=2, label=f'Youth bulge {YB_year}')
    if display_SIR_data:
        ax1.plot(period, I_sum, color='r', linestyle='-', label='(I) Radical fraction')
        ax1.plot(period, R_sum, color='k', linestyle='-', label='(R) Moderate fraction')
    
    ax1.plot(period, elit * scale, color='b', linestyle='-', label=f'(e*{scale}) Elite fraction of population')
    excessive_elite = np.where(elit > max_elit)[0]
    if excessive_elite.size > 0:
        ax1.plot(period[excessive_elite], elit[excessive_elite] * scale, color='r', linestyle='-', linewidth=2, label='Excessive elite fraction')
    
    ax1.plot(period, alpha_rec, color=nice_green, linestyle='--', linewidth=2, label=r'($\alpha$) elite forcing ($\Psi$)')
    
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.grid(True)
    ax1.set_ylim([-0.1, 2.5])
    ax1.set_xlim([period[0], period[-1]])
    if True:
        ax1.set_title(f'{tag} {SIR_increment}')
    
    if 2020 in period:
        ax1.axvline(x=2020, color='k', linestyle='--')

    ax2.plot(period, I_sum, color='r', linestyle='-', label='(I) Radical fraction')
    ax2.plot(period, R_sum, color='k', linestyle='-', label='(R) Moderate fraction')
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Fraction', fontsize=12)
    ax2.grid(True)
    ax2.set_ylim([-0.1, 1.0])