import numpy as np
import matplotlib.pyplot as plt

# Enable DST forcing on SIR model
enable_DST = 1

# Initialize S0, I0, R0, and SIR_parms with starting values
T_ages = 45
initial_fraction = 1 / T_ages

S0 = np.tile(initial_fraction * 0.9, (T_ages, 60))
I0 = np.tile(initial_fraction * 0.1, (T_ages, 60))
R0 = np.tile(initial_fraction * 0.0, (T_ages, 60))
SIR_parms = np.array([0])  # Example starting parameter

show_SIR_variation = 0
if show_SIR_variation:
    SIR_increment = 5
    SIR_starts = np.arange(SIR_parms[0], S0.shape[1], SIR_increment)
else:
    SIR_increment = 0
    SIR_starts = [SIR_parms[0]]

pt_original = 0
force_one_percent_elites = 1
if pt_original:
    force_one_percent_elites = 0

qin_transitions = 0
display_SIR_data = 1
if enable_DST == 0:
    display_SIR_data = 1

ls = '-'
ls_psi = '--'
lw_psi = 2
add_title = 1

# US transitions and parameters
YB_year = 1965
age_factor = 0.2

w_0 = 0.90
w_m = 0.75

transitions = np.array([
    [1810, w_0], [1840, w_0], [1850, w_m], [1990, w_m],
    [1910, w_m], [1940, w_0], [1970, w_0], [2000, w_m],
    [2020, w_m], [2025, w_0], [2100, w_0]
])

tag = 'US: Mockup historical w; fast recovery 2025'

# Interpolating relative income
period = np.arange(transitions[0, 0], transitions[-1, 0] + 1)
w = np.interp(period, transitions[:, 0], transitions[:, 1], left=w_0, right=w_0)

# Reporting the transitions
for i in range(1, transitions.shape[0]):
    eyr = transitions[i, 0]
    syr = transitions[i - 1, 0]
    delta_t = eyr - syr
    print(f'{int(syr)}-{int(eyr)}: {int(delta_t)} years {100 * (transitions[i, 1] - transitions[i - 1, 1]) / delta_t:.1f}%')

# Radicalization parameters
if force_one_percent_elites:
    elit_0 = 0.01
    max_elit = 0.03
    elit_1 = elit_0
    eps_factor = (1 - w_0) / elit_0
else:
    elit_0 = 0.5
    max_elit = 0.6
    elit_1 = 0.5
    eps_factor = 10 * elit_0

mu = 0.3
a_w = 1.0
a_e = 0.5

scale = 100
mu /= scale
a_e *= 100

if not enable_DST:
    a_w = 0.0
    a_e = 0.0
    YB_year = 0.0

# Base SIR model parameters
sigma_0 = 0.0005
a_0 = 0.1
a_max = 1
gamma = 1
delta = 0.5
tau = 10

T_ages = 45
t_ages = 45

I_0 = 0.1
R_0 = 0.0

YB_A20 = np.zeros(len(period))
if YB_year:
    YB_A20 = age_factor * np.exp(-((period - YB_year) ** 2) / 10**2)

elit = np.full(len(period), np.nan)
elit[0] = elit_1
epsln = np.full(len(period), np.nan)
epsln[0] = eps_factor * (1 - w[0]) / elit[0]

S = np.zeros((T_ages, len(period)))
S[:, 0] = (1 - I_0 - R_0) / t_ages
I = np.zeros((T_ages, len(period)))
I[:, 0] = I_0 / t_ages
R = np.zeros((T_ages, len(period)))
R[:, 0] = R_0 / t_ages

# Display colors
brown = '#A52A2A'
nice_green = '#00BFFF'

# Adjust figure size
fig1, ax1 = plt.subplots(figsize=(12, 8))
fig2, ax2 = plt.subplots(figsize=(12, 8))

for y_i in SIR_starts:
    S[:, 0] = S0[:, y_i]
    I[:, 0] = I0[:, y_i]
    R[:, 0] = R0[:, y_i]

    S_sum = np.full(len(period), np.nan)
    I_sum = np.full(len(period), np.nan)
    R_sum = np.full(len(period), np.nan)
    alpha_rec = np.full(len(period), np.nan)

    if not pt_original:
        R_sum[0] = np.sum(R[:, 0])
    
    S_sum[0] = np.sum(S[:, 0])
    I_sum[0] = np.sum(I[:, 0])
    R_sum[0] = np.sum(R[:, 0])

    for t in range(len(period) - 1):
        t1 = t + 1
        if pt_original:
            elit[t1] = elit[t] + mu * (w_0 - w[t]) / w[t]
        else:
            elit[t1] = elit[t] + mu * (w_0 - w[t]) / w[t] - (elit[t] - elit_0) * R_sum[t]

        epsln[t1] = eps_factor * (1 - w[t1]) / elit[t1]
        
        alpha = a_0 + a_w * (w_0 - w[t1]) + a_e * (elit[t1] - elit_0)
        alpha += YB_A20[t1]
        alpha = np.clip(alpha, a_0, a_max)
        
        sigma = (alpha - gamma * np.sum(R[:, t])) * np.sum(I[:, t]) + sigma_0
        sigma = np.clip(sigma, 0, 1)
        if t > tau:
            rho = delta * np.sum(I[:, t - tau])
        else:
            rho = 0
        rho = np.clip(rho, 0, 1)
        alpha_rec[t1] = alpha
        
        S[0, t1] = 1 / t_ages
        for age in range(t_ages - 1):
            age1 = age + 1
            S[age1, t1] = (1 - sigma) * S[age, t]
            I[age1, t1] = (1 - rho) * I[age, t] + sigma * S[age, t]
            R[age1, t1] = R[age, t] + rho * I[age, t]
        
        S_sum[t1] = np.sum(S[:, t1])
        I_sum[t1] = np.sum(I[:, t1])
        R_sum[t1] = np.sum(R[:, t1])
        if pt_original:
            elit[t1] = elit[t1] - (elit[t1] - elit_0) * R_sum[t1]

    ax1.plot(period, w, color=brown, linestyle=ls, linewidth=2, label='(w) Relative income')
    if YB_year:
        ax1.plot(period, YB_A20, color='k', linestyle='-.', linewidth=2, label=f'Youth bulge {YB_year}')
    if display_SIR_data:
        ax1.plot(period, I_sum, color='r', linestyle=ls, label='(I) Radical fraction')
        ax1.plot(period, R_sum, color='k', linestyle=ls, label='(R) Moderate fraction')
    
    ax1.plot(period, elit * scale, color='b', linestyle=ls, label=f'(e*{scale}) Elite fraction of population')
    x_i = np.where(elit > max_elit)[0]
    if x_i.size > 0:
        ax1.plot(period[x_i], elit[x_i] * scale, color='r', linestyle='-', linewidth=2, label='Excessive elite fraction')
    
    ax1.plot(period, alpha_rec, color=nice_green, linestyle=ls_psi, linewidth=lw_psi, label='(\\alpha) elite forcing (\\Psi)')
    
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.grid(True)
    ax1.set_ylim([-0.1, 2.5])
    ax1.set_xlim([period[0], period[-1]])
    if add_title:
        ax1.set_title(f'{tag} {SIR_increment}')
    
    this_year = 2020
    if this_year in period:
        ax1.axvline(x=this_year, color='k', linestyle='--')


    ax2.plot(period, I_sum, color='r', linestyle=ls, label='(I) Radical fraction')
    ax2.plot(period, R_sum, color='k', linestyle=ls, label='(R) Moderate fraction')
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Fraction', fontsize=12)
    ax2.grid(True)
    ax2.set_ylim([-0.1, 1.0]) 
    ax2.set_xlim([period[0], period[-1]])
    ax2.set_title(f'SIR: \\sigma_0: {sigma_0:.4f} \\alpha = {a_0:.2f} \\gamma = {gamma:.2f} \\delta = {delta:.2f} \\tau = {tau:.1f} years', fontsize=12)
    
    plt.pause(0.1)

plt.show()