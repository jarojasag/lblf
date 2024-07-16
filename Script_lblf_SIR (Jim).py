import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

import sys
import pickle
def save_pkl_file(filename,data_tuple):
    fh = open(filename,"wb")
    pickle.dump(data_tuple,fh)
    fh.close()
    return True

def load_pkl_file(filename):
    try:
        fh = open(filename,"rb")
        if sys.version_info[:3] >= (3,0):
            data_tuple = pickle.load(fh,encoding='latin1') # built with numpy arrays
        else:
            data_tuple = pickle.load(fh)
        fh.close()
    except:
        raise RuntimeError("Unable to load %s!" % filename)
    return data_tuple

pt_original = False # which code variation to run, Peter's (True) or Jim's (False)

# There are three 'modes' of operation
# 1. Run the full model with Peter's original parameters and initialization (initialize_SIR = True; show_SIR_variation = False; enable_SDT = True)
# 2. Run the SIR model w/o the SDT model and store the SIR results for later variation  (initialize_SIR = True; show_SIR_variation = False; enable_SDT = False)
# 3. Run the full model against a single or multiple variations based on SIR starting state (initialize_SIR = False; show_SIR_variation = True/False; enable_SDT = True)
initialize_SIR = False
# Vary the initialization state of the SIR model based on a stored run
show_SIR_variation = True
## SDT impact on radicalization:
# Enable SDT forcing on SIR model; set to False to see how the SIR model oscillates by itself, w/o elite forcing
enable_SDT = True

if initialize_SIR:
    # Initialize SIR constants
    # TODO note that since the Revolutionary War the mean lifespan has increased at least a decade and thus so has politcal action
    # This means that the number of age classes that apply increases over time rather than being fixed; need to account for that.
    T_ages = 45 # number of age classes
    # TODO verify that these are the same as loaded from SIR_parms above
    # BASE EMPIRICAL SIR model w/o influence of income dynamics
    # Conversion to Radicals (I)
    sigma_0 = 0.0005 # BASE EMPIRICAL base spontaneous radicalization rate/fraction per year (normally zero for infections...unless patient zero)
    a_0 = 0.1 # BASE EMPIRICAL modulate (down) how many radicals form in the presence of radicals (more moderates, less radicals) (ala R0 for infectious diseases)
    a_max = 1 # max fraction of susceptiables than can radicalize (100% of course)
    gamma = 1 # BASE EMPIRICAL fraction of all recovered (R) that influences (slows) the conversion to radicals (via a_0)
    # Conversion to Moderates (R)
    delta = 0.5 # BASE EMPIRICAL fraction of radicalized (I) converting to moderate (R) as you move along in years (depends on the radicalized fraction 10 years prior)
    tau = 10  # BASE EMPIRICAL time delay for infected (radicalized) people to recover (become moderate) -- given that naive age starts at 18 to 21 this is 28 to 31, around the time males marry in industrial countries

    # Peter's original initialization of SIR
    I_0 = 0.1 # initially 10% are infected
    R_0 = 0.0 # no one has recovered to being moderate
    period_n = 1
    S0 = np.zeros((T_ages,period_n)); S0[0,:] = (1 - I_0 - R_0)/T_ages;
    I0 = np.zeros((T_ages,period_n)); I0[0,:] = I_0/T_ages;
    R0 = np.zeros((T_ages,period_n)); R0[0,:] = R_0/T_ages;
    SIR_starts = [0] # this is Peter's original initialization state using I_0,R_0 = 0.1,0.0; % initially 10% are infected and no one has recovered to moderate
    show_SIR_variation = False # no variation to start
    # enable_SDT = False if you want to record a different initialization sequence
else:
    # reload previous SIR state with its creation parameters
    (T_ages,a_0,a_max,gamma,sigma_0,delta,tau,S0,I0,R0) = load_pkl_file('initial_SIR.pkl')

if show_SIR_variation:
    SIR_increment = 5 # sample the variation in response in the SIR model starting distribution every 5 years after T_age years
    SIR_starts = np.arange(T_ages, S0.shape[1], SIR_increment)
else:
    SIR_increment = 0 # no variation
    SIR_starts = [min(T_ages,S0.shape[1]-1)] # avoid the strange iniitalization in Peter's startup sequence above, for at least T_age years

## SDT parameters
show_year = 0
## Parameters that impact radicalization based on YB and w0 expectations of elite and populace
mu_0 = 0.3 #  EMPIRICAL rate of upward mobility to elites given decreasing relative income (leads to 2-3x increase in elite numbers over 30 years)
## weight of (general) radicalization of populace as misery increases, proxied by decline in relative income from expected
a_w = 1.0 # how radicalization changes as relative worker income (w) changes from w_0, the EXPECTED relative income for a worker: (w_0 - w) so radicalization increases when w is less than w_0
## weight of (general) radicalization of elites given overproduction, which is encoded as increase of elites vs expected elite positions
a_e = 0.5 # how radicalization changes as relative elite numbers (e) changes from elit_0, the EXPECTED relative number of elite positions: (e - e_0) so radicalization decreases when as e is overproduced wrt to e_0

scale = 100
mu_0 /= scale
a_e *= scale

YB_year = 0 # Assume no youth bulge

## These are the exogenous drivers and display options for the example you are looking at...
# US transitions and parameters
# Plot Data
if True:
    tag = 'US: Mockup historical w; fast recovery 2025'
    YB_year = 1965
    age_factor = 0.2
    
    w_0 = 0.90
    w_m = 0.75
    e_0 = 0.01 # assume, for the US, that the expected proportion of elite at w_0 is 1% of the population
    
    show_year = 2020
    transitions = np.array([
        [1810, w_0], [1840, w_0], [1850, w_m], 
        [1910, w_m], [1940, w_0], [1970, w_0], [2000, w_m],
        # Fast recovery to w_0 by 2025 and stable for the rest of the century
        [2020, w_m], [2025, w_0], [2100, w_0]
        ])

if False:
    tag = 'US: w based in actual income data; no recovery'
     # This is the first year YB impacts radicalization but since it is gaussian shouldn't it be later? so it's peak is 1968?
    YB_year = 1965
    age_factor = 0.2

    w_0 = 0.90
    w_m = 0.75
    e_0 = 0.01 # assume, for the US, that the expected proportion of elite at w_0 is 1% of the population

    show_year = 2020
    transitions = np.array([
        [1800, 0.85], [1825, 1.20], [1863, 0.90], 
        [1885, 0.92], [1910, 0.75], [1930, 1.15], 
        [1970, 1.12], [2020, 0.75], [2100, 0.75],
        ]) # See Figure 3
    p = np.polyfit([0.7, 1.2],[w_m,w_0],1); # rel wage to rel income Figure 4 
    transitions[:,1] = np.polyval(p,transitions[:,1]); # convert to relative income
    
## End exogenous drivers

if not enable_SDT:
    # setting these parameters to zero eliminates impact of w and YB on elite and popular radicalization
    a_w, a_e,YB_year = 0.0, 0.0,0

# Interpolating relative income
period = np.arange(transitions[0, 0], transitions[-1, 0] + 1)
interp_fn = PchipInterpolator(transitions[:, 0], transitions[:, 1])
w = interp_fn(period)
YB_A20 = np.zeros(len(period))
if YB_year:
    YB_A20 = age_factor * np.exp(-((period - YB_year) ** 2) / 10**2)
else:
    YB_year = '(None)' # for legend display

# Reporting the transitions
for i in range(1, len(transitions)):
    eyr, syr = transitions[i, 0], transitions[i - 1, 0]
    delta_t = eyr - syr
    change = 100 * (transitions[i, 1] - transitions[i - 1, 1]) / delta_t
    print(f'{int(syr)}-{int(eyr)}: {int(delta_t)} years {change:.1f}%')


# What is the expectation wealth per elite when rel wage is w_0 (NOTE that if w_m is < w_0, the expectation would be higher)
eps_factor = (1 - w_0) / e_0

# Track the relative proportion of population that are elite (because of upward/downward mobilization)
elit = np.full(len(period), np.nan)
elit[0] = e_0 # assumes w[0] == w_0
# track the change in elite proportion given the attractiveness of elite mobility
epsln = np.full(len(period), np.nan) 
epsln[0] = eps_factor * (1 - w[0]) / elit[0] 

# create the SIR matricies to track proportions per age class over all the time periods
S, I, R = np.zeros((T_ages, len(period))), np.zeros((T_ages, len(period))), np.zeros((T_ages, len(period)))

#DEAD originally PT just initialized the first age class in the first year and let the model run from there
# essentially assuming that there are no other people alive above them (or they don't matter politically)
# thus it took T_ages to get 'initialize' everyone...
#DEAD I_0, R_0 = 0.1, 0.0# these were used once to generate SIR data
#DEAD S[:, 0], I[:, 0], R[:, 0] = (1 - I_0 - R_0) / T_ages, I_0 / T_ages, R_0 / T_ages

# Adjust figure size
fig1, ax1 = plt.subplots(figsize=(12, 8))
brown, nice_green = '#A52A2A', '#00BFFF'

for y_i in SIR_starts:
    y_i = int(y_i) # ensure index
    S[:, 0], I[:, 0], R[:, 0] = S0[:, y_i], I0[:, y_i], R0[:, y_i]

    S_sum, I_sum, R_sum, alpha_rec = [np.full(len(period), np.nan) for _ in range(4)]
    S_sum[0], I_sum[0], R_sum[0] = np.sum(S[:, 0]), np.sum(I[:, 0]), np.sum(R[:, 0])

    for t in range(len(period) - 1):
        t1 = t + 1
        if pt_original:
             elit[t] + mu_0 * (w_0 - w[t]) / w[t] # eqn [10] update relative elite numbers wrt total population
        else:
            elit[t1] = elit[t] + mu_0 * (w_0 - w[t]) / w[t] - (elit[t] - e_0) * R_sum[t] # eqn [13] directly update relative elite numbers wrt total population
        epsln[t1] = eps_factor * (1 - w[t1]) / elit[t1]
        
        alpha = np.clip(a_0 + a_w * (w_0 - w[t1]) + a_e * (elit[t1] - e_0) + YB_A20[t1], a_0, a_max) # eqn [14], limited to a_max
        sigma = np.clip((alpha - gamma * np.sum(R[:, t])) * np.sum(I[:, t]) + sigma_0, 0, 1) # eqn [4]
        # eqn [5] remove a fraction of radicals after the spontaneously revert and get on with lives under system == delta*I_sum(t-tau)
        rho = np.clip(delta * np.sum(I[:, t - tau]) if t > tau else 0, 0, 1)
        alpha_rec[t1] = alpha # record alpha -- the *rate* of radicalization, not the magnitude of radicals (PSI)
        
        S[0, t1] = 1 / T_ages
        for age in range(T_ages - 1):
            age1 = age + 1
            S[age1, t1] = (1 - sigma) * S[age, t] # eqn [1]
            I[age1, t1] = (1 - rho) * I[age, t] + sigma * S[age, t] # eqn [2]
            R[age1, t1] = R[age, t] + rho * I[age, t] # eqn [3]
        
        S_sum[t1], I_sum[t1], R_sum[t1] = np.sum(S[:, t1]), np.sum(I[:, t1]), np.sum(R[:, t1])
        if pt_original:
            elit[t1] -= (elit[t1] - e_0) * R_sum[t1] # eqn [13] but using t1 R_sum
    if enable_SDT:
        w_ax1,  = ax1.plot(period, w, color=brown, linestyle='-', linewidth=2, label='(w) Relative income')
        YB_ax1, = ax1.plot(period, YB_A20, color='k', linestyle='-.', linewidth=2, label=f'Youth bulge {YB_year}')

        elit_ax1,  = ax1.plot(period, elit * scale, color='b', linestyle='-', label=f'(e*{scale}) Elite fraction of population')
        alpha_ax1, = ax1.plot(period, alpha_rec, color=nice_green, linestyle='--', linewidth=2, label=r'($\alpha$) radicalization rate') # elite forcing ($\Psi$) ?? NOT PSI
    
        if show_year in period:
            ax1.axvline(x=show_year, color='k', linestyle='--')

    I_ax1, = ax1.plot(period, I_sum, color='r', linestyle='-', label='(I) Radical fraction') # PSI??
    R_ax1, = ax1.plot(period, R_sum, color='k', linestyle='-', label='(R) Moderate fraction')
    

# Only show one entry for each line, even if we compute many SIR initialization variants
handles = [w_ax1,YB_ax1,elit_ax1, alpha_ax1,I_ax1,R_ax1] if enable_SDT else [I_ax1,R_ax1]
ax1.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
ax1.set_xlabel('Year', fontsize=12)
ax1.grid(True)
ax1.set_ylim([-0.1, 2.5])
ax1.set_xlim([period[0], period[-1]])
ax1.set_title(f'{tag} {show_SIR_variation} incr: {SIR_increment}')
plt.show(block=False)


if not enable_SDT:
    # record the state of the SIR model over time as possible initialization states for later
    save_pkl_file('initial_SIR.pkl',(T_ages,a_0,a_max,gamma,sigma_0,delta,tau,S,I,R))
