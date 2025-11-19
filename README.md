# Climate–Inequality–Radicalization Integrated Model  
### A Systems Approach to Polycrisis Dynamics and Robust Decision Making  

**Author:** Javier Rojas  
**Co-Authors (Looking Back to Look Forward Project):**  
Robert Lempert · Zora Kovacic · Daniel Hoyer · Steven Popper · Jonathan Lamb · Juan Antonio Robledo  

---

## 📘 Overview

This repository hosts an integrated **Climate–Inequality–Radicalization** model developed as part of the *Looking Back to Look Forward* (LBLF) project.  

The model fuses insights from:
- Climate damage and emissions science  
- Economic inequality and capital concentration (Piketty’s r > g)  
- Structural-demographic theory and elite overproduction (Turchin)  
- Social tipping dynamics and radicalization (SIR structure)  
- Trust erosion and political stress  
- Adaptive policy under deep uncertainty  

---

## 🧠 Project Context: *Looking Back to Look Forward (LBLF)*  

This work was produced as part of the LBLF initiative integrating history, computational modeling, and DMDU exploratory analysis.

**Model Co-Authors:**  
Robert Lempert · Zora Kovacic · Daniel Hoyer · Steven Popper · Jonathan Lamb · Juan Antonio Robledo  
**Lead Modeller:** Javier Rojas  

## 📂 Repository Structure

├── main.py # Main deterministic simulation routine
├── core_model.py # Integrated system model (climate–inequality–radicalization)
├── adaptive_policy_system.py # Trigger-based adaptive policy engine
├── analysis.py # Scenario discovery, clustering, PCA, PRIM
├── visualization.py # All plots: SIR, temperature, β, trust, PSI, etc.
├── parameters.py # Loader and schema for config.yml
├── config.yml # Parameter file based on literature values
└── utils.py # Math helpers (logit, clipping, interpolation)


## ▶️ How the Main Routine Works (`main.py`)

`main.py` runs a full simulation using the parameters defined in **config.yml**.

### The routine:

1. Load configuration  
2. Initialize the model  
3. Load policy settings  
4. Simulate coupled system dynamics:  
   - Radicalization (S,I,R)  
   - Inequality (β dynamics, wealth concentration)  
   - Climate damage & temperature  
   - Trust and political stress index  
   - Fiscal and adaptive policy response  
5. Export results & generate plots  

This allows direct examination of individual crisis pathways or baseline trajectories.

---

## 🧪 Exploratory Modeling Analysis (EMA)

`EMA_climate_inequality.py` integrates the model with **EMA Workbench** to support:

### ✔ Uncertainty Exploration  
Vary climate sensitivity, inequality elasticities, elite growth, adaptation, trust erosion, policy effectiveness, etc.

### ✔ Robustness Testing  
Identify policy combinations resilient across thousands of plausible futures.

### ✔ Scenario Discovery  
Use PRIM to identify futures associated with:  
- Trust collapse  
- Radicalization tipping points  
- Climate-driven instability  
- Fiscal crises  
- Polycrisis patterns  

### ✔ Tradeoff & Pattern Analysis  
Parallel coordinates



