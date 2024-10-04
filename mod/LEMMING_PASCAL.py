"""
Data-driven modeling in Python, winter term 2023/2024
Project: The Norway Lemming - Hippolyte PASCAL
"""

import sys
#print(sys.path)
sys.path.append('C:/Users/hippo_kq2e550/OneDrive/Desktop/KIT/Datengetriebene Modellierung mit Python/DMP_Project_WS2324')
import numpy as np
import matplotlib.pyplot as plt
import mod.my_plotter as mp
from mod.helper_functions import return_b, return_d, return_f
from mod.helper_functions import get_weather, fit_d_sd, get_d, fit_f_sd_T
from mod.lemming import Lemming
from datetime import datetime, timedelta


# Initialization 
N_INIT = 500 # Initial population size 
SIM_TIME = 3*365 # Simulation time in days
SIM_TIME_ENVI = 303 # Start date from 1st January to 31st October
dt = 1 # Unit of time = days

N = np.zeros(SIM_TIME) # Array to store population size
N[0] = N_INIT

for j in range(1, SIM_TIME): # days
    b = return_b()
    d = return_d() # Envi parameters are made optional, here is the default case
    f = return_f(N[j-1]) 
    dN_dt = (b-d-f)*N[j-1] # Rate of change
    N[j] = N[j-1] + dN_dt * dt # New population size 

L = [Lemming() for _ in range(N_INIT)]
population_N = np.zeros(SIM_TIME)

# Run the simulation
for day in range(SIM_TIME):
    new_lemmings = []

    # Each lemming live for a day + check for new birth
    for lemming in L:
        alive, babies = lemming.live_a_day(len(L))

        if alive:
            new_lemmings.append(lemming)
        
        # Add new baby lemmings to the population
        new_lemmings.extend([Lemming() for _ in range(babies)])
    
    L = new_lemmings  # Update the population with the surviving + baby lemmings
    population_N[day] = len(L)  # Store the current population size

plt.close('all')
# Creation of the figure + axis
mp.init_plot()
fig, ax = plt.subplots()

# Plot the evolution of the population over 3 years
ax.plot(range(SIM_TIME), N, color = mp.lightgreen, label='Lemming Population')

# Set labels + title + legend
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population Size')
ax.set_title('Evolution of Lemming Population Over Time (Simple Approach)', fontweight='bold')
ax.legend()

# Save figure with 300 DPI in exp/
plt.savefig('./exp/part1.png', dpi=300)

# Plot the evolution of the population over 3 years
ax.plot(range(SIM_TIME), population_N, color = mp.purple, label='Lemming Population Throwing Dices')

# Set labels + title + legend
ax.set_title('Evolution of Lemming Population Over Time (Throwing Dices)', fontweight='bold')
ax.legend()

# Save figure with 300 DPI in exp/
plt.savefig('./exp/part2.png', dpi=300)

# While maintaining a fairly similar shape to part 1, the result of part 2 shows fluctuations due to individual variations.
# allowed by instances of lemmings. Each lemming provides an individual output (death, reproduction) based on a stochastic approach
# compared to the first part, which considers lemmings as a homogeneous group. Individual simulation allows for more
# nuanced interactions and survival chances, e.g. Eq. 10.6 due to the resetting of the food counter, which gives lemmings the chance to
# to survive longer

fit_d_sd()  # Fit the model and plot the data
fit_f_sd_T() 

# Same process with the environmental parameters took in account

N = np.zeros(SIM_TIME_ENVI) # Array to store population size
N[0] = N_INIT

start_date = "01-01"  # Starting from January 1st
for j in range(1, SIM_TIME_ENVI):  
    snow_depth, temp = get_weather(start_date, j-1)  

    b = return_b()  # Assuming birth rate is constant
    d = return_d(snow_depth)
    f = return_f(N[j-1], snow_depth, temp)
    
    dN_dt = (b - d - f) * N[j-1]  # Rate of change considering environmental effects
    N[j] = N[j-1] + dN_dt * dt  

L = [Lemming() for _ in range(N_INIT)]
population_N = np.zeros(SIM_TIME_ENVI)

# Run the simulation
for day in range(SIM_TIME_ENVI):
    new_lemmings = []

    for lemming in L:
        alive, babies = lemming.live_a_day(len(L))

        if alive:
            new_lemmings.append(lemming)
        
        new_lemmings.extend([Lemming() for _ in range(babies)])
    
    L = new_lemmings  
    population_N[day] = len(L)  

plt.close('all')
mp.init_plot()
fig, ax1 = plt.subplots()

# Plot the evolution of the population over 3 years
ax1.plot(range(SIM_TIME_ENVI), N, color = mp.lightgreen, label='Continuous')
ax1.plot(range(SIM_TIME_ENVI), population_N, color = mp.purple, label='Montecarlo')

# Set labels + title + legend
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Population Size')
ax1.set_title('Evolution of Lemming Population Over Time with Environemental Parameters', fontweight='bold')
ax1.legend()
plt.savefig('./exp/part3.png', dpi=300)




