"""
Data-driven modeling in Python, winter term 2023/2024
Project: The Norway Lemming - Hippolyte PASCAL
"""

import numpy as np
import matplotlib.pyplot as plt
import mod.my_plotter as mp
from mod.helper_functions import return_b, return_d, return_f
from mod.helper_functions import get_weather, fit_d_sd, get_d, fit_f_sd_T, get_f
from mod.lemming import Lemming
from datetime import datetime, timedelta


# Initialization 
N_INIT = 500 # Initial population size 
SIM_TIME_1 = 3*365 # Simulation time in days
SIM_TIME_2 = 303 # From 1st of January until the 31st of October (environmental parameters)
dt = 1 # Unit of time = days

N = np.zeros(SIM_TIME_1) # Array to store population size
N[0] = N_INIT

for j in range(1, SIM_TIME_1): # days
    b = return_b() 
    d = return_d() # Environemental arguments are optional, here we are in the default case... 
    f = return_f(N[j-1]) 
    dN_dt = (b-d-f)*N[j-1] # Rate of change 
    N[j] = N[j-1] + dN_dt * dt # New population size  

L = [Lemming() for _ in range(N_INIT)]
population_N = np.zeros(SIM_TIME_1)

# Run the simulation
for day in range(SIM_TIME_1):
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
ax.plot(range(SIM_TIME_1), N, color = mp.lightgreen, label='Lemming Population')

# Set labels + title + legend
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population Size')
ax.set_title('Evolution of Lemming Population Over Time (Simple Approach)')
ax.legend()

# Save figure with 300 DPI in exp/
plt.savefig('./exp/part1.png', dpi=300)

# Plot the evolution of the population over 3 years
ax.plot(range(SIM_TIME_1), population_N, color = mp.purple, label='Lemming Population Throwing Dices')

# Set labels + title + legend
ax.set_title('Evolution of Lemming Population Over Time (Throwing Dices)')
ax.legend()

plt.savefig('./exp/part2.png', dpi=300)

# While maintaining a fairly similar shape to part 1, the result of part 2 shows fluctuations due to individual variations.
# allowed by instances of lemmings. Each lemming provides an individual output (death, reproduction) based on a stochastic approach
# compared to the first part, which considers lemmings as a homogeneous group. Individual simulation allows for more
# nuanced interactions and survival chances, e.g. Eq. 10.6 due to the resetting of the food counter, which gives lemmings the chance to
# to survive longer

fit_d_sd()  
fit_f_sd_T() 
death_prob = get_d(None)  # Get the death probability for a given snow depth

# Same process but this time using environnmental parameters

N = np.zeros(SIM_TIME_2) # Array to store population size
N[0] = N_INIT

start_date = "01-01"  # Starting from January the 1st

for j in range(1, SIM_TIME_2):  # days
    snow_depth, temp = get_weather(start_date, j-1)  # Going through each day and associated snow depth + temp

    b = return_b()  # Assuming birth rate not depending on environement 
    d = return_d(snow_depth)
    f = return_f(N[j-1], snow_depth, temp)
    
    dN_dt = (b - d - f) * N[j-1]  # Rate of change considering environmental effects
    N[j] = N[j-1] + dN_dt * dt  # New population size 

L = [Lemming() for _ in range(N_INIT)]
population_N = np.zeros(SIM_TIME_2)

# Run the simulation
for day in range(SIM_TIME_2):
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
mp.init_plot()
fig, ax1 = plt.subplots()

# Plot the evolution of the population over 3 years
ax1.plot(range(SIM_TIME_2), N, color = mp.lightgreen, label='Lemming Population')

# Plot the evolution of the population over 3 years
ax1.plot(range(SIM_TIME_2), population_N, color = mp.purple, label='Lemming Population Throwing Dices')

ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Population Size')
ax1.set_title('Evolution of Lemming Population Over Time (Environnment)')
ax1.legend()

plt.savefig('./exp/part3.png', dpi=300)
    





