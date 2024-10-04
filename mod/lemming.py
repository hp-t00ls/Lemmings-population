"""
Data-driven modeling in Python, winter term 2023/2024
Project: The Norway Lemming - Hippolyte PASCAL
"""

import random

LITTERS_PER_YEAR = 2.5
NEWBORNS_PER_LITTER = 7
NEWBORNS_PER_YEAR = LITTERS_PER_YEAR * NEWBORNS_PER_LITTER
LIFESPAN_IN_DAYS = 2*365
F = 75 # Number of lemmings that can eat per day
D = 5 # Number of days to reach starvation

class Lemming:
    def __init__(self):
        self.age = 0  # The age of this lemming in days
        self.toughness = 5  # The number of days this lemming can survive without eating
        self.last_eaten = 0  # The number of consecutive days this lemming has gone without food
        self.name = self.assign_random_name()  # Assign a random name from the list.

    @staticmethod
    def assign_random_name(): 
        # Try & catch in case the file is not found + default name
        try:
            with open('./data/rand_names.txt', 'r') as file:
                names = [line.strip() for line in file.readlines()]
            return random.choice(names)
        except FileNotFoundError:
            return "Bob"  
   
    """ def check_dead(self):
        # Probability of death for a given day taking in account toughness + # of days without food
        starv_risk = min(self.last_eaten / self.toughness, 1) # Up to one 
        death_prob = (1/ LIFESPAN_IN_DAYS) * (1 + starv_risk) # Increases naturally with starvation risk 
        return random.random() < death_prob #  Return True if the lemming dies  """

    def check_dead(self):
        # Probability of death for a given day 
        death_prob = (1/ LIFESPAN_IN_DAYS)  # Increases naturally with starvation risk 
        return random.random() < death_prob #  Return True if the lemming dies  

    def check_reproduce(self):
        # Half the population is female + all give birth 
        newborns_litter = random.randint(1, 7)
        if random.random() < (( 2.5  / 2)/ 365):
            return newborns_litter # If reproduction occurs # of babies bewteen 1-7 per litter
        return 0

    def check_food(self, N, F=75): 
        # With F still the # of lemmings that can eat per day
        # Probability of finding food based on the ratio of available food for population
        if N <= F:
            return True  # No ratio needed...
        else:
            food_prob = F / N # The bigger is the population => smallest chance to find food
            return random.random() < food_prob # If True, lemming finds food on that day
        
    def live_a_day(self,N):
        # Increment the lemming's age + 1 day
        self.age += 1

        # Check if the lemming finds food
        if self.check_food(N):
            self.last_eaten = 0  # Reset the days without food counter
        else:
            self.last_eaten += 1  # Increment the days without food counter

        # Check for reproduction
        reproduce = self.check_reproduce()

        # Check if the lemming is still alive considering its current state
        alive = not self.check_dead() and self.last_eaten <= self.toughness

        return alive, reproduce 