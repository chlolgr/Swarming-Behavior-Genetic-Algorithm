import pickle
from Evolution import Evolution
from Predator import Predator
from Prey import Prey
from Grid import Grid
import time
import numpy as np


def open_evolution(name):
	evolution = pickle.load(open('evolutions/'+name+'.pkl','rb'))
	return evolution


#========================== CONTINUE EVOLUTION ==========================#

def continue_evolution(evolution,generations,name):
	print('Continuing evolution')
	for i in range(generations):
		print('Running generation',i+1,'out of',generations)
		evolution.run_tests()
		ranked_predators = evolution.rank_predators()
		ranked_preys = evolution.rank_preys()
		evolution.update_pools(ranked_predators,ranked_preys)
		evolution.save(name)
		time.sleep(100)


#========================== VISUALIZE A PREDATOR ==========================#

def visualize(evolution,n_predator,n_prey):
	predator = evolution.predator_pool[n_predator]
	prey = evolution.prey_pool[n_prey]
	preys = [Prey(predator.grid,brain=prey.brain) for i in range(50)]
	predator.grid.visualize()
	print('There are',predator.grid.n_preys(),'preys left alive.\n')

def see_fitnesses(evolution):
	print(evolution.generation)
	ranked_predators = evolution.rank_predators()
	print(evolution.get_fitnesses(ranked_predators),'\n')


#========================== MAKE NEW EVOLUTION ==========================#

def make_new_evolution(name):

	print('Making new evolution')

	evolution = Evolution()
	evolution.run_tests()
	ranked_preys = evolution.rank_preys()
	ranked_predators = evolution.rank_predators()
	evolution.update_pools(ranked_predators,ranked_preys)
	evolution.save(name)
	
	return evolution


#========================== SCRIPT ==========================#

#name = ''
#evolution = open_evolution(name)
#print(evolution.generation)
#continue_evolution(evolution,1000,name)
#visualize(evolution,0,4)
#evolution.plot_fitness_per_generation(smoother=5)

