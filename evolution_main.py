import pickle
from Evolution import Evolution
from Predator import Predator
from Prey import Prey
from Grid import Grid
import time
import numpy as np
import math


def open_evolution(name):
	evolution = pickle.load(open('evolutions/'+name+'.pkl','rb'))
	return evolution


#========================== CONTINUE EVOLUTION ==========================#

# evolution = Evolution object
# generations = number of generations to evolve
# name = name for saving
def continue_evolution(evolution,generations,name):
	print('Continuing evolution')
	for i in range(generations):
		print('Running generation',i+1,'/',generations)
		evolution.run_tests()
		ranked_predators = evolution.rank_predators()
		ranked_preys = evolution.rank_preys()
		evolution.update_pools(ranked_predators,ranked_preys)
		evolution.save(name)
		time.sleep(100)


#========================== VISUALIZE A PREDATOR ==========================#

# creates a visualization of the evolution
# n_predator = rank of the predator you want to use in the evolution (0 = best predator)
# n_prey = same thing but with preys
# prey_card = size of the initial swarm of preys
def visualize(evolution,n_predator,n_prey,prey_card):
	predator = evolution.predator_pool[n_predator]
	prey = evolution.prey_pool[n_prey]
	preys = [Prey(predator.grid,brain=prey.brain) for i in range(prey_card)]
	predator.grid.visualize()
	print('There are',predator.grid.n_preys(),'preys left alive.\n')

# see the list of all predator fitnesses at the current generation
def see_fitnesses(evolution):
	print(evolution.generation)
	ranked_predators = evolution.rank_predators()
	print(evolution.get_fitnesses(ranked_predators),'\n')


#========================== MAKE NEW EVOLUTION ==========================#

# name: for saving
# confusion: True -> activate predator confusion
# mask: predator's mask for view
def make_new_evolution(name,confusion,mask):

	print('Making new evolution')

	evolution = Evolution(confusion,mask)
	evolution.run_tests()
	ranked_preys = evolution.rank_preys()
	ranked_predators = evolution.rank_predators()
	evolution.update_pools(ranked_predators,ranked_preys)
	evolution.save(name)
	
	return evolution


#========================== MEASURE MASK EFFICIENCY ==========================#

# run n_tests tests to see how well the predator/prey is doing
# n_preys sets the initial size of the swarm
# confusion: True -> activate predator confusion
# prints the average number of preys left, average number of visible preys at attack, and success rate over n_tests tests
def measure_efficiency(evolution,n_preys,n_tests,confusion):
	total_preys,total_visible_preys,total_successes,attempts = list(),list(),list(),list()

	if confusion:
		for t in range(n_tests):
			grid = Grid(512,512)
			predator = evolution.predator_pool[0]
			Predator(grid,brain=predator.brain,mask=predator.mask,confusion=True)
			prey = evolution.prey_pool[0]
			for _ in range(n_preys): Prey(grid,brain=prey.brain)
			print("\nRunning test",t+1,"out of",n_tests)
			visible_preys,successes = list(),list()
			for _ in range(2000):
				for animal in grid.inhabitants: 
					result = animal.look_around()
					if result: 
						visible_preys.append(result[0])
						successes.append(result[1])
				for animal in grid.inhabitants: animal.pick_move()
			#print(grid.n_preys(),'preys left')
			#print('Average number of visible preys in attack:',np.mean(visible_preys))
			#print('Proportion of successes:',len(np.where(successes)[0])/len(successes))
			total_preys.append(grid.n_preys())
			total_visible_preys.append(np.mean(visible_preys))
			total_successes.append(len(np.where(successes)[0])/len(successes))
			attempts.append(len(successes))
		print('\nAverage number of preys left:',np.mean(total_preys))
		print('\nAverage number of visible preys in attacks:',np.mean(total_visible_preys))
		print('\nProportion of successes:',np.mean(total_successes))
		print('\nAverage number of attacks:',np.mean(attempts))

	else:
		preys_left = list()
		for t in range(n_tests):
			grid = Grid(512,512)
			predator = evolution.predator_pool[0]
			Predator(grid,brain=predator.brain,mask=predator.mask,confusion=False)
			prey = evolution.prey_pool[0]
			for _ in range(n_preys): Prey(grid,brain=prey.brain)
			print("\nRunning test",t+1,"out of",n_tests)
			for _ in range(2000):
				for animal in grid.inhabitants: animal.look_around()
				for animal in grid.inhabitants: animal.pick_move()
			preys_left.append(grid.n_preys())
		print('\nAverage number of preys left:',np.mean(preys_left))


#========================== SEE OUTPUT ==========================#

# animal = Predator or Prey object
# view = vector of the animal's view
# returns the move picked by the animal when presented with this view
def see_output(animal,view):
	predictions = animal.brain.decision_function(view.reshape(1,-1))
	predictions = predictions.flatten()
	predictions = np.array([1/(1+math.exp(-p)) for p in predictions])
	print(view.astype(int))
	actions = ['stay still','turn right','turn left','move forward']
	print(actions[np.argmax(predictions)])


#========================== SCRIPT ==========================#

name = 'card_10_basic_ter'
evolution = open_evolution(name)
for predator in evolution.predator_pool: predator.view_mode = 'boolean'
for predator in evolution.predator_pool: predator.mask = np.ones(13)
print(evolution.generation)
#continue_evolution(evolution,1000,name)
visualize(evolution,0,0,50)
#evolution.plot_fitness_per_generation(smoother=5)
#evolution.prey_pool[0].see_brain()
#measure_efficiency(evolution,50,10,True)
