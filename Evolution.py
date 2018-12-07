from Grid import Grid
from Prey import Prey
from Predator import Predator
import time
import numpy as np
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import sys


class Evolution():

	
	def __init__(self,predator_pool=None,prey_pool=None):

		self.card_predator_pool = 10
		self.card_prey_pool = 10
		self.n_preys = 50
		self.n_iters = 2000
		self.grid_size = (512,512)
		self.generation = 0
		self.generation_fitnesses = dict()
	
		if predator_pool: 
			if len(predator_pool)==self.card_predator_pool: self.predator_pool = predator_pool
			else: print("\nGiven predator pool not the expected size.\n")
		else: self.predator_pool = self.generate_random_predators(self.card_predator_pool)

		if prey_pool:
			if len(predator_pool)==self.card_prey_pool: self.prey_pool = prey_pool
			else: print("\nGiven prey pool not the expected size.\n")
		else: self.prey_pool = self.generate_random_preys(self.card_prey_pool)

	
	def generate_random_predators(self,n):
		return [Predator(Grid(self.grid_size[0],self.grid_size[1])) for i in range(n)]

	
	def generate_random_preys(self,n):
		return [Prey(Grid(self.grid_size[0],self.grid_size[1])) for i in range(n)]

	
	def run_tests(self):

		def test_one_on_one(predator,prey,return_dict,num_test):
			grid = Grid(512,512)
			test_predator = Predator(grid,brain=predator.brain)
			swarm = [Prey(grid,brain=prey.brain) for _ in range(self.n_preys)]
			for i in range(self.n_iters): self.next_timeStep(grid)
			return_dict[num_test] = {'predator':test_predator.fitness, 'prey':swarm[0].fitness}

		processes = list()
		manager = mp.Manager()
		return_dict = manager.dict()

		for num_predator,predator in enumerate(self.predator_pool):
			num_predator_str = str(num_predator)
			if len(num_predator_str)==1: num_predator_str = '0'+num_predator_str
			for num_prey,prey in enumerate(self.prey_pool):
				num_prey_str = str(num_prey)
				if len(num_prey_str)==1: num_prey_str = '0'+num_prey_str
				num_test = num_predator_str+num_prey_str
				processes.append(mp.Process(target=test_one_on_one, args=(predator,prey,return_dict,num_test)))
		for process in processes: process.start()
		for process in processes: process.join()

		# decode return_dict
		predator_fitnesses = dict()
		prey_fitnesses = dict()
		for num_test,fitness_tuple in return_dict.items():
			num_predator,num_prey = int(num_test[:2]),int(num_test[3:])
			if num_predator in predator_fitnesses: predator_fitnesses[num_predator] += fitness_tuple['predator']
			else: predator_fitnesses[num_predator] = fitness_tuple['predator']
			if num_prey in prey_fitnesses: prey_fitnesses[num_prey] += fitness_tuple['prey']
			else: prey_fitnesses[num_prey] = fitness_tuple['prey']
		
		for num_predator,predator in enumerate(self.predator_pool):
			predator.fitness = 1.*predator_fitnesses[num_predator] / self.card_prey_pool
		for num_prey,prey in enumerate(self.prey_pool):
			prey.fitness = 1.*prey_fitnesses[num_prey] / self.card_predator_pool
		
	
	def rank_predators(self):

		fitnesses = np.array([predator.fitness for predator in self.predator_pool])
		reverse_sort = np.argsort(fitnesses)
		ranked_predators = []
		for i in range(len(reverse_sort)):
			index = reverse_sort[-(i+1)]
			ranked_predators.append((self.predator_pool[index], fitnesses[index]))

		if not self.generation in self.generation_fitnesses: self.generation_fitnesses[self.generation] = dict()
		self.generation_fitnesses[self.generation]['predators'] = [sum([predator.fitness for predator in self.predator_pool])]
		self.generation_fitnesses[self.generation]['predators'].append(ranked_predators[0][1])
		return ranked_predators

	
	def rank_preys(self):
		
		fitnesses = np.array([prey.fitness for prey in self.prey_pool])
		reverse_sort = np.argsort(fitnesses)
		ranked_preys = []
		for i in range(len(reverse_sort)):
			index = reverse_sort[-(i+1)]
			ranked_preys.append((self.prey_pool[index], fitnesses[index]))
			
		if not self.generation in self.generation_fitnesses: self.generation_fitnesses[self.generation] = dict()
		self.generation_fitnesses[self.generation]['preys'] = [sum([prey.fitness for prey in self.prey_pool])]
		self.generation_fitnesses[self.generation]['preys'].append(ranked_preys[0][1])
		return ranked_preys


	def make_up_next_predator_generation(self,ranked_predators):
	
		p = [ranked_predators[i][0].brain for i in range(3)]
		p+= [ranked_predators[6][0].brain]
		
		kids = []

		for i in range(self.card_predator_pool-2):

			# initialize weight matrices for the kid's MLP
			w1 = np.zeros(p[0].coefs_[0].shape)
			w2 = np.zeros(p[0].coefs_[1].shape)
			
			p1,p2 = np.random.choice(p,2,replace=False)
			
			for row in range(w1.shape[0]):
				for col in range(w1.shape[1]):
					r = np.random.random()
					if r<.45: w1[row,col] = p1.coefs_[0][row,col]
					elif r<.9: w1[row,col] = p2.coefs_[0][row,col]
					else: w1[row,col] = np.random.random()
			
			for row in range(w2.shape[0]):
				for col in range(w2.shape[1]):
					r = np.random.random()
					if r<.45: w2[row,col] = p1.coefs_[1][row,col]
					elif r<.9: w2[row,col] = p2.coefs_[1][row,col]
					else: w2[row,col] = np.random.random()


			kid = Predator(Grid(self.grid_size[0],self.grid_size[1]))
			kid.brain.coefs_ = [w1,w2]
			kids.append(kid)

		return kids,p[0],p[1]


	def make_up_next_prey_generation(self,ranked_preys):
	
		p = [ranked_preys[i][0].brain for i in range(3)]
		p+= [ranked_preys[6][0].brain]
		
		kids = []

		for i in range(self.card_prey_pool-2):

			# initialize weight matrices for the kid's MLP
			w1 = np.zeros(p[0].coefs_[0].shape)
			w2 = np.zeros(p[0].coefs_[1].shape)
			
			p1,p2 = np.random.choice(p,2,replace=False)
			
			for row in range(w1.shape[0]):
				for col in range(w1.shape[1]):
					r = np.random.random()
					if r<.45: w1[row,col] = p1.coefs_[0][row,col]
					elif r<.9: w1[row,col] = p2.coefs_[0][row,col]
					else: w1[row,col] = np.random.random()
			
			for row in range(w2.shape[0]):
				for col in range(w2.shape[1]):
					r = np.random.random()
					if r<.45: w2[row,col] = p1.coefs_[1][row,col]
					elif r<.9: w2[row,col] = p2.coefs_[1][row,col]
					else: w2[row,col] = np.random.random()


			kid = Prey(Grid(self.grid_size[0],self.grid_size[1]))
			kid.brain.coefs_ = [w1,w2]
			kids.append(kid)

		return kids,p[0],p[1]


	def update_predator_pool(self,ranked_predators):
		kids,p1,p2 = self.make_up_next_predator_generation(ranked_predators)
		self.predator_pool = [Predator(Grid(self.grid_size[0],self.grid_size[1]),brain=p1), 
			Predator(Grid(self.grid_size[0],self.grid_size[1]),brain=p2)]
		for kid in kids: self.predator_pool.append(kid)


	def update_prey_pool(self,ranked_preys):
		kids,p1,p2 = self.make_up_next_prey_generation(ranked_preys)
		self.prey_pool = [Prey(Grid(self.grid_size[0],self.grid_size[1]),brain=p1),
			Prey(Grid(self.grid_size[0],self.grid_size[1]),brain=p2)]
		for kid in kids: self.prey_pool.append(kid)

	
	def update_pools(self,ranked_predators,ranked_preys):
		self.update_predator_pool(ranked_predators)
		self.update_prey_pool(ranked_preys)
		self.generation+=1

		
	def next_timeStep(self,grid):
		for animal in grid.inhabitants: animal.look_around()
		for animal in grid.inhabitants: animal.pick_move()

	
	def get_predator_fitnesses(self,ranked_predators):
		return [fitness for predator,fitness in ranked_predators]

	
	def get_prey_fitnesses(self,ranked_preys):
		return [fitness for prey,fitness in ranked_preys]


	def plot_fitness_per_generation(self,smoother=2):

		x,y_predator_avg,y_predator_max,y_prey_avg,y_prey_max = [],[0]*smoother,[0]*smoother,[0]*smoother,[0]*smoother
		for generation,fitnesses in self.generation_fitnesses.items():
			x.append(generation)
			y_predator_avg.append(fitnesses['predators'][0] / self.card_predator_pool)
			y_predator_max.append(fitnesses['predators'][1])
			y_prey_avg.append(fitnesses['preys'][0] / self.card_prey_pool)
			y_prey_max.append(fitnesses['preys'][1])


		x = np.array(x)+1	
		
		if smoother>0:	

			for i in range(len(y_predator_avg)-smoother): y_predator_avg[i] = np.mean(y_predator_avg[i:i+smoother])
			for i in range(len(y_predator_max)-smoother): y_predator_max[i] = np.mean(y_predator_max[i:i+smoother])
			for i in range(len(y_prey_avg)-smoother): y_prey_avg[i] = np.mean(y_prey_avg[i:i+smoother])
			for i in range(len(y_prey_max)-smoother): y_prey_max[i] = np.mean(y_prey_max[i:i+smoother])

			y_predator_avg = y_predator_avg[:-smoother]
			y_predator_max = y_predator_max[:-smoother]
			y_prey_avg = y_prey_avg[:-smoother]
			y_prey_max = y_prey_max[:-smoother]

		plt.plot(x,y_predator_avg,label='average predator')
		plt.plot(x,y_predator_max,label='best predator')
		plt.plot(x,y_prey_avg,label='average prey')
		plt.plot(x,y_prey_max,label='best prey')
		plt.legend()
		plt.xlabel('generation')
		plt.ylabel('fitness')
		plt.xticks(np.arange(1,len(x)+1,step=int(len(x)/10)-1))
		plt.show()


	def save(self,name):
		pickle.dump(self,open('evolutions/'+name+'.pkl','wb'))