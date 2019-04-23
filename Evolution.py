from Grid import Grid
from Prey import Prey
from Predator import Predator
import time
import numpy as np
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import sys
from utils import masks


class Evolution():

		
	"""
	Evolutions contain all predators and preys.
	They are set with a value for predator confusion and a mask.
	You can activate the evolve_mask parameter if you want to optimize the predators' field of vision.

	"""

	
	def __init__(self,confusion,mask,predator_pool=None,prey_pool=None,evolve_mask=False,view_mode='boolean'):

		# we choose whether to activate predator confusion
		# and set a mask for the predators' vision
		self.confusion = confusion
		self.mask = mask
		self.evolve_mask = evolve_mask
		self.view_mode = view_mode

		# numbers of individuals in the pool (at each generation)
		self.card_predator_pool = 10
		self.card_prey_pool = 10
		# number of copies of a prey in a simulation
		self.n_preys = 50
		# number of timesteps in a simulation
		self.n_iters = 2000
		self.grid_size = (512,512)
		self.generation = 0
		# remember the evolution of fitnesses along time
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
		return [Predator(Grid(*self.grid_size),confusion=self.confusion,mask=self.mask,view_mode=self.view_mode) for _ in range(n)]

	
	def generate_random_preys(self,n):
		return [Prey(Grid(*self.grid_size)) for _ in range(n)]

	
	"""
	The run_tests() method is the key part of evolution.
	It tries every possible combination of its preys and predators (for 10 predator and 10 preys, it runs 10*10 = 100 simulations). To minimize the calculation time, it uses multiprocessing.
	It's possible to print out the progress of the simulations with the pr dict, but that's only necessary if you're impatient. I don't recommend activating that. If you want to, all you need to do is remove all the # in the test_one_on_one() function.
	
	To keep track of the results that come in a random order because of the multiprocessing, run_tests() has a manager.dict caled return_dict into which test_one_on_one() writes the results (fitnesses) at the end of every simulation. The keys to that dict are the index of the prey and the predator put together into a string. 
	-> simulation of predator n°1 against prey n°3 will be under the key '0103'.
	/!\ NOTE THAT THIS ONLY WORKS FOR PREY AND PREDATOR POOLS OF UNDER 100 INDIVIDUALS.

	"""	


	def run_tests(self):

		def test_one_on_one(predator,prey,return_dict,num_test,pr):
			grid = Grid(512,512)
			# copy the predator
			test_predator = Predator(grid,brain=predator.brain,confusion=self.confusion,mask=self.mask)
			# copy and duplicate the prey (n_preys instances)
			swarm = [Prey(grid,brain=prey.brain) for _ in range(self.n_preys)]

			for i in range(self.n_iters): 
				# next 4 lines to print the % of progress in the simulation
				#if i%(self.n_iters/10)==0: 
					#if pr[int(i/(self.n_iters/10))]<self.card_prey_pool*self.card_predator_pool-10:
					#	pr[int(i/(self.n_iters/10))]+=1
					#else:
					#	t = np.round((time.time()-start)/60,2)
					#	t = str(t).split('.')
					#	t = [t[0],str(int(int(t[1])*.6))]
					#	print(int(i*100/self.n_iters),'%\t'+t[0]+'m'+t[1]+'s')
					#	pr[int(i/(self.n_iters/10))] = -self.card_prey_pool*self.card_predator_pool

				self.next_timeStep(grid)
			return_dict[num_test] = {'predator':test_predator.fitness, 'prey':swarm[0].fitness}

		processes = list()
		manager = mp.Manager()
		# dict to store the results through multiprocessing
		# 	(prey,predator) -> their fitnesses at the end of their simulation
		return_dict = manager.dict()
		# list used to print the % of progress, not really important
		pr = manager.dict()
		#for i in range(10): pr[i]=0

		for num_predator,predator in enumerate(self.predator_pool):
			# turn predator index into a str for the keys in our manager dict
			num_predator_str = str(num_predator)
			if len(num_predator_str)==1: num_predator_str = '0'+num_predator_str
			for num_prey,prey in enumerate(self.prey_pool):
				# same str instead of index for preys
				num_prey_str = str(num_prey)
				if len(num_prey_str)==1: num_prey_str = '0'+num_prey_str
				# concatenate the lists to make the key for a pair
				num_test = num_predator_str+num_prey_str
				# next three lines are the multiprocessing to save years off our lives
				processes.append(mp.Process(target=test_one_on_one, args=(predator,prey,return_dict,num_test,pr)))
		start = time.time()		
		for process in processes: process.start()
		for process in processes: process.join()
		t = np.round((time.time()-start)/60,2)
		t = str(t).split('.')
		t = [t[0],str(int(int(t[1])*.6))]
		print('100%\t'+t[0]+'m'+t[1]+'s')

		# decode our manager dict
		predator_fitnesses = dict()
		prey_fitnesses = dict()
		for num_test,fitness_tuple in return_dict.items():
			num_predator,num_prey = int(num_test[:3]),int(num_test[3:])
			if num_predator in predator_fitnesses: predator_fitnesses[num_predator] += fitness_tuple['predator']
			else: predator_fitnesses[num_predator] = fitness_tuple['predator']
			if num_prey in prey_fitnesses: prey_fitnesses[num_prey] += fitness_tuple['prey']
			else: prey_fitnesses[num_prey] = fitness_tuple['prey']
		for num_predator,predator in enumerate(self.predator_pool):
			predator.fitness = 1.*predator_fitnesses[num_predator] / self.card_prey_pool
		for num_prey,prey in enumerate(self.prey_pool):
			prey.fitness = 1.*prey_fitnesses[num_prey] / self.card_predator_pool
		
	
	def rank_predators(self):

		# This is pretty straightforward.
		#	-> calculate average and top fitnesses for predators and store them into generation_fitnesses

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
		
		# same thing as rank_predators but with preys

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
	
		# select top 3 predators for reproduction
		p = [ranked_predators[i][0] for i in range(3)]
		# adding predator n°5 for diversity
		p+= [ranked_predators[4][0]]
		
		kids = []

		for i in range(self.card_predator_pool-2):

			# initialize weight matrix for the kid's perceptron
			w = np.zeros(p[0].brain.coef_.shape)
			
			# pick the parents randomly
			p1,p2 = np.random.choice(p,2,replace=False)
			
			# give the kid its parents' genes, +10% mutation
			for row in range(w.shape[0]):
				for col in range(w.shape[1]):
					r = np.random.random()
					if r<.45: w[row,col] = p1.brain.coef_[row,col]
					elif r<.9: w[row,col] = p2.brain.coef_[row,col]
					else: w[row,col] = np.random.random()*2-1

			kid_mask = self.mask.copy()
			if self.evolve_mask:
				r = np.random.random()
				if r<.45: kid_mask = p1.mask
				elif r<.9: kid_mask = p2.mask
				else: kid_mask = masks[np.random.choice(masks.shape[0])]
				

			kid = Predator(Grid(*self.grid_size),confusion=self.confusion,mask=kid_mask,view_mode=self.view_mode)
			kid.brain.coef_ = w
			kids.append(kid)

		return kids,p[0].brain,p[1].brain


	def make_up_next_prey_generation(self,ranked_preys):
	
		# same syntax as make_up_next_predator_generations, but with preys

		p = [ranked_preys[i][0].brain for i in range(3)]
		p+= [ranked_preys[4][0].brain]
		
		kids = []

		for i in range(self.card_prey_pool-2):

			w = np.zeros(p[0].coef_.shape)
			
			p1,p2 = np.random.choice(p,2,replace=False)
			
			for row in range(w.shape[0]):
				for col in range(w.shape[1]):
					r = np.random.random()
					if r<.45: w[row,col] = p1.coef_[row,col]
					elif r<.9: w[row,col] = p2.coef_[row,col]
					else: w[row,col] = np.random.random()*2-1

			kid = Prey(Grid(*self.grid_size))
			kid.brain.coef_ = w
			kids.append(kid)

		return kids,p[0],p[1]


	def update_predator_pool(self,ranked_predators):
		kids,p1,p2 = self.make_up_next_predator_generation(ranked_predators)
		# keep first 2 predators for next generation to put them up against their children
		self.predator_pool = list()
		self.predator_pool.append(Predator(Grid(*self.grid_size),confusion=self.confusion,mask=self.mask,brain=p1)) 
		self.predator_pool.append(Predator(Grid(*self.grid_size),confusion=self.confusion,mask=self.mask,brain=p2))
		for kid in kids: self.predator_pool.append(kid)


	def update_prey_pool(self,ranked_preys):
		# same syntax as update_predator_pool but with preys
		kids,p1,p2 = self.make_up_next_prey_generation(ranked_preys)
		self.prey_pool = [Prey(Grid(*self.grid_size),brain=p1),
			Prey(Grid(*self.grid_size),brain=p2)]
		for kid in kids: self.prey_pool.append(kid)

	
	def update_pools(self,ranked_predators,ranked_preys):
		self.update_predator_pool(ranked_predators)
		self.update_prey_pool(ranked_preys)
		# This is where the generation is incremented, be careful not to call the update functions on their own.
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
