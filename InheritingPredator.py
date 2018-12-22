from sklearn.linear_model import Perceptron
import random
import numpy as np
import time
from Prey import Prey
from Grid import Grid
import math
import pickle
import matplotlib.pyplot as plt
from utils import *
import sys
from Animal import Animal



class Predator(Animal):
	

	def __init__(self,grid,coefs=None,brain=None,confusion=True,mask=np.ones(13)):

		Animal.__init__(self,grid)
		self.agility = 6
		self.speed = 3
		self.sight = 200

		self.confusion = confusion

		if brain: self.brain = brain
		else:
			self.brain = Perceptron(max_iter=1)
			self.initialize_brain(coefs)
		# view is the input vector for the predator's perceptron
		# (12 sections + bias)
		self.view = np.zeros(13)
		# mask is used to impair the predator's vision
		# if it's a vector of ones, all information goes through
		# we can set some values to 0 to blind certain angles
		self.mask = mask
		# wait is for the digesting period
		# (the predator has to wait 10 iterations between 2 attacks)
		self.wait = 0

	
	def copy(self,grid=None):
		if grid: copy = Predator(grid)
		else: copy = Predator(self.grid)
		copy.brain = self.brain
		return copy

	
	def set_mask(self,mask):
		self.mask = mask


	def initialize_brain(self,coefs=None):
		# make random input and target vectors
		x = np.random.choice([0,1],(4,13))
		y = np.arange(4)
		# fit it (1 iteration) to initialize all parameters
		self.brain.fit(x,y)
		# can used pre-defined coefs (for inheritance for instance)
		if coefs: self.brain.coefs_ = coefs
		# otherwise, weights attributed randomly
		else: self.brain.coef_ = np.random.rand(4,13)*2-1


	def pick_move(self):

		super().pick_move()

		# update fitness at each iteration
		self.fitness += (50-self.grid.n_preys())
		# decrement the waiting period if it is activated
		if self.wait>0: self.wait-=1


	# other animals = list of other animals on the grid
	def look_around(self):
		other_animals = [a for a in self.grid.inhabitants if a!=self]
		view = np.zeros(13)
		for animal in other_animals:
			see = self.scan(animal)
			if see: view[see]=1
			if self.can_attack(animal) and self.wait==0:
				# activate waiting period
				self.wait=10
				r = self.attack(animal)
				if self.confusion: return r
				
		# update bias
		view[-1] = 1
		self.view = view

		
	def attack(self,animal,printing=False):
		if self.confusion:
			# counting the number of preys the predator can see
			n_visible = 1
			for animal_ in self.grid.inhabitants:
				# calculate distance between target and other preys
				if isinstance(animal_,Prey) and animal_!=animal:
					coordinates = (animal.pos_x,animal.pos_y,animal_.pos_x,animal_.pos_y)
					d = distance(*coordinates,self.grid.width,self.grid.height)
					# if distance<30 units, it adds to confusion
					if self.scan(animal_) and d<=30: n_visible+=1
			if printing: print(n_visible,'prey(s) in sight')
			if np.random.rand() < 1/n_visible: 
				animal.die()
				if printing: 
					print('Attack: successful')
					print(self.grid.n_preys(),'preys left\n')
				return n_visible,True
			else: 
				if printing: print('Attack: unsuccessful\n')
				return n_visible,False
		else: animal.die()

	
	def can_attack(self,animal):
		# calculate distance
		d = distance(self.pos_x,self.pos_y,animal.pos_x,animal.pos_y,self.grid.width,self.grid.height)
		if d > 5: return False

		# calculate angle -- see if the prey in in field of vision

		angle = math.radians(self.orientation)

		ox,oy = self.pos_x,self.pos_y
		px,py = self.pos_x+1,self.pos_y
		new_x = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
		new_y = oy + math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)

		da = np.array([animal.pos_x - ox, animal.pos_y - oy])
		do = np.array([new_x - ox, new_y - oy])

		cosangle = np.dot(da,do) / (np.linalg.norm(da) * np.linalg.norm(do))
		aangle = math.degrees(math.acos(cosangle))
		det = do[0]*da[1] - do[1]*da[0]

		return self.mask[angle_to_section(aangle,det)]==1
			

	def scan(self,animal):
		if self.can_see(animal):

			angle = math.radians(self.orientation)

			ox,oy = self.pos_x,self.pos_y
			px,py = self.pos_x+1,self.pos_y
			new_x = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
			new_y = oy + math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)

			da = np.array([animal.pos_x - ox, animal.pos_y - oy])
			do = np.array([new_x - ox, new_y - oy])

			cosangle = np.dot(da,do) / (np.linalg.norm(da) * np.linalg.norm(do))
			aangle = math.degrees(math.acos(cosangle))
			det = do[0]*da[1] - do[1]*da[0]

			if isinstance(animal,Prey): 
				section = angle_to_section(aangle,det)
				if self.mask[section]==1: return section
			return False


	def see_brain(self):
		weights = np.concatenate((np.arange(-6,0,1).astype(str),np.arange(1,7,1).astype(str)),axis=0)
		weights = np.concatenate((weights,np.array(['bias'])),axis=0)
		x = np.arange(13)
		y_still = self.brain.coef_[0]
		y_right = self.brain.coef_[1]
		y_left = self.brain.coef_[2]
		y_forward = self.brain.coef_[3]
		
		plt.title('Weights in Deciding to Stay Still')
		plt.xticks(x,weights)
		plt.xlabel('view section (left to right)')
		plt.ylabel('weight')
		p1 = plt.bar(x,y_still,alpha=1)
		plt.show()

		plt.title('Weights in Deciding to Turn Right')
		plt.xticks(x,weights)
		plt.xlabel('view section (left to right)')
		plt.ylabel('weight')
		p1 = plt.bar(x,y_right,alpha=1)
		plt.show()
		
		plt.title('Weights in Deciding to Turn Left')
		plt.xticks(x,weights)
		plt.xlabel('view section (left to right)')
		plt.ylabel('weight')
		p1 = plt.bar(x,y_left,alpha=1)
		plt.show()

		plt.title('Weights in Deciding to Go Forward')
		plt.xticks(x,weights)
		plt.xlabel('view section (left to right)')
		plt.ylabel('weight')
		p1 = plt.bar(x,y_forward,alpha=1)
		plt.show()
		


	def save(self,name):
		pickle.dump(self,open('predators/'+name+'.pkl','wb'))
