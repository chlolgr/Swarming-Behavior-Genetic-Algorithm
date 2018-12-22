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



class Predator():

	"""
	Predator are created with a Grid object associated to them.

	You can initialize their brain either with an already trained perceptron, or the weights for the perceptron.

	You can set predator confusion to True or False, and give them a mask for their field of vision.
	A mask is an np.array of shape (13,), and the default mask is np.ones(13).

	How masks work:
	-> Each digit tells us whether the corresponding section is activated. The sections are listed from left to right, and represent 15° each. The last digit in the array is always set to 1 as it represents the bias.
	-> mask [0,0,0,1,1,1,1,1,1,0,0,0,1] means the predator sees at an angle of 90°.
	-> mask np.ones(13) means the predator sees at an angle of 180° (widest angle possible).
	-> mask [0,0,0,0,0,0,0,0,0,0,0,0,1] means the predator is blind.
	

	"""
	

	def __init__(self,grid,coefs=None,brain=None,confusion=True,mask=np.ones(13),view_mode='boolean'):

		self.grid = grid
		self.grid.add_animal(self)

		self.confusion = confusion
		self.view_mode = view_mode

		if brain: self.brain = brain
		else:
			self.brain = Perceptron(max_iter=1)
			self.initialize_brain(coefs)
		# randomly place the predator on the grid
		self.pos_x = random.random()*self.grid.width
		self.pos_y = random.random()*self.grid.height
		self.orientation = int(random.random()*360)
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
		self.fitness = 0

	
	def copy(self,grid=None):
		if grid: copy = Predator(grid)
		else: copy = Predator(self.grid)
		copy.brain = self.brain
		return copy


	def set(self,x,y,o):
		self.pos_x = x
		self.pos_y = y
		self.orientation = o

	
	def set_mask(self,mask):
		self.mask = mask


	def initialize_brain(self,coefs=None):
		# make random input and target vectors
		x = np.random.choice([0,1],(4,13))
		y = np.arange(4)
		# fit it (1 iteration) to initialize all parameters
		self.brain.fit(x,y)
		# can used pre-defined coefs (for inheritance for instance)
		if coefs: self.brain.coef__ = coefs
		# otherwise, weights attributed randomly
		else: self.brain.coef_ = np.random.rand(4,13)*2-1
		

	def move(self,mode):

		if mode=='stay still': return

		elif mode=='turn right':
			self.orientation -=6
			self.orientation %= 360

		elif mode=='turn left':
			self.orientation +=6
			self.orientation %= 360
		
		angle = math.radians(self.orientation)
		new_x,new_y = new_position(self.pos_x,self.pos_y,3,angle)
		# normalizing the predator's position
		self.pos_x = new_x%self.grid.width
		self.pos_y = new_y%self.grid.height


	def pick_move(self):

		predictions = self.brain.decision_function(self.view.reshape(1,-1))
		predictions = predictions.flatten()
		# converting to a predict_proba format
		# (not necessary but easier to read in case we ever need to)
		predictions = np.array([1/(1+math.exp(-p)) for p in predictions])
		# pick most likely next move
		next_move = np.argmax(predictions)
		if next_move==0: self.move('stay still')
		elif next_move==1: self.move('turn right')
		elif next_move==2: self.move('turn left')
		elif next_move==3: self.move('move forward')
		else: print("\nCareful, orientation not recognized.\n")

		# update fitness at each iteration
		self.fitness += (50-self.grid.n_preys())
		# decrement the waiting period if it is activated
		if self.wait>0: self.wait-=1


	"""
	When the predator looks around, it lists all other animals and initiates its view to zeros.
	For each animal, if it is visible, the section is set to one (boolean view). We can toy with that by making each section the count of animals in it, which would give our predators a sense of density of preys in such and such directions.
	If an animal is close enough (5 units), the predator attempts an attack. The waiting period is set to 10.
	If predator confusion is activated, the look_around() function will return the number of preys that were visible at the time of the attack, and whether the attack was successful, so we can measure averages and deduce predator efficienty in evolution_main.py.

	"""


	# other animals = list of other animals on the grid
	def look_around(self):
		other_animals = [a for a in self.grid.inhabitants if a!=self]
		view = np.zeros(13)
		attack = False
		for animal in other_animals:
			see = self.scan(animal)
			if see: 
				if self.view_mode=='boolean': view[see]=1
				elif self.view_mode=='cummulative': view[see]+=1
			if self.can_attack(animal) and self.wait==0:
				# activate waiting period
				self.wait=10
				r = self.attack(animal)
				attack = True
				
		# update bias
		view[-1] = 1
		self.view = view
		if self.confusion and attack: return r


	"""
	The attack() function supports both values of predator confusion. If the confusion is activated, you can print out at each attack the number of preys visible, whether the attack was successful, and the remaining number of preys on the grid. Simply set the printing parameter to True to activate printing (useful when doing visualizations).

	"""

		
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


	def can_see(self,animal):
		# calculate distance
		d = distance(self.pos_x,self.pos_y,animal.pos_x,animal.pos_y,self.grid.width,self.grid.height)
		if d > 200: return False
		return True

	
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
				# check the prey is in one of the activated sections
				if self.mask[section]==1: return section
			return False

	"""
	The see_brain() function displays the weights in the predator's perceptrons for each possible action, so we can see roughly how the view affects them.

	"""


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

