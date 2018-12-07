from sklearn.neural_network import MLPClassifier
import random
import numpy as np
import time
import math
from initialize_prey_brains import *
import matplotlib.pyplot as plt
from Grid import *



class Prey():
	
	def __init__(self,grid,brain=None):

		self.grid = grid
		self.grid.add_animal(self)

		if brain: self.brain = brain
		else: 
			self.brain = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', max_iter=1)
			self.initialize_brain()
		self.pos_x = random.random()*self.grid.width
		self.pos_y = random.random()*self.grid.height
		self.orientation = int(random.random()*360)
		self.view = np.zeros(25)
		self.fitness = 0

	
	def copy(self):
		copy = Prey(self.grid)
		copy.brain = self.brain
		return copy


	def set(self,x,y,o):
		self.pos_x = x
		self.pos_y = y
		self.orientation = o


	def initialize_brain(self):
		x = np.array([np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])])
		y = np.array([np.array([0,1,0,1])])
		self.brain.fit(x,y)
		self.brain.coefs_ = [np.random.rand(25,15)*2-1, np.random.rand(15,4)*2-1]


	def move(self,mode):

		if mode=='stay still': return

		elif mode=='turn right':
			self.orientation -=8
			self.orientation %= 360
			angle = math.radians(self.orientation)
			ox,oy = self.pos_x,self.pos_y
			px,py = self.pos_x+1,self.pos_y
			new_x = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
			new_y = oy + math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)

		elif mode=='turn left':
			self.orientation +=8
			self.orientation %= 360
			angle = math.radians(self.orientation)
			ox,oy = self.pos_x,self.pos_y
			px,py = self.pos_x+1,self.pos_y
			new_x = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
			new_y = oy + math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)
		
		elif mode=='move forward':
			angle = math.radians(self.orientation)
			ox,oy = self.pos_x,self.pos_y
			px,py = self.pos_x+1,self.pos_y
			new_x = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
			new_y = oy + math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)

		self.pos_x = new_x%511
		self.pos_y = new_y%511


	def pick_move(self):
		predictions = self.brain.predict_proba(self.view.reshape(1,-1))
		next_move = np.argmax(predictions)
		if next_move==0: self.move('stay still')
		elif next_move==1: self.move('turn right')
		elif next_move==2: self.move('turn left')
		elif next_move==3: self.move('move forward')
		else: print("\nCareful, orientation not recognized.\n")

		self.fitness += (self.grid.n_preys())


	# other animals = list of other animals on the grid
	def look_around(self):
		# define field of vision for the prey
		#	-> 100 units around but have to remember they can cross the borders
		other_animals = [a for a in self.grid.inhabitants if a!=self]
		view = np.zeros(25)
		for animal in other_animals:
			see = self.scan(animal)
			if see: view[see]=1
		view[-1] = random.random()
		self.view = view


	def can_see(self,animal):
		# distance
		x1 = self.pos_x
		y1 = self.pos_y
		x2 = animal.pos_x
		y2 = animal.pos_y
		if 511-x1 < x1: x1 = x1-511
		if 511-x2 < x2: x2 = x2-511
		if 511-y1 < y1: y1 = y1-511
		if 511-y2 < y2: y2 = y2-511
		dx = abs(x2-x1)**2
		dy = abs(y2-y1)**2
		if math.sqrt(dx+dy) > 100: return False
		else: return True


	# section_id = int starting at 0 indicating number of section from left side
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

				if det>=0:
					if aangle>=0 and aangle<=15: return 5
					if aangle>15 and aangle<=30: return 4
					if aangle>30 and aangle<=45: return 3
					if aangle>45 and aangle<=60: return 2
					if aangle>60 and aangle<=75: return 1
					if aangle>75 and aangle<=90: return 0

				else:
					if aangle>0 and aangle<=15: return 6
					if aangle>15 and aangle<=30: return 7
					if aangle>30 and aangle<=45: return 8
					if aangle>45 and aangle<=60: return 9
					if aangle>60 and aangle<=75: return 10
					if aangle>75 and aangle<=90: return 11

			else:
				
				if det>=0:
					if aangle>=0 and aangle<=15: return 17
					if aangle>15 and aangle<=30: return 16
					if aangle>30 and aangle<=45: return 15
					if aangle>45 and aangle<=60: return 14
					if aangle>60 and aangle<=75: return 13
					if aangle>75 and aangle<=90: return 12

				else:
					if aangle>0 and aangle<=15: return 18
					if aangle>15 and aangle<=30: return 19
					if aangle>30 and aangle<=45: return 20
					if aangle>45 and aangle<=60: return 21
					if aangle>60 and aangle<=75: return 22
					if aangle>75 and aangle<=90: return 23

			return False

	def die(self):
		self.grid.delete_animal(self)
					
	
