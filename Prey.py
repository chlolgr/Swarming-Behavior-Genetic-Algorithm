from sklearn.linear_model import Perceptron
import random
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from Grid import *
from utils import *


"""
To understand the Prey class, just read the comments on the Predator class. They're basically the same thing. Actually ideally I'll have time to put the similarities in an Animal class and make Prey and Predator inherit it, but that's unlikely. If you've inherited this project, it might be a good idea to start by doing that. 
There's also a bunch of functions you can carefully move to the utils file while we're at it. Have fun!

"""


class Prey():
	
	def __init__(self,grid,brain=None):

		self.grid = grid
		self.grid.add_animal(self)

		if brain: self.brain = brain
		else: 
			self.brain = Perceptron(max_iter=1)
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
		x = np.random.choice([0,1],(4,25))
		y = np.arange(4)
		self.brain.fit(x,y)
		self.brain.coef_ = np.random.rand(4,25)*2-1


	def move(self,mode):

		if mode=='stay still': return

		elif mode=='turn right':
			self.orientation -=8
			self.orientation %= 360

		elif mode=='turn left':
			self.orientation +=8
			self.orientation %= 360

		angle = math.radians(self.orientation)
		self.pos_x,self.pos_y = new_position(self.pos_x,self.pos_y,1,angle)

		self.pos_x = self.pos_x%self.grid.width
		self.pos_y = self.pos_y%self.grid.height


	def pick_move(self):

		predictions = self.brain.decision_function(self.view.reshape(1,-1))
		predictions = predictions.flatten()
		# convert to probability distribution
		predictions = np.array([1/(1+math.exp(-p)) for p in predictions])
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
		view[-1] = 1
		self.view = view


	def can_see(self,animal):
		# distance
		d = distance(self.pos_x,self.pos_y,animal.pos_x,animal.pos_y,self.grid.width,self.grid.height)
		if d > 100: return False
		return True


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
			
			section = angle_to_section(aangle,det)

			if isinstance(animal,Prey): return section
			else:
				if section: return section+12
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
		p1 = plt.bar(x,np.concatenate((y_still[:12],np.array([y_still[-1]])),axis=0),alpha=.5,color='lime')
		p2 = plt.bar(x,y_still[12:],alpha=.5,color='orange')
		plt.scatter(x,np.concatenate((y_still[:12],np.zeros(1)),axis=0)+y_still[12:],c='k',marker='.')
		plt.legend((p1[0],p2[0]),('preys','predators'))
		plt.show()

		plt.title('Weights in Deciding to Turn Right')
		plt.xticks(x,weights)
		plt.xlabel('view section (left to right)')
		plt.ylabel('weight')
		p1 = plt.bar(x,np.concatenate((y_right[:12],np.array([y_right[-1]])),axis=0),alpha=.5,color='lime')
		p2 = plt.bar(x,y_right[12:],alpha=.5,color='orange')
		plt.scatter(x,np.concatenate((y_right[:12],np.zeros(1)),axis=0)+y_right[12:],c='k',marker='.')
		plt.legend((p1[0],p2[0]),('preys','predators'))
		plt.show()
		
		plt.title('Weights in Deciding to Turn Left')
		plt.xticks(x,weights)
		plt.xlabel('view section (left to right)')
		plt.ylabel('weight')
		p1 = plt.bar(x,np.concatenate((y_left[:12],np.array([y_left[-1]])),axis=0),alpha=.5,color='lime')
		p2 = plt.bar(x,y_left[12:],alpha=.5,color='orange')
		plt.scatter(x,np.concatenate((y_left[:12],np.zeros(1)),axis=0)+y_left[12:],c='k',marker='.')
		plt.legend((p1[0],p2[0]),('preys','predators'))
		plt.show()

		plt.title('Weights in Deciding to Go Forward')
		plt.xticks(x,weights)
		plt.xlabel('view section (left to right)')
		plt.ylabel('weight')
		p1 = plt.bar(x,np.concatenate((y_forward[:12],np.array([y_forward[-1]])),axis=0),alpha=.5,color='lime')
		p2 = plt.bar(x,y_forward[12:],alpha=.5,color='orange')
		plt.scatter(x,np.concatenate((y_forward[:12],np.zeros(1)),axis=0)+y_forward[12:],c='k',marker='.')
		plt.legend((p1[0],p2[0]),('preys','predators'))
		plt.show()	
	

	def die(self):
		self.grid.delete_animal(self)


	def save(self,name):
		pickle.dump(self,open('preys/'+name+'/pkl','wb'))
					
	
