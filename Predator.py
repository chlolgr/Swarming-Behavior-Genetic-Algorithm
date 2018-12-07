from sklearn.linear_model import Perceptron
import random
import numpy as np
import time
from Prey import Prey
from Grid import Grid
import math
import pickle
import matplotlib.pyplot as plt



class Predator():
	

	def __init__(self,grid,coefs=None,brain=None):

		self.grid = grid
		self.grid.add_animal(self)

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


	def initialize_brain(self,coefs=None):
		# make random input and target vectors
		x = np.random.choice([0,1],(4,13))
		y = np.arange(4)
		# fit it (1 iteration) to initialize all parameters
		self.brain.fit(x,y)
		# can used pre-defined coefs (for inheritance for instance)
		if coefs: self.brain.coefs_ = coefs
		# otherwise, weights attributed randomly
		else: self.brain.coef_ = np.random.rand(4,13)%2-1
		

	def move(self,mode):

		if mode=='stay still': return

		elif mode=='turn right':
			self.orientation -=6
			self.orientation %= 360
			angle = math.radians(self.orientation)
			# calculate new position
			ox,oy = self.pos_x,self.pos_y
			px,py = self.pos_x+3,self.pos_y
			new_x = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
			new_y = oy + math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)

		elif mode=='turn left':
			self.orientation +=6
			self.orientation %= 360
			angle = math.radians(self.orientation)
			# calculate new position
			ox,oy = self.pos_x,self.pos_y
			px,py = self.pos_x+3,self.pos_y
			new_x = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
			new_y = oy + math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)
		
		elif mode=='move forward':
			angle = math.radians(self.orientation)
			# calculatee new position
			ox,oy = self.pos_x,self.pos_y
			px,py = self.pos_x+3,self.pos_y
			new_x = ox + math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
			new_y = oy + math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)

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


	# other animals = list of other animals on the grid
	def look_around(self):
		# define field of vision for the prey
		#	-> 100 units around but have to remember they can cross the borders
		other_animals = [a for a in self.grid.inhabitants if a!=self]
		view = np.zeros(13)
		for animal in other_animals:
			see = self.scan(animal)
			if see: view[see]=1

			if self.can_attack(animal) and self.wait==0:
				self.attack(animal)
				# activate waiting period
				self.wait=10
				
		# update bias
		view[-1] = random.random()
		self.view = view

		
	def attack(self,animal,printing=True,confusion=True):
		if confusion:
			# counting the number of preys the predator can see
			n_visible = 1
			for animal_ in self.grid.inhabitants:
				# calculate distance between target and other preys
				if isinstance(animal_,Prey) and animal_!=animal:
					x1 = animal.pos_x
					y1 = animal.pos_y
					x2 = animal_.pos_x
					y2 = animal_.pos_y
					if self.grid.width-x1 < x1: x1 = x1-self.grid.width
					if self.grid.width-x2 < x2: x2 = x2-self.grid.width
					if self.grid.height-y1 < y1: y1 = y1-self.grid.height
					if self.grid.height-y2 < y2: y2 = y2-self.grid.height
					dx = abs(x2-x1)**2
					dy = abs(y2-y1)**2
					# if distance<30 units, it adds to confusion
					if self.scan(animal_) and math.sqrt(dx+dy)<=30: n_visible+=1
			if printing: print(n_visible,'prey(s) in sight')
			if np.random.rand() < 1/n_visible: 
				animal.die()
				if printing: 
					print('Attack: successful')
					print(self.grid.n_preys(),'preys left\n')
			else: 
				if printing: print('Attack: unsuccessful\n')
		else: animal.die()


	def can_see(self,animal):
		# calculate distance
		x1 = self.pos_x
		y1 = self.pos_y
		x2 = animal.pos_x
		y2 = animal.pos_y
		if self.grid.width-x1 < x1: x1 = x1-self.grid.width
		if self.grid.width-x2 < x2: x2 = x2-self.grid.width
		if self.grid.height-y1 < y1: y1 = y1-self.grid.height
		if self.grid.height-y2 < y2: y2 = y2-self.grid.height
		dx = abs(x2-x1)**2
		dy = abs(y2-y1)**2
		if math.sqrt(dx+dy) > 200: return False
		else: return True

	
	def can_attack(self,animal):
		# calculate distance
		x1 = self.pos_x
		y1 = self.pos_y
		x2 = animal.pos_x
		y2 = animal.pos_y
		if self.grid.width-x1 < x1: x1 = x1-self.grid.width
		if self.grid.width-x2 < x2: x2 = x2-self.grid.width
		if self.grid.height-y1 < y1: y1 = y1-self.grid.height
		if self.grid.height-y2 < y2: y2 = y2-self.grid.height
		dx = abs(x2-x1)**2
		dy = abs(y2-y1)**2

		if math.sqrt(dx+dy) > 5: return False

		else:

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

			if aangle>90: return False
			else: return True
			

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

			# this part isn't used in the original experiment because there's only one wolf
			# BUT HERE COMES THE FUTURE
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
