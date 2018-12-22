import matplotlib.pyplot as plt
import numpy as np
from Prey import Prey


"""
The Grid class is the easiest to get. It contains all animal on the grid and has the function to do visualizations. That's pretty much it. 
The show() function isn't animated. Why is it there, you ask? I don't know anymore. Could be useful, could be a draft of the visualize function. I've forgotten and to be 100% real with you I'm just afraid of deleting it. 

"""


class Grid():
	

	def __init__(self,width,height):
		self.width = width
		self.height = height
		self.inhabitants = []


	def add_animal(self,animal):
		self.inhabitants.append(animal)


	def delete_animal(self,animal):
		self.inhabitants.remove(animal)

	
	def n_preys(self):
		n_preys = 0
		for animal in self.inhabitants:
			if isinstance(animal,Prey): n_preys+=1
		return n_preys


	def show(self):

		x_prey,y_prey = [],[]
		x_pred,y_pred = [],[]
		for animal in self.inhabitants:
			if isinstance(animal,Prey):
				x_prey.append(animal.pos_x)
				y_prey.append(animal.pos_y)
			else:
				x_pred.append(animal.pos_x)
				y_pred.append(animal.pos_y)
		
		x_prey = np.array(x_prey)
		y_prey = np.array(y_prey)

		x_pred = np.array(x_pred)
		y_pred = np.array(y_pred)

		fig = plt.figure(figsize=(7,7))

		if len(x_prey)>0: plt.plot(x_prey,y_prey,'k.')
		if 'x_pred' in locals(): plt.plot(x_pred,y_pred,'r*')

		plt.ylim(top=self.height-1, bottom=0)
		plt.xlim(left=0, right=self.width-1)
		
		plt.show()


	def visualize(self):

		fig = plt.figure(figsize=(7,7))

		x_prey,y_prey = [],[]
		x_pred,y_pred = [],[]

		for animal in self.inhabitants:
			if isinstance(animal,Prey):
				x_prey.append(animal.pos_x)
				y_prey.append(animal.pos_y)
			else:
				x_pred.append(animal.pos_x)
				y_pred.append(animal.pos_y)


		ax = plt.gca()
		ax.set_xlim(0,self.width-1)
		ax.set_ylim(0,self.height-1)
		line, = ax.plot(x_prey,y_prey,'k.')

		ax1 = plt.gca()
		ax1.set_xlim(0,self.width-1)
		ax1.set_ylim(0,self.height-1)
		line1, = ax1.plot(x_pred,y_pred,'r*')


		for i in range(2000):

			for animal in self.inhabitants: animal.look_around()
			for animal in self.inhabitants: animal.pick_move()

			x_prey,y_prey = [],[]
			x_pred,y_pred = [],[]
			for animal in self.inhabitants:
				if isinstance(animal,Prey):
					x_prey.append(animal.pos_x)
					y_prey.append(animal.pos_y)
				else:
					x_pred.append(animal.pos_x)
					y_pred.append(animal.pos_y)

			line.set_xdata(x_prey)
			line.set_ydata(y_prey)

			line1.set_xdata(x_pred)
			line1.set_ydata(y_pred)
	
	
			#plt.draw()
			plt.pause(1e-17)
			#time.sleep(.1)

		for animal in self.inhabitants: 
			if not isinstance(animal,Prey): return animal.fitness


	# to be used with several predators on the grid
	# not used in this version of the project
	def new_visualize(self):

		fig = plt.figure(figsize=(7,7))

		x_prey,y_prey = [],[]
		x_pred,y_pred = [],[]

		for animal in self.inhabitants:
			if not isinstance(animal,Prey): predator_copy = animal.copy(grid=Grid(self.width,self.height))
			

		for animal in self.inhabitants:
			if isinstance(animal,Prey): animal.copy(predator_copy.grid)
	
		for animal in predator_copy.grid.inhabitants:
			if isinstance(animal,Prey):
				x_prey.append(animal.pos_x)
				y_prey.append(animal.pos_y)
			else:
				x_pred.append(animal.pos_x)
				y_pred.append(animal.pos_y)


		ax = plt.gca()
		ax.set_xlim(0,self.width-1)
		ax.set_ylim(0,self.height-1)
		line, = ax.plot(x_prey,y_prey,'k.')

		ax1 = plt.gca()
		ax1.set_xlim(0,self.width-1)
		ax1.set_ylim(0,self.height-1)
		line1, = ax1.plot(x_pred,y_pred,'r*')


		for i in range(2000):

			for animal in predator_copy.grid.inhabitants: animal.look_around()
			for animal in predator_copy.grid.inhabitants: animal.pick_move()

			x_prey,y_prey = [],[]
			x_pred,y_pred = [],[]
			for animal in predator_copy.grid.inhabitants:
				if isinstance(animal,Prey):
					x_prey.append(animal.pos_x)
					y_prey.append(animal.pos_y)
				else:
					x_pred.append(animal.pos_x)
					y_pred.append(animal.pos_y)

			line.set_xdata(x_prey)
			line.set_ydata(y_prey)

			line1.set_xdata(x_pred)
			line1.set_ydata(y_pred)
	
	
			#plt.draw()
			plt.pause(1e-17)
			#time.sleep(.1)

		for animal in predator_copy.grid.inhabitants: 
			if not isinstance(animal,Prey): return animal.fitness

