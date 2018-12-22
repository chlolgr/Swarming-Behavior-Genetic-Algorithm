class Animal(Object):
	

	def __init__(self,grid):

		self.grid = grid
		grid.add_animal(self)
		self.pos_x = random.random()*self.grid.width
		self.pos_y = random.random()*self.grid.height
		self.orientation = int(random.random()*360)

		self.fitness = 0


	def set(self,x,y,o):
		self.pos_x = x
		self.pos_y = y
		self.orientation = o


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


	def move(self,mode):

		if mode=='stay still': return

		elif mode=='turn right':
			self.orientation -= self.agility
			self.orientation %= 360

		elif mode=='turn left':
			self.orientation += self.agility
			self.orientation %= 360

		angle = math.radians(self.orientation)
		self.pos_x,self.pos_y = new_position(self.pos_x,self.pos_y,self.speed,angle)

		self.pos_x = self.pos_x%self.grid.width
		self.pos_y = self.pos_y%self.grid.height


	def can_see(self,animal):
		# calculate distance
		d = distance(self.pos_x,self.pos_y,animal.pos_x,animal.pos_y,self.grid.width,self.grid.height)
		if d > self.sight: return False
		return True
