import math
import numpy as np

# all possible (logical) masks for predators
# could add dissymmetrical masks for predators losing an eye if you're really enjoying yourself
masks = np.array([
	np.array([0,0,0,0,0,1,1,0,0,0,0,0,1]),
	np.array([0,0,0,0,1,1,1,1,0,0,0,0,1]),
	np.array([0,0,0,1,1,1,1,1,1,0,0,0,1]),
	np.array([0,0,1,1,1,1,1,1,1,1,0,0,1]),
	np.array([0,1,1,1,1,1,1,1,1,1,1,1,1]),
	np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
	])

# This is math. Takes some visualization to get but it's basic trigonometry.
def new_position(x,y,norm,angle):
	px,py = x+norm,y
	new_x = x + math.cos(angle) * (px-x) - math.sin(angle) * (py-y)
	new_y = y + math.sin(angle) * (px-x) + math.cos(angle) * (py-y)
	return new_x,new_y

# euclidian distance as the animal can move however they like
# /!\ REMEMBER ANIMALS CAN CROSS THE BORDERS OF THE GRID
# w_bound and h_bound are the furthest they can go on the x and y axes
def distance(x1,y1,x2,y2,w_bound,h_bound):
	x1 = x1*(1-int(w_bound-x1<x1)) - (w_bound-x1)*(int(w_bound-x1<x1))
	x2 = x2*(1-int(w_bound-x2<x2)) - (w_bound-x2)*(int(w_bound-x2<x2))
	y1 = y1*(1-int(h_bound-y1<y1)) - (h_bound-y1)*(int(h_bound-y1<y1))
	y2 = y2*(1-int(h_bound-y2<y2)) - (h_bound-y2)*(int(h_bound-y2<y2))
	dx = abs(x2-x1)**2
	dy = abs(y2-y1)**2
	return math.sqrt(dx+dy)

# This just associates angles to sections.
# If you don't get it just make yourself a little visualization and it should be pretty clear.
def angle_to_section(angle,det):
	if angle>90: return False
	# det>0 means the point you're looking at is on your left so it'll be in the first sections
	if det>=0:
		if angle>=0 and angle<=15: return 5
		if angle>15 and angle<=30: return 4
		if angle>30 and angle<=45: return 3
		if angle>45 and angle<=60: return 2
		if angle>60 and angle<=75: return 1
		if angle>75 and angle<=90: return 0
	# det <0 means the point you're looking at is on your right so it'll be in the last sections
	else:
		if angle>0 and angle<=15: return 6
		if angle>15 and angle<=30: return 7
		if angle>30 and angle<=45: return 8
		if angle>45 and angle<=60: return 9
		if angle>60 and angle<=75: return 10
		if angle>75 and angle<=90: return 11
