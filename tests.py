import matplotlib.pyplot as plt
import numpy as np


"""
This is the file I use to display most of the results on the effect of masks and predator confusion.
I ran tests with the measure_efficiency() function in evolution_main.py, wrote them by hand here (it's kind of cheating but also I'm doing this at the last minute like an idiot so I don't have time to do it a cleaner way.

The function names are hella long but at least they're clear. Sorry and you're welcome. 


"""

def show_effect_of_predator_confusion_on_preys_consumed_per_swarm_size():
	# data is a list of tuples
	# tuples go: (swarm size, average number of preys left alive without confusion, average number of preys left aline with confusion)
	# It's an average on 10 simmulations for each data point.
	data = [(5,.4,1),(10,.4,1),(15,1,3.5),(20,2.2,7.7),(25,3.6,13.3),(30,9.3,16.5),(35,12.8,23),(40,18.6,27.6),(45,21.3,34.7),(50,25,39.3)]

	x = [point[0] for point in data]
	y_false = [point[0]-point[1] for point in data]
	y_true = [point[0]-point[2] for point in data]

	plt.plot(x,y_false,label='without confusion',marker='o')
	plt.plot(x,y_true,label='with confusion',marker='o')
	plt.title("Number of preys consumed depending on total number of preys")
	plt.xlabel("total number of preys")
	plt.ylabel("number of preys consumed")
	plt.legend()
	plt.xticks(np.linspace(0,50,11))
	plt.show()


def show_effect_of_vision_angle_on_efficiency():
	# data is still a list of tuples
	# tuples go: (vision angle, avg number of preys eaten, avg number of visible preys at times of attack, avg attack success rate)
	data = [(30,5.2,1.78,.79,6.8), (60,11.6,2.29,.69,16.2), (90,11,3,.55,19.2), (120,11.4,3.74,.5,24.6), (150,10.1,4.9,.44,24.2), (180,10.1,4.7,.43,26)]
	x = [point[0] for point in data]
	y_consumed = [point[1] for point in data]
	y_visible = [point[2] for point in data]
	y_rate = [point[3] for point in data]
	y_trials = [point[4] for point in data]

	plt.title("Average number of preys eaten for different angles of vision")
	plt.xlabel("angle of vision")
	plt.ylabel("number of preys eaten")
	plt.plot(x,y_consumed,marker='o')
	plt.xticks(np.arange(30,181,30))
	plt.show()

	plt.title("Average number of visible preys at times of attack for different angles of vision")
	plt.xlabel("angle of vision")
	plt.ylabel("number of visible preys at attack")
	plt.plot(x,y_visible,marker='o')
	plt.xticks(np.arange(30,181,30))
	plt.show()

	plt.title("Average success rate for different angles of vision")
	plt.xlabel("angle of vision")
	plt.ylabel("success_rate")
	plt.plot(x,y_rate,marker='o')
	plt.xticks(np.arange(30,181,30))
	plt.show()

	plt.title("Average number of attack attempts for different angles of vision")
	plt.xlabel("angle of vision")
	plt.ylabel("number of attacks")
	plt.plot(x,y_trials,marker='o')
	plt.xticks(np.arange(30,181,30))
	plt.show()

show_effect_of_vision_angle_on_efficiency()
