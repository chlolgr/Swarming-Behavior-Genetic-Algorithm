# Predator Confusion is Sufficient to Evolve Swarming Behavior


## Introduction

This project was developed in my Robotics AI class (M.Sc. Data Science) within the Faculté des Sciences Sorbonne Université in Paris, France. 

I used [a research paper from the State University of Michigan](http://rsif.royalsocietypublishing.org/content/10/85/20130305) to write this code. The original code is also on github. It is however not Python, and I did not use any of it in this version.


## Tools and Methods

I used perceptrons to model the brains of preys and predators. The point is to evolve predators to chase preys, and preys to survive predators. Animals look around themselves with a field of vision of 180° divided into 12 slots. If they see an animal in a slot, it activates. There is only one predator per simulation, so predators have 12 slots. Preys can tell other preys apart from the predator, so they have 24 slots. These slots put together form vectors that serve as inputs for the perceptrons (size 13 for predators and 25 for preys counting the bias). The perceptrons are optimized with a genetic algorithm. 

I used pools of 10 predators and 10 preys for each generation. Every (predator,prey) pair is tested, using 1 predator and 50 instances of the prey for each simulation. I used multiprocessing to quicken up the calculations (they're still quite long though, and can definitely be made less so by grouping some calculations within). Then, I calculated the average of every animal's fitnesses, and ranked them using that. 

Children are generated from the 3 animals with the best fitnesses, plus the 6th of the group (to maintain some diversity). Two parents are picked at random, and the weights of their perceptrons are distributed randomly to the child. I added 10% of mutations. 


## Visualizations

### Swarms at generations 33 and 113:

<img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/generation_33_perceptrons_circles.png" width="400"> <img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/generation_113_perceptrons_swarms.png" width="400">

### Predator efficiency with and without confusion:

<img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/confusion_true_vs_false_swarm_size.png" width="800">

### Importance of retina angle: 

<img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/preys_consumed_per_vision_angle.png" width="400"> <img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/attempts.png" width="400">
<img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/success_rate_per_vision_angle.png" width="400"> <img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/visible_preys_different_vision_angles.png" width="400">

### Adding cummulative instead of boolean vision at generation 124:

<img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/fitness_generations.png" width="800">

### Predator perceptrons weights:

<img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/weights_stay_still.png" width="400"> <img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/weights_forward.png" width="400">
<img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/weights_left.png" width="400"> <img src="https://raw.githubusercontent.com/chlolgr/swarming-behavior-genetic-algorithm/master/pictures/weights_right.png" width="400">
