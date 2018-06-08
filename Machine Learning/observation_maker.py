def time_to_observation(time, attention, nr_policies, index):

	time = max(0, min(attention, time))
	fitness = (attention - time)/attention

	observation = [(1-fitness)/(nr_policies-1) for x in range(nr_policies)]
	observation[index] = fitness
	
	return observation
	
	
time = 5
attention = 20
nr_policies = 2 # Gaze-following and random
index = 1	# Chose random apparently

print(time_to_observation(time, attention, nr_policies, index))
