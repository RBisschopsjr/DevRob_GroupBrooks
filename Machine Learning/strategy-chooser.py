import numpy as np
import matplotlib.pyplot as plt

class Agent:

	def __init__(self, policy_names):
	
		# Maximum time of looking around
		self.attention = 6	 								
		self.policy_names = policy_names
		# Stores values to base decision on
		self.policy_values = [1.0 for _ in policy_names] 
	
	
	'''
		Returns the probability of choosing each of its different policies as a list.
		This begins as a uniform distribution over the possible policies.
	'''
	def get_probs(self):
		return [x/sum(self.policy_values) for x in self.policy_values]
	
	
	'''
		Chooses a specific policy (as a string name) to enact pseudo-randomly, 
		given its preferences. 
	'''
	def get_policy(self):
		probs = self.get_probs()
		choice = np.random.uniform()
		for i, p in enumerate(probs):
			if p > choice:
				choice -= p
			else:
				return self.policy_names[i]
		return self.policy_names[-1]
		
		
	'''
		Updates the beliefs given an observations, which must be a probability vector as
		as list	with length equal to the number of policies.
	'''
	def update_policies(self, observations):
		if len(observations) != len(self.policy_values):
			print("Observation length must equal the mumber of different policies")
		else:		
			self.policy_values = [x + y for x, y in zip(self.policy_values, observations)] 





'''
	Helper function: provides a pseudo-random time spent using a specific strategy.
'''
def get_observation(agent, mu, sigma, index):

	time_spent = max(0, min(agent.attention, np.random.normal(mu, sigma)))
	policy_eval = 1 - (time_spent/agent.attention)
	
	observation = [(1-policy_eval)/(len(agent.policy_names)-1) for _ in agent.policy_names]
	observation[index] = policy_eval

	return observation



############################
#                          #
# Defines an example agent #
#                          #
############################

policies = ["random", "gaze-directed"]
policy_params = [(5,5), (2, 5)]

agent = Agent(policies)


epochs = 1000
beliefs = [agent.get_probs()[1]]
for _ in range(epochs):
	
	# Get the policy from the agent (exploration)
	policy = agent.get_policy()
	
	
	
	# Get an observation based on that policy
	# THIS CODE SHOULD BE REPLACED BY THE AGENT BEHAVIOUR INSTEAD
	index = policies.index(policy)
	mu, sigma = policy_params[index]
	observation = get_observation(agent, mu, sigma, index)
	# END OF TO-REPLACE
	
	
	
	# Agent updates its beliefs on the best policy
	# given the new observation
	agent.update_policies(observation)
	beliefs.append(agent.get_probs()[1])
	
	
# Plotting the policy choice progression
plt.plot(beliefs)
plt.xlabel("Epochs")
plt.xlim([0, epochs])
plt.ylabel("P(gaze-directed)")
plt.ylim([0, 1])
plt.show()

# Note that anything with p(x) > 0.5 will always be chosen
# if we do exploitation instead of exploration.




	

