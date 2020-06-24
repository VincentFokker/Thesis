import gym
import random
from IPython.display import clear_output
import numpy as np

#build environment
env = gym.make('FrozenLake8x8-v0').env

env.reset()
env.render()

print("Action space {}".format(env.action_space))
print("State space {}".format(env.observation_space))

# build the Q-table
q_table= np.zeros([env.observation_space.n, env.action_space.n])


#hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1


#for plotting metrics
all_epochs = []
all_penalties = []


for i in range(1, 100001): #define amount of iterations
    state = env.reset()
    
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample() # explore the action space
        else:
            action = np.argmax(q_table[state]) # exploit learned values
            
        next_state, reward, done, info = env.step(action)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state,action] = new_value
        
        if reward ==-10:
            penalties += 1
        
           
        state = next_state
        epochs += 1
    
    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode: {}".format(i))
print("Training Finished.\n")