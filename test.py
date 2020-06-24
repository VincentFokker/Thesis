import gym
import numpy as np 

env = gym.make('CartPole-v0')

bestlenght = 0
episode_lengths = []

best_weights = np.zeros[4]

for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)

    length = []

    for j in range (100):
        observation = env.reset()
        done = False
        cnt = 0

while not done:
    #env.render()

    cnt += 1
    action = env.action_space.sample()

    observation, reward, done, _ = env.step(action)

    if done:
        break
print('game lasted', cnt, 'moves')
