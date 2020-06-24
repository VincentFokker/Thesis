from random import *
from PIL import Image, ImageDraw
import cv2
import numpy as np

class xywalking():
    def __init__(self):
        self.location = [100,100]
        #print("start location = (0,0)")
        
    def step(self, action):
        if   action==0: self.location[0] += 10
        elif action==1: self.location[0] -= 10
        elif action==2: self.location[1] += 10
        elif action==3: self.location[1] -= 10
        return self.location
    
                    
    def reset(self):
        self.location = [100,100]
    
    def render(self):
        #define shape
        height = 200
        width = 200
        img_array = np.zeros((height, width, 3))

        #transform to PIL object
        im = Image.fromarray(img_array, mode="RGB")

        #draw the system
        draw = ImageDraw.Draw(im)

        #draw rectangle
        x,y = self.location
        draw.rectangle([x-5,y-5,x+5,y+5], fill='#b7472a')

        cv2.imshow("Simulation-v0.1", np.array(im))
        cv2.waitKey(10)

env = xywalking()
env.reset()

for _ in range(2000):
    state = env.step(randint(0,3))
    if (state[0] >= 200 or state[0] <= 0) or (state[1] >= 200 or state[1] <= 0):
        env.reset()
    #render situation
    env.render()
