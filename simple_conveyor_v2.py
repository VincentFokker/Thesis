import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class simple_conveyor():

    def __init__(self, queues):
        """initialize states of the variables, the lists used"""
        #init queues
        self.queues = queues
        self.init_queues(self.queues)
        
        #initialize divert points: 0=no diversion, 1=diversion
        self.D1 = False
        self.D2 = False
        self.D3 = False
        
        #initialize output points
        self.O1 = False #output of box size S 
        self.O2 = False #output of box size M
        self.O3 = False #output of box size L

        # initialize transition points: 0=no transition, 1=transition
        ## CURRENTLY NOT USED ##
        # self.T1 = 0
        # self.T2 = 0
        # self.T3 = 0

        # #initialize merge points
        # self.M1 = 0
        # self.M2 = 0
        # self.M3 = 0

        self.items_on_conv = []
        self.queue1 = []
        self.queue2 = []
        self.queue3 = []

        #intialize variables for later usage
        self.O1_location = False
        self.O2_location = False
        self.O3_location = False
        
        self.carrier_type_map = np.zeros((13,31,1))
        
    def init_queues(self, queues_list):
        """Initialize the queues with items from the queues list: is a nested list"""
        self.queue1 = queues_list[0]
        self.queue2 = queues_list[1]
        self.queue3 = queues_list[2]
        
    def update_queues(self, quenr, variable):
        'For a given queue 1-3, add a variable (1,2,3)'
        if quenr == 1:
            self.queue1.append(variable)
        if quenr == 2:
            self.queue2.append(variable)
        if quenr == 3:
            self.queue3.append(variable)
            
    def remove_from_queue(self, quenr):
        'For a given queue 1-3, remove the first in the queue'
        if quenr == 1:
            self.queue1 = self.queue1[1:]
        if quenr == 2:
            self.queue2 = self.queue2[1:]
        if quenr == 3:
            self.queue3 = self.queue3[1:]     
        
            
    def reset(self):
        "reset all the variables to zero, empty queues"
        self.D1 = False
        self.D2 = False
        self.D3 = False
        
        #initialize output points
        self.O1 = False #output of box size S 
        self.O2 = False #output of box size M
        self.O3 = False #output of box size L
        
        #empty amount of items on conv.
        self.items_on_conv = []

        #reset this var
        self.O1_location = False
        self.O2_location = False
        self.O2_location = False
        
        self.carrier_type_map = np.zeros((13,31,1))
        self.init_queues(self.queues)

    def set_step(self):
        """For all items on the conveyor, do one step in designated direction"""
        #do some step
        for item in self.items_on_conv:
                #check diverters; is any of them True and is there an order carrier? move into lane.
                if item[0] == [12,7] and self.D1 == True:
                    item[0][1] +=1
                elif item[0] == [8,7] and self.D2 == True:
                    item[0][1] +=1
                elif item[0] == [4,7] and self.D3 == True:
                    item[0][1] +=1
                
                #otherwise; all items set a step in their moving direction 
                elif item[0][1] == 7 and item[0][0] > 2: #if on the lower line, and not reached left corner:
                    item[0][0] -=1                     #move left
                elif item[0][0] ==2 and item[0][1] >2: #if on left lane, and not reached top left corner:
                    item[0][1] -=1                     #move up
                elif item[0][1] == 2 and item[0][0] <28: #if on the top lane, and not reached right top corner:
                    item[0][0] +=1                      #Move right
                elif item[0][0] == 28 and item[0][1] <7: #if on right lane, and not reached right down corner:
                    item[0][1] +=1
                elif item[0][1] > 7 and item[0][1] < 12 and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0:
                    item[0][1] +=1

        #update occupation-states of output points
        if self.carrier_type_map[7][19] == 0:
            self.O1_location = False
        else:
            self.O1_location = True
            
        if self.carrier_type_map[7][22] == 0:
            self.O2_location = False
        else:
            self.O2_location = True
            
        if self.carrier_type_map[7][25] == 0:
            self.O3_location = False
        else:
            self.O3_location = True

    def perform_actions_in_space(self):
        """Does 3 things:
        1. Check if any diverters need to be toggled (based on demand at GTP)
        2. Set a step for all items in the system in their designated direction
        3. outputs any new order carrier at the output point when O1-On == True"""
        
        ## 1. ## check if any diverters need to be toggled
        self.carrier_type_map = np.zeros((13,31,1))
        for item in self.items_on_conv:
                self.carrier_type_map[item[0][1]][item[0][0]] = item[1]
        
        #toggle D1 if needed:
        try:
            if self.carrier_type_map[7][12] == self.queue1[0]:
                self.D1 = True
                self.remove_from_queue(1)
            else:
                self.D1 = False
        except:
            self.D1=False
        
        #toggle D2 if needed:    
        try:
            if self.carrier_type_map[7][8] == self.queue2[0]:
                self.D2 = True
                self.remove_from_queue(2)
            else:
                self.D2 = False
        except:
            self.D2 = False
        
        #toggle D3 if needed:
        try:
            if self.carrier_type_map[7][4] == self.queue3[0]:
                self.D3 = True
                self.remove_from_queue(3)
            else:
                self.D3 = False
        except:
            self.D3 = False
            
        ## 2. ## set step:
        self.set_step()
        
        ## 3. ## output new order carrier(s)
        #if satisfied; output carrier type 1
        if self.O1 == True and self.carrier_type_map[7][19] == 0:
            self.O1_location =True            #occupy the output
            self.items_on_conv.append([[19,7], 1])
            self.O1 = False                   #turn output off
        
        #if statisfied; output carrier type 2
        elif self.O2 == True and self.carrier_type_map[7][22] == 0:
            self.O2_location =True            #occupy the output
            self.items_on_conv.append([[22,7], 2])
            self.O2 = False                   #turn output off
            
        #if satisfied, output carrier type 3
        elif self.O3 == True and self.carrier_type_map[7][25] == 0:
            self.O3_location =True            #occupy the output
            self.items_on_conv.append([[25,7], 3])
            self.O3 = False                   #turn output off

    def step(self, action):
        """Step function for the environment, takes an action from the action-space as input"""
        if   action==0: self.perform_actions_in_space() #set a step for all items
        elif action==1: self.D1 = not self.D1 #toggle D1
        elif action==2: self.D2 = not self.D2 #Toggle D2
        elif action==3: self.D3 = not self.D3 #Toggle D3
        elif action==4: self.O1 = not self.O1 #toggle O1
        elif action==5: self.O2 = not self.O2 #toggle O2
        elif action==6: self.O3 = not self.O3 #toggle O3
        
        # - Fix returns
        # - build reward function
        # - termination criteria
        # - additional info

    def render(self):
        """Simple render function, uses matplotlib to render the image + some additional information on the transition points"""
        print('items on conveyor:')
        print(self.items_on_conv)
        print('states of Divert points:')
        print('D1 = {}, D2 = {}, D3 = {}'.format(self.D1, self.D2, self.D3))
        print('States of output points:')
        print('O1 = {}, O2 = {}, O3 = {}'.format(self.O1, self.O2, self.O3))
        print('States of output location:')
        print('O1 = {}, O2 = {}, O3 = {}'.format(self.O1_location, self.O2_location, self.O3_location))
        print('Queue GTP 1: {}'.format(self.queue1))
        print('Queue GTP 2: {}'.format(self.queue2))
        print('Queue GTP 3: {}'.format(self.queue3))
        
        df = pd.read_csv('representation3.csv', delimiter=';', ).fillna(0)
        listoflists = df.values.tolist()
        image = np.asarray([[(255,255,255) if x =='x' else (220,220,220) if x =='y' else (0,0,0) for x in item] for item in listoflists])

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([(255, 213, 171) if item[1] ==1 else (235, 151, 136) if item[1] ==2 else (183, 71, 42)]) 
        plt.imshow(np.asarray(image))
        plt.show()
    
    def render_cv2(self):
        """render with opencv, for faster processing"""
        df = pd.read_csv('representation3.csv', delimiter=';', ).fillna(0)
        listoflists = df.values.tolist()
        image = np.asarray([[(255,255,255) if x =='x' else (220,220,220) if x =='y' else (0,0,0) for x in item] for item in listoflists])

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([(255, 213, 171) if item[1] ==1 else (235, 151, 136) if item[1] ==2 else (183, 71, 42)]) 
        
        #resize with PIL
        im = Image.fromarray(np.uint8(image))
        img = im.resize((600,240))
        cv2.imshow("Simulation-v0.1", np.array(img))
        cv2.waitKey(0)

## Test the item
queues = [[1,2,3,2,3,1], [2,3,1,3,1,2], [1,3,2,1,2,2]] #sample queues for format WHERE 1=S, 2=M, 3=L
env = simple_conveyor(queues)
env.reset()

#Build action list
order_list = []
for index in range(len(env.queues[0])):
    order_list.append([item[index] for item in env.queues])

#flat_list = [item for sublist in l for item in sublist]
order_list = [item for sublist in order_list for item in sublist]
print("Sequence of order is:",  order_list)
order_action_list = [4 if x == 1 else 5 if x ==2 else 6 for x in order_list]
print("Resulting in sequence of actions: ", order_action_list)

#run short trail:
env.reset()

for item in order_action_list:
    env.step(item)
    for _ in range(4):
        env.step(0)
        env.render_cv2()


    
while env.queue1 + env.queue2 + env.queue3 != []:
    env.step(0)
    env.render_cv2()