#########################################################################
# Version: 8                                                            #
# Feature: Scaleable version with gready approach to GTP allocation     #


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.colors import ColorConverter
from copy import copy
from random import randint
import seaborn as sns
import random


class simple_conveyor():

######## INITIALIZATION OF VARIABLES ###############################################################################################################
    def __init__(self, queues, amount_gtp=3, amount_output=3):
        """initialize states of the variables, the lists used"""
        #init queues
        self.queues = queues
        self.init_queues = self.queues
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(queues))]



        #init config
        self.amount_of_gtps = amount_gtp
        self.amount_of_outputs = amount_output
        self.exception_occurence = 0.05        # % of the times, an exception occurs
        self.process_time_at_GTP = 6          # takes 30 timesteps

        self.reward = 0.0
        self.terminate = False

        #colors
        self.pallette = (np.asarray(sns.color_palette("Reds", self.amount_of_outputs)) * 255).astype(int)

        # build env
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        #define where the operators, diverts and outputs of the GTP stations are
        self.operator_locations = [[i, self.empty_env.shape[0]-1] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        self.output_locations = [[i,7]for i in range(self.empty_env.shape[1]-self.amount_of_outputs*2-1,self.empty_env.shape[1]-2,2)]
        self.diverter_locations = [[i, 7] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        self.merge_locations = [[i-1, 7] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        print("operator locations: ", self.operator_locations)
        print("output locations: ", self.output_locations)
        print("diverter locations: ", self.diverter_locations)
        print("Merge locations: ", self.merge_locations)

        #initialize divert points: False=no diversion, True=diversion
        self.D_states = {}
        for i in range(1,len(self.diverter_locations)+1):
            self.D_states[i] = False
        
        #initialize output points
        self.O_states = {}
        for i in range(1,len(self.output_locations)+1):
            self.O_states[i] = 0

        # initialize transition points: 0=no transition, 1=transition
        self.T_states = {}
        for i in range(1,len(self.operator_locations)+1):
            self.T_states[i] = False

        #initialize merge points
        self.M_states = {}
        for i in range(1,len(self.merge_locations)+1):
            self.M_states[i] = False

####### FOR SIMULATION ONLY 
        self.W_times = {}
        for i in range(1,len(self.operator_locations)+1):
            self.W_times[i] = self.process_time_at_GTP + 150+8*self.amount_of_gtps + randint(-10, 10)
        print("Process times at operator are:", self.W_times)
####### FOR SIMULATION ONLY

        #initialize conveyor memory
        self.items_on_conv = []        
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1))

#### Generate the visual conveyor ##########################################################################################################

    def generate_env(self, no_of_gtp, no_of_output):
        """returns empty env, with some variables about the size, lanes etc."""
        empty = np.zeros((15, 4*no_of_gtp + 4 + 3*no_of_output +2, 3))                   # height = 15, width dependent on amount of stations
        for i in range(2,empty.shape[1]-2):
            empty[2][i]=(255,255,255)                               #toplane = 2
            empty[7][i]=(255,255,255)                               #bottom lane = 7
        for i in range(2,8):
            empty[i][1]=(255,255,255)                               #left lane
            empty[i][empty.shape[1]-2]=(255,255,255)                #right lane

        for i in range(8,empty.shape[0]): 
            for j in range(4,no_of_gtp*4+1,4):                      #GTP lanes
                empty[i][j] = (255,255,255)                         #Gtp in
                empty[i][j-1] = (250, 250,250)                      #gtp out
            for j in range(empty.shape[1]-no_of_output*2-1,empty.shape[1]-2,2):   #order carrier lanes
                empty[i][j] = (255,255,255)
        for i in range(4,no_of_gtp*4+1, 4):                         #divert and merge points
            empty[7][i-1] = (255, 242, 229)                         #merge
            empty[7][i] = (255, 242, 229)                           #divert

        for i in range(empty.shape[1]-no_of_output*2-1,empty.shape[1]-2,2):       #output points
            empty[7][i] = (255, 242, 229)
        
        for i in range(8,empty.shape[0],1):                                     #order carriers available in lanes
            for j in range(empty.shape[1]-no_of_output*2-1,empty.shape[1]-2,2):
                x= empty.shape[1]-no_of_output*2-1
                empty[i][j] = self.pallette[int((j-x)*0.5)]

        return empty

###### HELPER FUNCTIONS ############################################################################################################################
    def get_candidate_lists(self, list_lists, x):
        """Returns all lists, starting with x in list of lists"""
        return [nestedlist for nestedlist in list_lists if nestedlist[0] == x]

    def len_longest_sublist(self, listoflists):
        """returns length of the longest list in the sublists"""
        try:
            return max([len(sublist) for sublist in listoflists]) 
        except:
            return 0

    def len_shortest_sublist(self, listoflists):
        """returns length of the shortest list in the sublists"""
        return min([len(sublist) for sublist in listoflists])
        
    def update_queues(self, quenr, variable):
        'For a given queue 1-3, add a variable (1,2,3)'
        for i in range(self.amount_of_gtps):
            if quenr == i+1:
                self.init_queues[i].append(variable)
            
    def remove_from_queue(self, quenr):
        'For a given queue 1-3, remove the first in the queue'
        for i in range(self.amount_of_gtps):
            if quenr == i+1:
                self.init_queues[i]= self.init_queues[i][1:]

    def add_to_in_que(self, que_nr, to_add):
        'for a given queue, add item to the queue'
        self.in_queue[que_nr].append(to_add)

    
    def simulate_operator_action(self):
        'processes an item at all the GTP stations, currently just accepts the item allways'
        self.items_on_conv = [sublist for sublist in self.items_on_conv if sublist[0] not in self.operator_locations]
 ########################################################################################################################################################       
 ## RESET FUNCTION 
 #            
    def reset(self):
        "reset all the variables to zero, empty queues"
        self.D_states = {}
        for i in range(1,len(self.diverter_locations)+1):
            self.D_states[i] = False
        
        #initialize output points
        self.O_states = {}
        for i in range(1,len(self.output_locations)+1):
            self.O_states[i] = 0
        
        # initialize transition points: 0=no transition, 1=transition
        self.T_states = {}
        for i in range(1,len(self.operator_locations)+1):
            self.T_states[i] = False

        #initialize merge points
        self.M_states = {}
        for i in range(1,len(self.merge_locations)+1):
            self.M_states[i] = False

####### FOR SIMULATION ONLY 
        self.W_times = {}
        for i in range(1,len(self.operator_locations)+1):
            self.W_times[i] = self.process_time_at_GTP + 150 + 8*self.amount_of_gtps  + randint(-10, 10)
        print("Process times at operator are:", self.W_times)
####### FOR SIMULATION ONLY

        #empty amount of items on conv.
        self.items_on_conv = []
        
        
        self.init_queues = self.queues
        self.demand_queues = copy(self.queues)
        self.in_queue = [[] for item in range(len(self.queues))]
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1))

########################################################################################################################################################
## PROCESSING OF ORDER CARRIERS AT GTP
# 
    def process_at_GTP(self):
        # for each step; check if it needed to process an order carrier at GTP
        O_locs = copy(self.operator_locations)
        for Transition_point in O_locs:
            if self.W_times[O_locs.index(Transition_point)+1] == 0:
                print('Processing time at GTP {} is 0, check done on correctness:'.format(O_locs.index(Transition_point)+1))
                if random.random() < self.exception_occurence: #if the random occurence is below exception occurence (set in config) do:
                    #move order carrier at transition point to the merge lane
                    print('not the right order carrier, move to merge lane')
                    for item  in self.items_on_conv:
                        if item[0] == Transition_point:
                            item[0][0] -=1
                    
                else:
                    #remove the order form the items_on_conv
                    print('right order carrier is at GTP (location: {}'.format(Transition_point))
                    #print('conveyor memory before processing: ', self.items_on_conv)
                    self.items_on_conv = [item for item in self.items_on_conv if item[0] !=Transition_point]
                    print('order at GTP {} processed'.format(O_locs.index(Transition_point)+1))
                    #print('conveyor memory after processing: ', self.items_on_conv)

                #set new timestep for the next order
                try: 
                    next_type = [item[1] for item in env.items_on_conv if item[0] == [12,14]][0]
                except:
                    next_type = 99
                self.W_times[O_locs.index(Transition_point)+1] = self.process_time_at_GTP if next_type == 1 else self.process_time_at_GTP+30 if next_type == 2 else self.process_time_at_GTP+60 if next_type == 3 else self.process_time_at_GTP+60 if next_type == 4 else self.process_time_at_GTP+60
                print('new timestep set at GTP {} : {}'.format(O_locs.index(Transition_point)+1, self.W_times[O_locs.index(Transition_point)+1]))

                #remove from in_queue when W_times is 0
                try:
                    #remove item from the In_que list
                    self.in_queue[O_locs.index(Transition_point)] = self.in_queue[O_locs.index(Transition_point)][1:]
                    print('item removed from in-que')
                except:
                    print("Except: queue was already empty!")
            elif self.W_times[O_locs.index(Transition_point)+1] < 0:
                self.W_times[O_locs_locations.index(Transition_point)+1] = 0
                print("Waiting time was below 0, reset to 0")
            else:
                self.W_times[O_locs.index(Transition_point)+1] -= 1 #decrease timestep with 1
                print('waiting time decreased with 1 time instance')
                print('waiting time at GTP{} is {}'.format(O_locs.index(Transition_point)+1, self.W_times[O_locs.index(Transition_point)+1]))
            

########################################################################################################################################################
## STEP FUNCTION
#
    def step_env(self):

####make carrier type map
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1)).astype(int)
        for item in self.items_on_conv:
            self.carrier_type_map[item[0][1]][item[0][0]] = item[1]
            item[2] +=1

#### Process the orders at GTP > For simulation: do incidental transfer of order carrier
        self.process_at_GTP()

####toggle diverters if needed
        #toggle All D_states if needed:
        d_locs = copy(self.diverter_locations)
        carrier_map = copy(self.carrier_type_map)
        for loc2 in d_locs:
            try:
                #Condition 1 = if the carrier type at any of the diverter locations is EQUAL TO the next-up requested carrier type at GTP request lane of this specific diverter location
                condition_1 = carrier_map[loc2[1]][loc2[0]] == self.init_queues[d_locs.index(loc2)][0]
                #condition 2 = if the lenght of the in_queue is <= smallest queue that also demands order carrier of the same type
                condition_2 = len(self.in_queue[d_locs.index(loc2)]) <= min(map(len, self.in_queue)) 
                if condition_1 and condition_2: 
                    self.D_states[d_locs.index(loc2)+1] = True
                    print("set diverter state for diverter {} to TRUE".format(d_locs.index(loc2)+1))
                    self.remove_from_queue(d_locs.index(loc2)+1)
                    print("request removed from demand queue")
                    self.add_to_in_que(d_locs.index(loc2),int(carrier_map[loc2[1]][loc2[0]]))
                    print("Order carrier added to GTP queue")

                else:
                    self.D_states[d_locs.index(loc2)+1] = False
                    print("Divert-set requirement not met at cord :", loc2)
            except IndexError:
                print('Index error: queues are empty?')
                self.D_states[d_locs.index(loc2)+1] = False
            except:
                print('Another error occurred; this should not happen! Investigate the cause!')
                self.D_states[d_locs.index(loc2)+1] = False



        

####Divert when at diverter, and diverter is set to true
####Do step for all items on conv
        for item in self.items_on_conv:
            #check diverters; is any of them True and is there an order carrier? move into lane.
            if item[0] in self.diverter_locations:
                try:
                    if self.D_states[self.diverter_locations.index(item[0])+1] == True and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0:
                        item[0][1] +=1
                        print("moved order carrier into GTP lane {}".format(self.diverter_locations.index(loc1)+1))
                    else:
                        item[0][0] -=1 
                except:
                    print("Item of size {} not moved into lane: Divert value not set to true".format(item[1])) 

            #otherwise; all items set a step in their moving direction 
            elif item[0][1] == 7 and item[0][0] > 1 and self.carrier_type_map[item[0][1]][item[0][0]-1] ==0: #if on the lower line, and not reached left corner:
                item[0][0] -=1                     #move left
                print('item {} moved left'.format(item[0]))
            elif item[0][0] ==1 and item[0][1] >2 and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0: #if on left lane, and not reached top left corner:
                item[0][1] -=1
                print('item {} moved up'.format(item[0]))                    #move up
            elif item[0][1] == 2 and item[0][0] < self.empty_env.shape[1]-2 and self.carrier_type_map[item[0][1]][item[0][0]+1] ==0: #if on the top lane, and not reached right top corner:
                item[0][0] +=1                      #Move right
                print('item {} moved right'.format(item[0]))
            elif item[0][0] == self.empty_env.shape[1]-2 and item[0][1] <7 and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0: #if on right lane, and not reached right down corner:
                item[0][1] +=1
                print('item {} moved down'.format(item[0]))
            elif item[0][1] > 7 and item[0][0] in [lane[0] for lane in self.diverter_locations] and item[0][1] < self.empty_env.shape[0]-1 and item[0][0] < self.amount_of_gtps*4+3 and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0: #move down into lane
                item[0][1] +=1
                print('item {} moved into lane'.format(item[0]))
            elif item[0][1] > 7 and item[0][0] in [lane[0] for lane in self.merge_locations] and item[0][0] < self.amount_of_gtps*4+3 and self.carrier_type_map[item[0][1]-1][item[0][0]+1] ==0: #move up into merge lane
                item[0][1] -=1
            elif item[0][1] > 7 and item[0][0] > self.amount_of_gtps*4+3 and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0: #move up if on output lane
                item[0][1] -=1
                print('item {} moved onto conveyor'.format(item[0]))

        ####try to add new item from output when On!=0
        for cord2 in self.output_locations:
            loc =copy(cord2)
            if self.O_states[self.output_locations.index(loc)+1] !=0 and self.carrier_type_map[cord2[1]][cord2[0]+1] ==0:
                self.items_on_conv.append([loc,self.output_locations.index(loc)+1,0])
                self.O_states[self.output_locations.index(loc)+1] -=1
                print("Order carrier outputted at {}".format(loc))
                print("Items on conveyor: ", self.items_on_conv)
            else:
                print('No order carrier output on output {} .'.format(loc))

    def make_observation(self):
        '''Builds the observation from the available variables'''
        in_queue = []
        for item in self.in_queue:
            in_queue.append(item + [0]*(10-len(item)))
        in_queue = np.array(in_queue).flatten()
        demand_que = np.array([item[:10] for item in self.demand_queues]).flatten()
        carrier_type_map_obs = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],2)).astype(int)
        for item in self.items_on_conv:
            carrier_type_map_obs[item[0][1]][item[0][0]][0] = item[1]
            carrier_type_map_obs[item[0][1]][item[0][0]][1] = item[2]
        obs_queues = np.append(in_queue, demand_que)
        obs = np.append(obs_queues, carrier_type_map_obs.flatten())
        return obs

    def step(self, action):
        if action==0:
            self.step_env()
            print("- - action 0 executed")
            print("Divert locations :", self.diverter_locations)
            print('states of Divert points = {}'.format(self.D_states))
        elif action ==1: 
            self.O_states[1] +=1
            self.step_env()
            print("- - action 1 executed")
        elif action ==2:
            self.O_states[2] +=1
            self.step_env()
            print("- - action 2 executed")
        elif action ==3:
            self.O_states[3] +=1
            self.step_env()
            print("- - action 3 executed")
        elif action ==4:
            self.O_states[4] +=1
            self.step_env()
        elif action ==5:
            self.O_states[5] +=1
            self.step_env()
        elif action ==6:
            self.O_states[6] +=1
            self.step_env()

        print("states of O: ",self.O_states)
        print('init queues :', self.init_queues)
        print('conveyor memory : ', self.items_on_conv)
        print('')
        print('--------------------------------------------------------------------------------------------------------------------')

        next_state = self.make_observation()
        reward = self.reward
        terminate = self.terminate
        info = ''
        return next_state, reward, terminate, info

   

################## RENDER FUNCTIONS ################################################################################################
    def render_plt(self):
        """Simple render function, uses matplotlib to render the image + some additional information on the transition points"""
        print('items on conveyor:')
        print(self.items_on_conv)
        print('states of Divert points = {}'.format(self.D_states))
        print('states of Output points = {}'.format(self.O_states))
        for queue in self.init_queues:
            print('Queue GTP{}: {}'.format(self.init_queues.index(queue), queue))

        image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([self.pallette[0] if item[1] ==1 else self.pallette[1] if item[1] ==2 else self.pallette[2] if item[1] ==2 else self.pallette[3]]) 
        plt.imshow(np.asarray(image))
        plt.show()
    

    def render(self):
        """render with opencv, for faster processing"""
        image = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([self.pallette[0] if item[1] ==1 else self.pallette[1] if item[1] ==2 else self.pallette[2] if item[1] ==3 else self.pallette[3]]) 

        #resize with PIL
        im = Image.fromarray(np.uint8(image))
        img = im.resize((1200,480), resample=Image.BOX) #BOX for no anti-aliasing)
        cv2.imshow("Simulation-v0.1", cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(10)
    

############### MAIN ##############################################################################################################################################

## Test the item
#queues = [[1,2,3,2,3], [2,3,1,3,1], [1,3,2,1,2], [1,3,2,1,2], [1,3,2,1,2]] #sample queues for format WHERE 1=S, 2=M, 3=L
amount_gtp = 15
amount_output = 4
buffer_size = 10
queues = [[randint(1,amount_output) for i in range(buffer_size)] for item in range(amount_gtp)] # generate random queues
print(queues)
env = simple_conveyor(queues, amount_gtp, amount_output)
env.reset()

#Build action list according to FIFO and Round-Robin Policy
order_list = []
for index in range(len(env.queues[0])):
    order_list.append([item[index] for item in env.queues])

#flat_list = [item for sublist in l for item in sublist]
order_list = [item for sublist in order_list for item in sublist]
print("Resulting in sequence of actions: ", order_list)

#run short trail:
env.reset()

for item in order_list:
    env.step(item)
    env.render()
    env.step(0)


    
while env.in_queue != [[] * i for i in range(amount_gtp)]:
    env.step(0)
    env.render()
