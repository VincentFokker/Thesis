import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.colors import ColorConverter
from copy import copy
from random import randint
import seaborn as sns

class simple_conveyor():

######## INITIALIZATION OF VARIABLES ###############################################################################################################
    def __init__(self, queues, amount_gtp=3, amount_output=3):
        """initialize states of the variables, the lists used"""
        #init queues
        self.queues = queues
        self.init_queues = self.queues



        #init config
        self.amount_of_gtps = amount_gtp
        self.amount_of_outputs = amount_output
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)

        #colors
        self.pallette = (np.asarray(sns.color_palette("Reds", self.amount_of_outputs)) * 255).astype(int)

        #define where the operators, diverts and outputs of the GTP stations are
        self.operator_locations = [[i, 12] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        #self.output_locations = [[self.empty_env.shape[1]-i*3-6,7]for i in range(self.amount_of_outputs)][::-1]
        self.output_locations = [[i,7]for i in range(self.empty_env.shape[1]-self.amount_of_outputs*2-1,self.empty_env.shape[1]-2,2)]
        self.diverter_locations = [[i, 7] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        
        print("operator locations: ", self.operator_locations)
        print("output locations: ", self.output_locations)
        print("diverter locations: ", self.diverter_locations)
        #initialize divert points: False=no diversion, True=diversion
        self.D_states = {}
        for i in range(1,len(self.diverter_locations)+1):
            self.D_states[i] = False
        
        #initialize output points
        self.O_states = {}
        for i in range(1,len(self.output_locations)+1):
            self.D_states[i] = 0

        # initialize transition points: 0=no transition, 1=transition
        ## CURRENTLY NOT USED ##
        # T_states = {}
        # for i in range(1,len(self.diverter_locations)+1):
        #     T_states[i] = False

        # #initialize merge points
        # M_states = {}
        # for i in range(1,len(self.diverter_locations)+1):
        #     M_states[i] = False

        self.items_on_conv = []        
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1))

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
        
        #empty amount of items on conv.
        self.items_on_conv = []
        
        
        self.init_queues = self.queues
        self.empty_env = self.generate_env(self.amount_of_gtps, self.amount_of_outputs)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1))

########################################################################################################################################################
## STEP FUNCTION
#
    def step_env(self):

####make carrier type map
        self.carrier_type_map = np.zeros((env.empty_env.shape[0],env.empty_env.shape[1],1)).astype(int)
        for item in self.items_on_conv:
            self.carrier_type_map[item[0][1]][item[0][0]] = item[1]

####toggle diverters if needed
        #toggle All D_states if needed:
        d_locs = copy(self.diverter_locations)
        carrier_map = copy(self.carrier_type_map)
        queues_local = copy(self.init_queues)
        for loc2 in d_locs:
            #print('Carrier type map value :', carrier_map[loc2[1]][loc2[0]])
            #print('next up item in queue :', queues_local[d_locs.index(loc2)][0])
            try:
                if carrier_map[loc2[1]][loc2[0]] == self.init_queues[d_locs.index(loc2)][0]: #and len(self.init_queues[d_locs.index(loc2)]) >= self.len_longest_sublist(self.get_candidate_lists(self.init_queues, carrier_map[loc2[1]][loc2[0]])): 
                    self.D_states[d_locs.index(loc2)+1] = True
                    print("set diverter state for diverter {} to TRUE".format(d_locs.index(loc2)+1))
                    self.remove_from_queue(d_locs.index(loc2)+1)
                    print("request removed from queue")

                else:
                    self.D_states[d_locs.index(loc2)+1] = False
                    print("Divert-set requirement not met at cord :", loc2)
            except IndexError:
                print('Index error: queues are empty?')
                self.D_states[d_locs.index(loc2)+1] = False
            except:
                print('other error')
                self.D_states[d_locs.index(loc2)+1] = False

#### Toggle diverters 2
        # for cord0 in self.diverter_locations:
        #     dcord = copy(cord0)
        #     if self.carrier_type_map[dcord[1]][dcord[0]] == self.init_queues[self.diverter_locations.index(dcord)][0]:

        

####Divert when at diverter, and diverter is set to true
####Do step for all items on conv
        for item in self.items_on_conv:
            #check diverters; is any of them True and is there an order carrier? move into lane.
            if item[0] in self.diverter_locations:
                try:
                    if self.D_states[self.diverter_locations.index(item[0])+1] == True:
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
            elif item[0][1] > 7 and item[0][1] < self.empty_env.shape[0]-1 and item[0][0] < self.amount_of_gtps*4+3 and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0: #move down into lane
                item[0][1] +=1
                print('item {} moved into lane'.format(item[0]))
            elif item[0][1] > 7 and item[0][0] > self.amount_of_gtps*4+3 and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0:
                item[0][1] -=1
                print('item {} moved onto conveyor'.format(item[0]))

        ####try to add new item from output when On!=0
        for cord2 in self.output_locations:
            loc =copy(cord2)
            if self.O_states[self.output_locations.index(loc)+1] !=0 and self.carrier_type_map[cord2[1]][cord2[0]+1] ==0:
                self.items_on_conv.append([loc,self.output_locations.index(loc)+1])
                self.O_states[self.output_locations.index(loc)+1] -=1
                print("Order carrier outputted at {}".format(loc))
                print("Items on conveyor: ", self.items_on_conv)
            else:
                print('No order carrier output on output {} .'.format(loc))

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
        print("states of O: ",self.O_states)
        print('init queues :', self.init_queues)
        print('--------------------------------------------------------------------------------------------------------------------')

   

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
        cv2.waitKey(25)

############### MAIN ##############################################################################################################################################

## Test the item
#queues = [[1,2,3,2,3], [2,3,1,3,1], [1,3,2,1,2], [1,3,2,1,2], [1,3,2,1,2]] #sample queues for format WHERE 1=S, 2=M, 3=L
amount_gtp = 6
amount_output = 3
buffer_size = 5
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
    for _ in range(2):
        env.step(0)
        env.render()


    
while env.init_queues != [[] * i for i in range(amount_gtp)]:
    env.step(0)
    env.render()
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break