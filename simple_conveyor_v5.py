import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.colors import ColorConverter

class simple_conveyor():

######## INITIALIZATION OF VARIABLES ###############################################################################################################
    def __init__(self, queues):
        """initialize states of the variables, the lists used"""
        #init queues
        self.queues = queues
        self.init_queues = self.queues

        #init config
        self.amount_of_gtps = 3
        self.amount_of_outputs = 3
        self.empty_env = self.generate_env(self.amount_of_gtps)

        
        #define where the operators of the GTP stations are
        self.operator_locations = [[i, 12] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        self.output_locations = [[self.empty_env.shape[1]-i*3-6,7]for i in range(self.amount_of_outputs)][::-1]
        self.diverter_locations = [[i, 7] for i in range(4,self.amount_of_gtps*4+1,4)][::-1]
        print("operator locations: ", self.operator_locations)
        print("output locations: ", self.output_locations)
        print("diverter locations: ", self.diverter_locations)
        #initialize divert points: 0=no diversion, 1=diversion
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

    def generate_env(self, no_of_gtp):
        """returns empty env, with some variables about the size, lanes etc."""
        empty = np.zeros((15, 19+4*no_of_gtp, 3))
        for i in range(2,empty.shape[1]-2):
            empty[2][i]=(255,255,255)
            empty[7][i]=(255,255,255)
        for i in range(2,8):
            empty[i][2]=(255,255,255)
            empty[i][empty.shape[1]-2]=(255,255,255)
        for i in range(8,empty.shape[0]): 
            for j in range(4,no_of_gtp*4+1,4):
                empty[i][j] = (255,255,255)
                empty[i][j-1] = (250, 250,250)
            for j in range(empty.shape[1]-12,empty.shape[1]-5,3):
                empty[i][j] = (255,255,255)
        for i in range(4,no_of_gtp*4+1, 4):
            empty[7][i-1] = (255, 242, 229)
            empty[7][i] = (255, 242, 229)

        for i in range(empty.shape[1]-12,empty.shape[1]-5,3):
            empty[7][i] = (255, 242, 229)

        return empty

###### HELPER FUNCTIONS ############################################################################################################################
    def get_candidate_lists(self, list_lists, x):
        """Returns all lists, starting with x in list of lists"""
        return [nestedlist for nestedlist in list_lists if nestedlist[0] == x]

    def len_longest_sublist(self, listoflists):
        """returns length of the longest list in the sublists"""
        return max([len(sublist) for sublist in listoflists])

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
                self.init_queues[0]= self.init_queues[1:]

    
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
        self.empty_env = self.generate_env(self.amount_of_gtps)
        self.carrier_type_map = np.zeros((self.empty_env.shape[0],self.empty_env.shape[1],1))

########################################################################################################################################################
## STEP FUNCTION
#
    def step_env(self):

####make carrier type map
        self.carrier_type_map = np.zeros((env.empty_env.shape[0],env.empty_env.shape[1],1))
        for item in self.items_on_conv:
            self.carrier_type_map[item[0][1]][item[0][0]] = item[1]

####toggle diverters if needed
        #toggle All D_states if needed:
        for cord0 in self.diverter_locations:
            try:
                if len(self.init_queues[self.diverter_locations.index(cord0)]) >= self.len_longest_sublist(self.get_candidate_lists(self.init_queues, self.carrier_type_map[cord0[1]][cord0[0]])): #self.carrier_type_map[cord0[1]][cord0[0]] == self.init_queues[self.diverter_locations.index(cord0)][0] and
                    self.D_states[self.diverter_locations.index(cord0)+1] = True
                    print("set diverter state for diverter {} to TRUE".format(self.diverter_locations.index(cord0)+1))
                    self.remove_from_queue(self.diverter_locations.index(cord0)+1)
                    print("request removed from queue")

                else:
                    self.D_states[self.diverter_locations.index(cord0)+1] = False
                    print("catched in else statement")
            except ValueError:
                 self.D_states[self.diverter_locations.index(cord0)+1] = False
                 print("Value error, max() arg is an empty sequence")

        

####Divert when at diverter, and diverter is set to true
####Do step for all items on conv
        for item in self.items_on_conv:
            #check diverters; is any of them True and is there an order carrier? move into lane.
            for cord1 in self.diverter_locations:
                if item[0] == cord1 and self.D_states[self.diverter_locations.index(cord1)+1] == True:
                    item[0][1] +=1
                    print("moved order carrier into GTP lane {}".format(self.diverter_locations.index(cord1)+1))
        
        for item in self.items_on_conv:
            #otherwise; all items set a step in their moving direction 
            if item[0][1] == 7 and item[0][0] > 2 and self.carrier_type_map[item[0][1]][item[0][0]-1] ==0: #if on the lower line, and not reached left corner:
                item[0][0] -=1                     #move left
                print('item {} moved left'.format(item[0]))
            elif item[0][0] ==2 and item[0][1] >2 and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0: #if on left lane, and not reached top left corner:
                item[0][1] -=1
                print('item {} moved up'.format(item[0]))                    #move up
            elif item[0][1] == 2 and item[0][0] < self.empty_env.shape[1]-3 and self.carrier_type_map[item[0][1]][item[0][0]+1] ==0: #if on the top lane, and not reached right top corner:
                item[0][0] +=1                      #Move right
                print('item {} moved right'.format(item[0]))
            elif item[0][0] == self.empty_env.shape[1]-3 and item[0][1] <7 and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0: #if on right lane, and not reached right down corner:
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
            if self.O_states[self.output_locations.index(cord2)+1] !=0 and self.carrier_type_map[cord2[1]][cord2[0]+1] ==0:
                self.items_on_conv.append([cord2,self.output_locations.index(cord2)+1])
                self.O_states[self.output_locations.index(cord2)+1] -=1
                print("Order carrier outputted at {}".format(cord2))
                print("Items on conveyor: ", self.items_on_conv)
            else:
                print(self.output_locations)

    def step(self, action):
        if action==0:
            self.step_env()
            print("- - action 0 executed")
        elif action ==1: 
            self.O_states[1] +=1
            self.step_env()
            print("- - action 1 executed")
            print("states of O: ",self.O_states)
        elif action ==2:
            self.O_states[2] +=1
            self.step_env()
            print("- - action 2 executed")
            print("states of O: ",self.O_states)
        elif action ==3:
            self.O_states[3] +=1
            self.step_env()
            print("- - action 3 executed")
            print("states of O: ",self.O_states)

   

################## RENDER FUNCTIONS ################################################################################################
    def render_plt(self):
        """Simple render function, uses matplotlib to render the image + some additional information on the transition points"""
        print('items on conveyor:')
        print(self.items_on_conv)
        print('states of Divert points = {}'.format(self.D_states))
        print('states of Output points = {}'.format(self.O_states))
        for queue in self.init_queues:
            print('Queue GTP{}: {}'.format(self.init_queues.index(queue), queue))

        image = self.generate_env(self.amount_of_gtps)

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([(255, 213, 171) if item[1] ==1 else (235, 151, 136) if item[1] ==2 else (183, 71, 42)]) 
        plt.imshow(np.asarray(image))
        plt.show()
    

    def render(self):
        """render with opencv, for faster processing"""
        image = self.generate_env(self.amount_of_gtps)

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([(255, 213, 171) if item[1] ==1 else (235, 151, 136) if item[1] ==2 else (183, 71, 42)]) 

        #resize with PIL
        im = Image.fromarray(np.uint8(image))
        img = im.resize((600,240), resample=Image.BOX) #BOX for no anti-aliasing)
        cv2.imshow("Simulation-v0.1", np.array(img))
        cv2.waitKey(0)

############### MAIN ##############################################################################################################################################

## Test the item
queues = [[1,2,3,2,3], [2,3,1,3,1], [1,3,2,1,2]] #sample queues for format WHERE 1=S, 2=M, 3=L
env = simple_conveyor(queues)
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


    
while env.init_queues != []:
    env.step(0)
    env.render()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break