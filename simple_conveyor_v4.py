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
        self.init_queues(self.queues)
       
        #init config
        self.amount_of_gtps = 3
        self.amount_of_outputs = 3
        self.empty_env = self.generate_env(self.amount_of_gtps)
        
        #define where the operators of the GTP stations are
        self.operator_locations = [[4,12], [8,12], [12,12]]
        self.output_locations = [[7,19], [7,22], [7,25]]
        self.diverter_locations = [[7,12], [7,8], [7,4]]
        
        #initialize divert points: 0=no diversion, 1=diversion
        self.D1 = False
        self.D2 = False
        self.D3 = False
        
        #initialize output points
        self.O1 = 0 #output of box size S 
        self.O2 = 0 #output of box size M
        self.O3 = 0 #output of box size L

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

        #intialize variables for later usage
        self.O1_location = False
        self.O2_location = False
        self.O3_location = False
        
        self.carrier_type_map = np.zeros((13,31,1))

    def generate_env(self, no_of_gtp):
        """returns empty env, with some variables about the size, lanes etc."""
        empty = np.zeros((13, 19+4*no_of_gtp, 3))
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
    
    def simulate_operator_action(self):
        'processes an item at all the GTP stations, currently just accepts the item allways'
        self.items_on_conv = [sublist for sublist in self.items_on_conv if sublist[0] not in self.operator_locations]
 ########################################################################################################################################################       
 ## RESET FUNCTION 
 #            
    def reset(self):
        "reset all the variables to zero, empty queues"
        self.D1 = False
        self.D2 = False
        self.D3 = False
        
        #initialize output points
        self.O1 = 0 #output of box size S 
        self.O2 = 0 #output of box size M
        self.O3 = 0 #output of box size L
        
        #empty amount of items on conv.
        self.items_on_conv = []
        
        self.carrier_type_map = np.zeros((13,31,1))
        self.init_queues(self.queues)
        self.empty_env = self.generate_env(self.amount_of_gtps)

########################################################################################################################################################
## STEP FUNCTION
#
    def step_env(self):

####make carrier type map
        self.carrier_type_map = np.zeros((13,31,1))
        for item in self.items_on_conv:
            self.carrier_type_map[item[0][1]][item[0][0]] = item[1]

####toggle diverters if needed
        #toggle D1 if needed:
        try:
            if self.carrier_type_map[7][12] == self.queue1[0] and len(self.queue1) >= self.len_longest_sublist(self.get_candidate_lists([self.queue1, self.queue2, self.queue3], self.carrier_type_map[7][12])):
                self.D1 = True
                self.remove_from_queue(1)
            else:
                self.D1 = False
        except:
            self.D1=False
        
        #toggle D2 if needed:    
        try:
            if self.carrier_type_map[7][8] == self.queue2[0] and len(self.queue2) >= self.len_longest_sublist(self.get_candidate_lists([self.queue1, self.queue2, self.queue3], self.carrier_type_map[7][8])):
                self.D2 = True
                self.remove_from_queue(2)
            else:
                self.D2 = False
        except:
            self.D2 = False
        
        #toggle D3 if needed:
        try:
            if self.carrier_type_map[7][4] == self.queue3[0]:  #and len(self.queue3) >= self.len_longest_sublist(self.get_candidate_lists([self.queue1, self.queue2, self.queue3], self.carrier_type_map[7][4])):
                self.D3 = True
                self.remove_from_queue(3)
            else:
                self.D3 = False
        except:
            self.D3 = False

####Divert when at diverter, and diverter is set to true
####Do step for all items on conv
        for item in self.items_on_conv:
            #check diverters; is any of them True and is there an order carrier? move into lane.
            if item[0] == [12,7] and self.D1 == True:
                item[0][1] +=1
            elif item[0] == [8,7] and self.D2 == True:
                item[0][1] +=1
            elif item[0] == [4,7] and self.D3 == True:
                item[0][1] +=1
            
            #otherwise; all items set a step in their moving direction 
            elif item[0][1] == 7 and item[0][0] > 2 and self.carrier_type_map[item[0][1]][item[0][0]-1] ==0: #if on the lower line, and not reached left corner:
                item[0][0] -=1                     #move left
            elif item[0][0] ==2 and item[0][1] >2 and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0: #if on left lane, and not reached top left corner:
                item[0][1] -=1                     #move up
            elif item[0][1] == 2 and item[0][0] <28 and self.carrier_type_map[item[0][1]][item[0][0]+1] ==0: #if on the top lane, and not reached right top corner:
                item[0][0] +=1                      #Move right
            elif item[0][0] == 28 and item[0][1] <7 and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0: #if on right lane, and not reached right down corner:
                item[0][1] +=1
            elif item[0][1] > 7 and item[0][1] < 12 and item[0][0] < 15 and self.carrier_type_map[item[0][1]+1][item[0][0]] ==0:
                item[0][1] +=1
            elif item[0][1] > 7 and item[0][0] > 15 and self.carrier_type_map[item[0][1]-1][item[0][0]] ==0:
                item[0][1] -=1

        ####try to add new item from output when On!=0
        if self.O1 !=0 and self.carrier_type_map[7][20] == 0:
            self.items_on_conv.append([[19,7], 1])
            self.O1 -=1
        
        elif self.O2 !=0 and self.carrier_type_map[7][23] == 0:
            self.items_on_conv.append([[22,7], 2])
            self.O2 -=1

        elif self.O3 !=0 and self.carrier_type_map[7][26] == 0:
            self.items_on_conv.append([[25,7], 3])
            self.O3 -=1

    def step(self, action):
        if action==0:
            self.step_env()
        elif action ==1: 
            self.O1 +=1
            self.step_env()
        elif action ==2:
            self.O2 +=1
            self.step_env()
        elif action ==3:
            self.O3 +=1
            self.step_env()

   

################## RENDER FUNCTIONS ################################################################################################
    def render_plt(self):
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
        try:
            print(self.len_shortest_sublist(self.get_candidate_lists([self.queue1, self.queue2, self.queue3], self.carrier_type_map[7][12])))
        except:
            print('no item to merge argmin error')
        df = pd.read_csv('representation3.csv', delimiter=';', ).fillna(0)
        listoflists = df.values.tolist()
        image = np.asarray([[(255,255,255) if x =='x' else (220,220,220) if x =='y' else (0,0,0) for x in item] for item in listoflists])

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([(255, 213, 171) if item[1] ==1 else (235, 151, 136) if item[1] ==2 else (183, 71, 42)]) 
        plt.imshow(np.asarray(image))
        plt.show()
    
    def render1(self):
        """render with opencv, for faster processing"""
        df = pd.read_csv('representation3.csv', delimiter=';', ).fillna(0)
        listoflists = df.values.tolist()
        image = np.asarray([[(255,255,255) if x =='x' else (220,220,220) if x =='y' else (0,0,0) for x in item] for item in listoflists])

        for item in self.items_on_conv:
            image[item[0][1]][item[0][0]] = np.asarray([(255, 213, 171) if item[1] ==1 else (235, 151, 136) if item[1] ==2 else (183, 71, 42)]) 
    

        #resize with PIL
        im = Image.fromarray(np.uint8(image))
        img = im.resize((600,240), resample=Image.BOX) #BOX for no anti-aliasing)
        cv2.imshow("Simulation-v0.1", np.array(img))
        cv2.waitKey(0)

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


    
while env.queue1 + env.queue2 + env.queue3 != []:
    env.step(0)
    env.render()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break