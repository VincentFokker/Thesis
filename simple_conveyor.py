

class simple_conveyor():

    queues = [[1,2,3,2,3,1], [2,3,1,3,1,2], [1,3,2,1,2,2]] #sample queues for format WHERE 1=S, 2=M, 3=L

    def __init__(self, queues):
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
        self.queue1 = self.queues[0]
        self.queue2 = self.queues[1]
        self.queue3 = self.queues[2]

        #intialize variables for later usage
        self.O1_location = False
        self.O1_location = False
        self.O1_location = False

    def reset(self):
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
        self.O1_location = False
        self.O1_location = False

    def set_step(self):
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
                elif item[0][1] > 7 and item[0] not in [items_on_conv[item][0] for item in range(len(items_on_conv))] and item[0][1] <12:
                    item[0][1] +=1

        #update occupation-states of mergepoints
        if [19,7] in [items_on_conv[item][0] for item in range(len(items_on_conv))]:
            self.O1_location == True
        if [22,7] in [items_on_conv[item][0] for item in range(len(items_on_conv))]:
            self.O2_location == True
        if [25,7] in [items_on_conv[item][0] for item in range(len(items_on_conv))]:
            self.O3_location == True

    def perform_actions_in_space(self):
        #set step:
        self.set_step()
        
        ###output new order carrier(s)
        #if satisfied; output carrier type1
        if self.O1 == True and self.O1_location == False:
            self.O1_location ==True            #occupy the output
            self.O1 == False                   #turn output off
            self.items_on_conv.append([[19,7], 1])
        
        #if statisfied; output carrier type2
        elif self.O2 == True and self.O2_location == False:
            self.O2_location ==True            #occupy the output
            self.O2 == False                   #turn output off
            self.items_on_conv.append([[22,7], 2])
        
        #if satisfied, output carrier type3
        elif self.O3 == True and self.O3_location == False:
            self.O3_location ==True            #occupy the output
            self.O3 == False                   #turn output off
            self.items_on_conv.append([[25,7], 3])

    def step(self, action):
        if   action==0: self.perform_actions_in_space() #set a step for all items
        elif action==1: self.D1 = not self.D1 #toggle D1
        elif action==2: self.D2 = not self.D2 #Toggle D2
        elif action==3: self.D3 = not self.D3 #Toggle D3
        elif action==4: self.O1 = not self.O1 #toggle O1
        elif action==5: self.O2 = not self.O2 #toggle O2
        elif action==6: self.O3 = not self.O3 #toggle O3
        
        #todo:
        # - Fix returns
        # - build reward function
        # - termination criteria
        # - additional info

    def render(self):
        print('items on conveyor:')
        print(self.items_on_conv)
        print('states of Divert points:')
        print('D1 = {}, D2 = {}, D3 = {}'.format(self.D1, self.D2, self.D3))
        print('States of output points:')
        print('O1 = {}, O2 = {}, O3 = {}'.format(self.O1, self.O2, self.O3))
        
        
        
        #define shape
        # height = 200
        # width = 200
        # img_array = np.zeros((height, width, 3))

        # #transform to PIL object
        # im = Image.fromarray(img_array, mode="RGB")

        # #draw the system
        # draw = ImageDraw.Draw(im)

        # #draw rectangle
        # x,y = self.location
        # draw.rectangle([x-5,y-5,x+5,y+5], fill='#b7472a')

        # cv2.imshow("Simulation-v0.1", np.array(im))
        # cv2.waitKey(10)