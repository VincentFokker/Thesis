from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#colors used
large_c = '#b7472a'
medium_c= '#eb9788'
small_c = '#ffd5ab'
conveyor = '#fff2e5' 

#size render frame
height = 600
width = 1000

# dimensions
width_conv  =(0.05)*height
topleft_c = (.45/5)*width
top_right = (4.45/5)*width
down_right = (1.55/3)*height
length_input = (2.55/3)*height
large_b = width_conv-2
small_b = large_b/3
medium_b = large_b/2
tote_dist = (0.05/3)* height

#speed
speed = 5

def render_env():
    """Render in the background of the conveyer"""
    #create the background
    img_array = np.zeros((height, width, 3))

    #transform to PIL object
    im = Image.fromarray(img_array, mode="RGB")

    #draw the system
    draw = ImageDraw.Draw(im)

    #Conveyor
    draw.rectangle((topleft_c,topleft_c,top_right+width_conv,topleft_c+width_conv)) #A
    draw.rectangle((top_right,topleft_c,top_right+width_conv,down_right)) #B
    draw.rectangle((topleft_c,down_right-width_conv,top_right+width_conv,down_right)) #C, D, F, G, H
    draw.rectangle((topleft_c,topleft_c,topleft_c+width_conv,down_right)) #I

    #out of system
    draw.rectangle((0, down_right-width_conv, topleft_c+width_conv, down_right)) #n10
    draw.rectangle((top_right, down_right-width_conv, width, down_right)) #n10

    #conveyors to GTP
    draw.rectangle(((2/5)*width-0.5*width_conv, down_right-width_conv, (2/5)*width+0.5*width_conv, length_input))
    draw.rectangle(((1.5/5)*width-0.5*width_conv, down_right-width_conv, (1.5/5)*width+0.5*width_conv, length_input))
    draw.rectangle(((1/5)*width-0.5*width_conv, down_right-width_conv, (1/5)*width+0.5*width_conv, length_input))

    #convey from carton storage
    draw.rectangle(((4/5)*width-0.5*width_conv, down_right-width_conv, (4/5)*width+0.5*width_conv, length_input)) #n4
    draw.rectangle(((3.5/5)*width-0.5*width_conv, down_right-width_conv, (3.5/5)*width+0.5*width_conv, length_input)) #n5
    draw.rectangle(((3/5)*width-0.5*width_conv, down_right-width_conv, (3/5)*width+0.5*width_conv, length_input)) #n6

    #transition points
    draw.rectangle((topleft_c, topleft_c, topleft_c+width_conv, topleft_c+width_conv), fill=conveyor) #n1
    draw.rectangle((top_right, topleft_c, top_right+width_conv, topleft_c+width_conv), fill=conveyor) #n2
    draw.rectangle((top_right, down_right-width_conv, top_right+width_conv, down_right), fill=conveyor) #n3
    draw.rectangle((topleft_c, down_right-width_conv, topleft_c+width_conv, down_right), fill=conveyor) #n10

    for x in [(1/5)*width, (1.5/5)*width, (2/5)*width,(3/5)*width, (3.5/5)*width, (4/5)*width]:
        draw.rectangle((x-0.5*width_conv, down_right-width_conv, x+0.5*width_conv, down_right), fill=conveyor) #n4

    ## in storage totes
    # small
    small_totes = 10
    for i in range(small_totes):
        draw.rectangle(((4/5)*width-0.5*small_b, down_right+tote_dist+i*(tote_dist+small_b), (4/5)*width+0.5*small_b, down_right+tote_dist+small_b+i*(tote_dist+small_b)), fill=small_c)

    #medium
    medium_totes = 8
    for i in range(medium_totes):
        draw.rectangle(((3.5/5)*width-0.5*medium_b, down_right+tote_dist+i*(tote_dist+medium_b), (3.5/5)*width+0.5*medium_b, down_right+tote_dist+medium_b+i*(tote_dist+medium_b)), fill=medium_c)

    #large
    large_totes = 5
    for i in range(large_totes):
        draw.rectangle(((3/5)*width-0.5*large_b, down_right+tote_dist+i*(tote_dist+large_b), (3/5)*width+0.5*large_b, down_right+tote_dist+large_b+i*(tote_dist+large_b)), fill=large_c)
    
    return im

steps = 2000
start_location = ((3/5)* width, (1.5/3)*height)
current_x = start_location[0]
current_y = start_location[1]


# run a loop
for step in range(steps):
    im = render_env()
    draw = ImageDraw.Draw(im)
    
    #define action space
    if current_y == (1.5/3)*height and current_x > (0.5/5)* width:
        current_x = current_x -speed
    elif current_x ==(0.5/5)* width and current_y > (0.5/3) *height:
        current_y = current_y - speed
    elif current_y == (0.5/3)*height and current_x < (4.5/5)* width:
        current_x = current_x + speed
    elif current_x == (4.5/5)* width:
        current_y = current_y + speed
    else:
        pass
    
    #draw the new location:
    current_coordinates = (current_x - (1/2)*medium_b, current_y - (1/2)*medium_b, current_x+ (1/2)*medium_b, current_y + (1/2) * medium_b)
    #current_coordinates = (current_x, current_y - small_b, current_x+ small_b, current_y)
    draw.rectangle(current_coordinates, fill=medium_c)
    cv2.imshow("Simulation-v0.1", np.array(im))
    cv2.waitKey(10)

    





