
import numpy as np 
import cv2

def isrgb( image ):
    return len(image.shape)==3 and image.shape[2]==3 

def to_rgb( image ):
    #to rgb
    if not isrgb( image ):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def norm(x):    
    x = to_rgb(x)
    for i in range( 3 ):
        x[:,:,i] = x[:,:,i] - x[:,:,i].min()
        x[:,:,i] = x[:,:,i] / x[:,:,i].max()
    return (x).astype(float) 

