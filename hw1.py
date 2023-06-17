#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:19:43 2022

@author: vaibhaw
"""

from IPython import display
import matplotlib.pyplot as plt
import numpy as np

def load_image(filename):
    img = np.load(filename)
    img = img.astype("float32") / 255.
    return img

def gray2rgb(image):
    return np.repeat(np.expand_dims(image, 2), 3, axis=2)

def show_image(img):
    plt.imshow(img, interpolation='nearest')
    
images = [load_image('red.npy'),
          load_image('green.npy'),
          load_image('blue.npy')]
print("Showing images. axis 1.")
show_image(gray2rgb(np.concatenate(images, axis=1)))    

print("Showing unaligned image.")
show_image(np.stack(images, axis=-1))

# Store the height and width of the images
height, width = images[0].shape

# Pad each image with black by 30 pixels. You do not need to use this, but
# padding may make your implementation easier.
pad_size = 30
images_pad = [np.pad(x, pad_size, mode='constant') for x in images]
show_image(np.stack(images_pad, axis=-1))

# Given two matrices, write a function that returns a number of how well they are aligned.
# The lower the number, the better they are aligned. There are a variety of scoring functions
# you can use. The simplest one is the Euclidean distance between the two matrices. 
# QES: What is Euclidean distance for pixels
# QES: What is the level of perfection required, 
#      only Euclidean dist will not be able to achivev the perfect alignment
#      We will need to use other techniquest like manhattan norm, norm cross-corelation
# <POST> Cosie similarity, mahatten distance, principle component analysis.
#         quantify
# QES : What are other distance meaaurement techniques - chessboard, cityblock etc
def score_function(im1, im2):
    # ======================
    # You code here

    #method 1: eucledian distance
    newArr = np.subtract (im1,im2) #diff in color of same pixel location
    newArrSq = np.square (newArr) #square of diff
    newArrSqSum = np.sum(newArrSq) #summ of all squared
    eucDist = np.sqrt(newArrSqSum)
    return eucDist
    '''

    #method 2: manhattan norm
    rg1 = im1.max()-im1.min()
    im1_min = im1.min()
    im1n = (im1-im1_min)/rg1
    rg2 = im2.max()-im2.min()
    im2_min = im2.min()
    im2n = (im2-im2_min)/rg2
    
    diffArr = np.subtract (im1n,im2n)
    absArr = np.abs(diffArr)
    sumArr = np.sum(absArr)
    return sumArr

    #method 3: median
    diffArr = np.subtract (im1,im2)
    absArr = np.abs(diffArr)
    median = np.median(absArr)
    return median
    '''

    # ======================

# Given two matrices chan1 and chan2, return a tuple of how to shift chan2 into chan1. This
# function should search over many different shifts, and find the best shift that minimizes
# the scoring function defined above. 
def align_channels(chan1, chan2):
    best_offset = (0,0)
    best_score = np.inf #infinity constant
    # ======================
    # You code here
    # Hint: you can first define a callable function to shift the images, 
    # which will make your code clean in for-loops.
    # QES : Why do you require to call function to shift image
    
    #This is a window of pixels that line between the image
    #The channel alignment will be checked only in this window
    kernel = [200,200,200,200]

    for dy in range(-pad_size,pad_size):
        for dx in range(-pad_size,pad_size):
            y1,x1 = kernel[0],kernel[1]
            y2,x2 = kernel[2],kernel[3]
            #im1 = chan1[30:-30,30:-30]
            #im2 = chan2[30+dy:-30+dy,30+dx:-30+dx]
            #im1 = chan1[30+y1    :-30-y2   ,30+x1   :-30-x2   ]
            #im2 = chan2[30+y1+dy :-30-y2+dy,30+x1+dx:-30-x2+dx]
            im1 = chan1[pad_size + y1 : -pad_size -y2, 
                        pad_size + x1 : -pad_size -x2]
            im2 = chan2[pad_size + y1 +dy :-pad_size -y2 + dy,
                        pad_size + x1 +dx :-pad_size -x2 + dx]
            #im2tm = shift_img(chan2, dx, dy)
            #im2 = im2tm[30:-30,30:-30]
            this_score = score_function(im1,im2)
            if this_score < best_score:
                best_score, best_offset = this_score, (dy,dx)
                #print (">>> best_score %f (dx=%d,dy=%d)"%(best_score,dx,dy))
                 
    # ======================
    return best_offset

def shift_img(chan2, dx, dy):
    #newImg = chan2[30+dy:-30+dy,30+dx:-30+dx]
    #return chan2[dy:dy,dx:dx]
    newImg = chan2[dy:,dx:]
    return newImg
    
rg_dy, rg_dx  = align_channels(images_pad[0], images_pad[1])
rb_dy, rb_dx  = align_channels(images_pad[0], images_pad[2])
print ("Act>>","rg_dy = ", rg_dy, "rg_dx = ",rg_dx )
print ("Act>>","rb_dy = ", rb_dy, "rb_dx = ",rb_dx )
print ("Tgt>>","rg_dy = 11","rg_dx = 5 ")
print ("Tgt>>","rb_dy = 20","rb_dx = 8 ")

# Use the best alignments to now combine the three images. You should use any of the variables
# above to return a tensor that is (Height)x(Width)x3, which is a color image that you can visualize.
#QES: from tensor do you mean a 3 dim array with rgb channels ?

def combine_images():
    # ======================
    # You code here
    im1=images_pad[0][pad_size:-pad_size, 
                  pad_size:-pad_size] # R Channel : No Shift
    im2=images_pad[1][pad_size + rg_dy:-pad_size + rg_dy, 
                  pad_size + rg_dx:-pad_size + rg_dx] # G Channel
    im3=images_pad[2][pad_size + rb_dy:-pad_size + rb_dy, 
                  pad_size + rb_dx:-pad_size + rb_dx] # B Channel
    # ======================
    images = [im1, im2, im3] # 3 dimensional array for 3 channels
    return np.stack(images, axis=-1)

final_image = combine_images()
if final_image is not None:
    show_image(final_image)
    plt.savefig("outputFig.png")
    
