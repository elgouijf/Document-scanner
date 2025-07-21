# import opencv and numpy packages
import numpy as np
import cv2 as cv

## this version is vornuable to certain rotations or skews that might get the bottom left corner to the top 
""" def order_points(points):
    # let's start by initializing the list of our ordered coordinates
    # to insure a good performance we'll impose that numbers are coded into 32 bits 
    #instead of 64 bits (which is the default setting for numpy),
    # it's a good habit to have since you could incounter large images in cv.
    rectangle = np.zeros((4,2), dtype= "float32")

    points = np.array(points, dtype = "float32") #makes usre we're using arrays

    sorted_by_y = points[np.argsort(points[:,1])]

    top = sorted_by_y[:2]

    bottom = sorted_by_y[2:]

    if top[0][0] > top[1][0]: # the top left corresponds to the smallest x on top
        top[0], top[1] = top[1], top[0] 
        
    
    if bottom[0][0] < bottom[1][0]: # the bottom right corresponds to the largest x in the bottom
        bottom[0], bottom[1] = bottom[1], bottom[0] 

    rectangle[:2] = top
    rectangle[2:] = bottom

    return rectangle
 """

def order_points(points):
    # let's start by initializing the list of our ordered coordinates
    # to insure a good performance we'll impose that numbers are coded into 32 bits 
    #instead of 64 bits (which is the default setting for numpy),
    # it's a good habit to have since you could incounter large images in cv.
    rectangle = np.zeros((4,2), dtype= "float32")

    points = np.array(points, dtype = "float32") #makes usre we're using arrays

    sum = np.sum(points, axis = 1) # a sum over rows ( x + y )
    difference = np.diff(points, axis = 1) # a difference over rows ( x - y )

    tl = points[np.argmin(sum)]
    rectangle[0] = tl
    tr = points[np.argmin(difference)]
    rectangle[1] = tr
    br = points[np.argmax(sum)]
    rectangle[2] = br
    bl = points[np.argmax(difference)]
    rectangle[3] = bl

    return rectangle


def four_points_transform(image, points):

    rectangle = order_points(points)
    (tl, tr, br, bl) = rectangle

    #determinate the width of the image
    width1 = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
    width2 = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
    maxwidth = max(int(width1) , int(width2))

    #determinate the height of the image
    height1 = np.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)
    height2 = np.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
    maxheight = max(int(height1) , int(height2))

    #now we'll use the calculated dimensions to obtain a top-down view of the image

    top_down = np.array([(0,0), (maxwidth - 1, 0), (maxwidth - 1, maxheight -1), (0, maxheight - 1)]
                        , dtype = "float32")
    
    #get the projective matrix
    M = cv.getPerspectiveTransform(rectangle, top_down) # get virtual depth w'
    warped = cv.warpPerspective(image, M, (maxwidth,maxheight)) # loop into (x,y) points and devide them by w'

    return warped


