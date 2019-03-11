# Numpy-1.12.1
import numpy as np

# Opencv-3.2.0 compiled for Python3 with ffmpeg library support
import cv2

# Scipy-0.19.0
from scipy import ndimage as ndi

def freespace_detection(colorframe,scaleDownFactor=4, roadROIStartBelowGivenRowIndex=242):
   
    frame_gray = cv2.cvtColor(colorframe, cv2.COLOR_BGR2GRAY)
    # gray frame is required for marker generation
    frame_small_gray = cv2.resize(frame_gray, (0,0), fx=1/float(scaleDownFactor), fy=1/float(scaleDownFactor))
    # color frame for openCV's watershed algorithm
    frame_small_color = cv2.resize(colorframe, (colorframe.shape[1]/scaleDownFactor, colorframe.shape[0]/scaleDownFactor))

    # Assumption: area "just in front of car" will always be road
    roi_for_road_gray = frame_small_gray[(roadROIStartBelowGivenRowIndex/scaleDownFactor):,:]
    roi_for_road_color = frame_small_color[(roadROIStartBelowGivenRowIndex/scaleDownFactor):,:,:]

    # Median Filter
    denoised = cv2.medianBlur(roi_for_road_gray, 3)
   
    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    #disk(5) is used here to get a more smooth image
    # Valuse hardcoded (tuned to test video)
    markers = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT,np.ones((5, 5))) < 27
    #markers = rank.gradient(denoised, disk(5)) < 27
    markers = ndi.label(markers)[0]
    
    # local gradient (disk(2) is used to keep edges thin)
    #gradient = rank.gradient(denoised, disk(2))
    gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT,np.ones((3, 3))) 
    
    # process the watershed
    labels = cv2.watershed(roi_for_road_color, markers)
    
    # Assumption 2: Car will be always on road, so region just in front of car
    # will  be a road segment.
    # Values are hardcoded
    rowStart = (420/scaleDownFactor) - (roadROIStartBelowGivenRowIndex/scaleDownFactor)
    rowEnd = 500/scaleDownFactor
    colStart = 200/scaleDownFactor
    colEnd   = 500/scaleDownFactor
                        
    #road_segment = frame_small[rowStart:rowEnd, colStart:colEnd]
    road_segment_labels = set(np.unique(labels[rowStart:rowEnd, colStart:colEnd]))
                                
    # Road Non-road mask
    road_mask = np.zeros((frame_small_gray.shape[0], frame_small_gray.shape[1],3) , dtype=np.uint8)
    #road_mask = np.zeros(frame_small_color.shape , dtype=np.uint8)
    # Todo: Find pythonic way to do below computaion efficiently
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j] in road_segment_labels:
                road_mask[(roadROIStartBelowGivenRowIndex/scaleDownFactor)+i,j, 1] = 255

    road_mask = cv2.resize(road_mask, (0,0), fx=scaleDownFactor, fy=scaleDownFactor)
    detected_road = cv2.addWeighted(colorframe,0.7, road_mask ,0.3,0)
    return (detected_road, road_mask[:,:,1])                   
