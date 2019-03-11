#In "Python 2" result is zero when performing an integer divison (the denominator is greater than numerator)
#This will always perform a float division when using the / operator and use // for integer division
from __future__ import division  

import numpy as np
import cv2

#--------------------------------------------------Helper Functions------------------------------------------------#

#Function to check the segmented image and ground truth image dimension match
def Check_Dimensions(f_SegmentedImage, f_GroundTruthImage):
    SegmentedImage_height, SegmentedImage_width		= Image_Dimensions(f_SegmentedImage)
    GroundTruthImage_height, GroundTruthImage_width	= Image_Dimensions(f_GroundTruthImage)

    if (SegmentedImage_height != GroundTruthImage_height) or (SegmentedImage_width != GroundTruthImage_width):
		print"Ground Truth [h=%d,w=%d] and Segmented Image [h=%d,w=%d]----Dimension Mismatch!!"%(GroundTruthImage_height,GroundTruthImage_width,SegmentedImage_height,SegmentedImage_width)
		print("Image dimensions has to be the same ")
		print("Exiting the Application")
		exit(0)
	
#Function to extract the number of classes in the given image    
def Extract_Classes(f_GroundTruthImage):
    #The unique pixel intensity values in the ground truth are extracted and stored in a sorted order
    ClassPixelIntensityValues_array   = np.unique(f_GroundTruthImage)
    #The number of classes obtained by just taking the length of the above array
    NumberOfClasses                   = len(ClassPixelIntensityValues_array)
  
    return ClassPixelIntensityValues_array, NumberOfClasses

#Function to create the masks 
#by comparing the obtained classes (which is in the "ClassPixelIntensityValues_array")
#one by one with the 
#2d image Matrix (eg: The segmented image and the ground truth image)
def Create_Masks(Image, ClassPixelIntensityValues_array, NumberOfClasses):
    #Obtain the height and width of the Image
    height, width  = Image_Dimensions(Image)
    #mask 3D array is created
    masks = np.zeros((NumberOfClasses, height, width))
    #Loop through the array "ClassPixelIntensityValues_array" 
    for counter, PixelIntensityValue in enumerate(ClassPixelIntensityValues_array):
        #comapares each element of the Image(Pixels) with PixelIntensityValue
        #and the result of each Image compared is moved to the 
        #masks 3D array
        masks[counter, :, :] = Image == PixelIntensityValue
        
    return masks

#Function to return the Image height and width
def Image_Dimensions(f_Image):
	height = f_Image.shape[0]
	width  = f_Image.shape[1]
	return height, width

#Function to return the unique, sorted array of classes that are in either of the two input Images.
#and also the number unique classes
def Union_Classes(f_Class_SegmentedImage_array, f_Class_GroundTruthImage_array):
    
    Classes = np.union1d(f_Class_SegmentedImage_array, f_Class_GroundTruthImage_array)
    NumberOf_Classes = len(Classes)

    return Classes, NumberOf_Classes

#--------------------------------------------------Helper Functions------------------------------------------------#


#--------------------------------------------------Metrics Computation Functions-------------------------------------------#


#computes ratio between the amount of properly 
#classified pixels and the total number of them
def Pixel_Accuracy(f_SegmentedImage, f_GroundTruthImage):
    
    #Variables initialized 
    AccuratelyClassifiedPixels_Total  =   0
    TotalNoPixels   =   f_GroundTruthImage.shape[0]*f_GroundTruthImage.shape[1]
    
    #Check if both the images obtained are of same dimensions
    Check_Dimensions(f_SegmentedImage, f_GroundTruthImage)
    
    for row in range(len(f_GroundTruthImage)):
        for col in range(len(f_GroundTruthImage[row])):
                if(f_GroundTruthImage[row][col] == f_SegmentedImage[row][col]):
                    AccuratelyClassifiedPixels_Total +=1;
    
    if (TotalNoPixels == 0):
        PixelAccuracy = 0
    else:
        PixelAccuracy = AccuratelyClassifiedPixels_Total / TotalNoPixels

    return PixelAccuracy

#Computes a ratio between the intersection and the union of 2 sets
#In our case the 2 sets are the GroundTruth Image and the predicted Segmented Image
def Mean_IntersectionOverUnion(f_SegmentedImage, f_GroundTruthImage):
    
    #Check if both the images obtained are of same dimensions
    Check_Dimensions(f_SegmentedImage, f_GroundTruthImage)
    
    #Obtain the pixel classes in segmented and the groundtruth image
    ClassPixelIntensityValues_GroundTruthImage_array, NumberOfClasses_GroundTruthImage      = Extract_Classes(f_GroundTruthImage)
    ClassPixelIntensityValues_SegmentedImage_array                                          = Extract_Classes(f_SegmentedImage)[0]
    
    
    #Obtain the Union of classes and the number of classes
    Classes,NumberOf_Classes=Union_Classes(ClassPixelIntensityValues_SegmentedImage_array, ClassPixelIntensityValues_GroundTruthImage_array)

    #create masks for the segmented and the ground truth image for each class(classes obatined after taking the union)
    SegmentedImage_masks        =   Create_Masks(f_SegmentedImage,Classes, NumberOf_Classes)
    GroundTruthImage_masks      =   Create_Masks(f_GroundTruthImage,Classes, NumberOf_Classes)
    
    #Array to hold the calculated intersection over union for each class
    IntersectionOverUnion = list([0]) * NumberOf_Classes
    

    for i, c in enumerate(Classes):
        Current_SegmentedImage_mask     = SegmentedImage_masks[i, :, :]
        Current_GroundTruthImage_mask   = GroundTruthImage_masks[i, :, :]
        
        
        #The Mask would be 0s for either of the image if they do not have the class of pixel 
        #who's presence is being searched for while creating the mask. For more details 
        #look into the function "Create_Masks". We could save some time by not processing them further
        if (np.sum(Current_SegmentedImage_mask) == 0) or (np.sum(Current_GroundTruthImage_mask) == 0):
            continue
            
        #The accurately classified pixels corresponds to the intersection region of 2 sets   
        AccuratelyClassifiedPixels_perClass     = np.sum(np.logical_and(Current_SegmentedImage_mask, Current_GroundTruthImage_mask))
        #1st Set
        GroundTruthImagePixels_perClass         = np.sum(Current_GroundTruthImage_mask)
        #2nd Set
        SegmentedImagePixels_perClass           = np.sum(Current_SegmentedImage_mask)
        
        IntersectionOverUnion[i]   = AccuratelyClassifiedPixels_perClass/(GroundTruthImagePixels_perClass + SegmentedImagePixels_perClass - AccuratelyClassifiedPixels_perClass)

    Mean_IntersectionOverUnion  = np.sum(IntersectionOverUnion)/ NumberOfClasses_GroundTruthImage
    
    
    return Mean_IntersectionOverUnion
    
#--------------------------------------------------Metrics Computation Functions-------------------------------------------#




#Important Note:A base implementation may be modified later