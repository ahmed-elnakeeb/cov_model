import cv2
import numpy as np           
import argparse, sys, os
from startProject import *
import pandas as pd
import sklearn
from startProject import *
from PyQt5.QtGui import QPixmap, QImage


def imgInfo(self, imgpath):
   #Reading the image by parsing the argument 
    org_img = cv2.imread(imgpath) 
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(img ,(450,650))
    
    imgcopy = image_resized.copy()
    img_colored = org_img.copy()
    img_colored = cv2.resize(img_colored ,(450,650))
    
    
    cv2.imshow('original',imgcopy)
    cv2.waitKey(0)

    #Guassian blur
    blured = cv2.GaussianBlur(imgcopy,(1,1),1)
    cv2.imshow('blurred',blured)
    cv2.waitKey(0)
    
    #Thresholding on hue image
    ret, thresh = cv2.threshold(blured,150,255,cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh)
    cv2.waitKey()
    
    #Finding contours for all infected regions
    contours,heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    Tarea = img.shape[0]*img.shape[1]
    
    counter = 0
    perimeter = 0
    Infarea = 0
    df = pd.DataFrame()
    for x in range(len(contours)):
    	cv2.drawContours(img_colored,contours[x],-1,(0,0,255), 2)
    	#Calculating area of infected region
    	Infarea += cv2.contourArea(contours[x])
    	perimeter += cv2.arcLength(contours[x], True)
    
    output = "Total Area: " + str(Tarea) + "\n"
    output += "Detected Area: " + str(Infarea) + "\n"
    output += "Perimeter: " + str(perimeter) + "\n"
    output += "_____________________________________"

    self.label_tarea.setText("Total Area:" + str(Tarea))
    self.label_infarea.setText("Detected Area:" + str(Infarea))
    self.label_perimeter.setText("Perimeter:" + str(perimeter))
    
    self.output_txt.setText(output)
    cv2.imshow('diagnos', img_colored)
    cv2.waitKey(0)
 

# imagePath = self.filename_text.text()
# imgInfo(self, imagePath)