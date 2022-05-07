import cv2
import numpy as np           
import argparse, sys, os
from startProject import *
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


des_dir = "images/" + self.comboBox.currentText() + "/"
health_dir = "images/healthy/"

print(des_dir)

def Analyze_img(df, imgpath, label, counter):
    #Reading the image by parsing the argument 
    org_img = cv2.imread(imgpath) 
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(img ,(450,650))
    
    imgcopy = image_resized.copy()
    img_colored = org_img.copy()
    img_colored = cv2.resize(img_colored ,(450,650))
    
    '''
    cv2.imshow('original',imgcopy)
    cv2.waitKey(0)
    '''
    #Guassian blur
    blured = cv2.GaussianBlur(imgcopy,(1,1),1)
    '''
    cv2.imshow('blurred',blured)
    cv2.waitKey(0)
    '''
    
    #Thresholding on hue image
    ret, thresh = cv2.threshold(blured,150,255,cv2.THRESH_BINARY)
    '''
    cv2.imshow('thresh', thresh)
    cv2.waitKey()
    '''
    
    #Finding contours for all infected regions
    contours,heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    Tarea = img.shape[0]*img.shape[1]
    
    perimeter = 0
    Infarea = 0
    for x in range(len(contours)):
    	cv2.drawContours(img_colored,contours[x],-1,(0,0,255), 2)
    	#Calculating area of infected region
    	Infarea += cv2.contourArea(contours[x])
    	perimeter += cv2.arcLength(contours[x], True)
    row = [perimeter, Infarea, Tarea]
    df[counter] = row
    '''
    cv2.imshow('Contour masked',img_colored)
    cv2.waitKey(0)
    '''

c = 0
y_values = []
df = pd.DataFrame()
for imgfilename in os.listdir(des_dir):
    imgpath = des_dir + imgfilename
    y_values.append(1)
    Analyze_img(df, imgpath, imgfilename, c)
    c += 1

for imgfilename in os.listdir(health_dir):
    imgpath = health_dir + imgfilename
    y_values.append(0)
    Analyze_img(df, imgpath, imgfilename, c)
    c += 1
    
x_data = df.T
print(x_data)
print(y_values)

param1 = self.label_perimeter.text().split(":")
param2 = self.label_infarea.text().split(":")
param3 = self.label_tarea.text().split(":")
samplerow = [param1[1], param2[1], param3[1]]

X_ul = pd.DataFrame([samplerow])

Sum = 0
for n in range(4):
    #Split the dataset into training and test sets (20 % Test size)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_values, test_size=0.3, stratify=y_values, random_state=42)
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    # split test
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(x_train, y_train)  
    pred = svclassifier.predict(X_ul)
    Sum = Sum + pred

output2 = " Prediction: " + str(Sum) + "\n"
output2 += "Accuracy: {}%".format(svclassifier.score(x_test, y_test) * 100 )  + " \n"


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svclassifier, x_test, y_test)                               
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


self.svmResults.setText(str(output2))


if(Sum < 2):
    output2 += "The image is sufficiently healthy!"
    self.svmResults.setText(output2)
else:
    output2 += "The image is infected!"
    self.svmResults.setText(output2)

    