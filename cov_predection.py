import os
from statistics import mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys
import numpy as np
import getopt
import cv2
import platform

def prepare_img(image_path,IMG_SIZE=300):
    img_array=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return np.array( new_array).reshape(-1,300,300,1)
    
def err():
    system = platform.system()
    if system == "Windows":
        print('todo')
    else:
        print('todo')
    sys.exit()

def main(argv):
    ###---        main        ---###
    image_path=""
    i_ok=False
    directory_path=""
    d_ok=False
    report_path=""
    r_ok=False
    model_path=""
    m_ok=False
    try:
        opts ,_= getopt.getopt(argv, "hi:d:m:r:", ["image=","directory=", "model=","report="])
    except getopt.GetoptError:
        err()

    for opt, arg in opts:
        if opt=="h":
            #todo
            pass
        ####################################
        if opt in ['-i','--image']:
            image_path=arg
            i_ok=True
        ####################################
        if opt in ['-d','--directory']:
            directory_path=arg
            d_ok=True
        ####################################
        if opt in ['-m','--model']:
            model_path=arg
            m_ok=True
        ####################################
        if opt in ['-r','--report']:
            report_path=arg
            r_ok=True
        ####################################
    arr=[]
    if i_ok:
        arr.append(prepare_img(image_path))
    from tensorflow import keras
    model=keras.models.load_model(model_path)
    print(model.predict(arr))
    # print(model.n_similarity(s,m))
   
    ###---        main-end       ---###
main(sys.argv[1:])
    
