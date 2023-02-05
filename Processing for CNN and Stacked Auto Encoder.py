import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#configure: set the mayplotlib to inline and displays graphs below 
%matplotlib inline

#selection model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras
#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

X=[]
Z=[]
IMG_SIZE=150
LUNG_NODULES_DIR='path to lung nodule images'
LUNG_NON_NODULES_DIR='path to lung non-nodule images'

def assign_label(img,lung_type):
    return lung_type
  
def make_train_data(lung_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img,lung_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150,150))
        
        
        X.append(np.array(img))
        Z.append(str(label))
        
make_train_data('Nodules',LUNG_NODULES_DIR)
make_train_data('non_nodules',LUNG_NON_NODULES_DIR)

# one-hot encoding
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255
