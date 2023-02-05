import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel

dir = 'path to your dataset which contains train and test set'
print(os.listdir(dir))

# resize images to
IMG_SIZE = 128

# capture images and labels to an array
# creating empty list
# train
train_images = []
train_labels = []
for directory_path in glob.glob(r"path to your dataset\\train\\*"):
    train_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) #Reading color images
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #Resize images
        train_images.append(img)
        train_labels.append(train_label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# test
test_images = []
test_labels = [] 
for directory_path in glob.glob(r"path to your dataset\\test\\*"):
    test_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_images.append(img)
        test_labels.append(test_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# one-hot encoding the label
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

# split data into test and train dataset
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# FEATURE EXTRACTOR function
def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):  
        #print(image)
        
        df = pd.DataFrame()
        
        input_img = x_train[image, :,:,:]
        img = input_img

        # FEATURE 1 - Pixel values
        #Add pixel values to the data frame
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values
        
        # FEATURE 2 - Bunch of Gabor filter responses
        #Generate Gabor features
        num = 1  
        kernels = []
        for theta in range(2):  
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  
                lamda = np.pi/4
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                
                #filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img 
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  
        # FEATURE 3 Sobel
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1
    
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        
    return image_dataset
  
image_features = feature_extractor(x_train)

# reshape to a vector for SVM training
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_SVM = np.reshape(image_features, (x_train.shape[0], -1))  # reshape to #images, features

from sklearn import svm
SVM_model = svm.SVC(C=1,kernel='rbf',gamma='auto')
SVM_model.fit(X_for_SVM, y_train)

# predict on Test data
# extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_SVM = np.reshape(test_features, (x_test.shape[0], -1))
test_prediction = SVM_model.predict(test_for_SVM)

# inverse le transform to get original label back. 
test_prediction = le.inverse_transform(test_prediction)

# print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))
