# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 08:16:50 2023

@author: Ayomide
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 00:43:26 2023

@author: Ayomide
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
keras.utils.set_random_seed(1)
#tf.config.experimental.enable_op_determinism()
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

print("Num GPUs Available: ", len(tf.config.list_physical_devices()))
# data preparation           
#num_classes = 2


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('MNIST X_train shape:', X_train.shape)
print('MNIST y_train shape:', y_train.shape)

#Create the training set for 4 and 9 digits
#Get indices of digits 4 and 9
digit_indices = [i for i in range(len(y_train)) if y_train[i] == 4 or y_train[i] == 9]

#Vector of 4 and 9 digits
y_train = np.array([y_train[i] for i in digit_indices])

#Label the digit 4 as 0 and the digit 9 as 1
for i in range(len(y_train)):
  if y_train[i] == 4:
    y_train[i] = 0
  elif y_train[i] == 9:
      y_train[i] =1

#Training input for 4 and 9 digits
X_train = np.array([X_train[i] for i in digit_indices])

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)

print('MNIST (0-1) X_train shape:', X_train.shape)
print('MNIST (0-1) y_train shape:', y_train.shape)

#Create the Testing Set for 4 and 9 digits
digit_indices = [i for i in range(len(y_test)) if y_test[i] == 4 or y_test[i] == 9]
print(len(digit_indices))

y_test = np.array([y_test[i] for i in digit_indices])

#Label the digit 4 as 0 and the digit 9 as 1
for i in range(len(y_test)):
  if y_test[i] == 4:
    y_test[i] = 0
  elif y_test[i] == 9:
      y_test[i] =1

X_test = np.array([X_test[i] for i in digit_indices])

#y_test = keras.utils.to_categorical(y_test, num_classes)

#X_testt = pd.DataFrame(X_test)
#X_trainn = pd.DataFrame(X_train)
#y_testt = pd.DataFrame(y_test)
#y_trainn =pd.DataFrame(y_train)

#
def train(X,y,epochs,lr_train):
    """
    Parameters
    ----------
    X : Independent variables for the train set
    y : Target variable for the train set
    epochs : Number of complete pass through the entire training dataset

    Returns
    -------
    model : The trained model.

    """
    # input image dimensions
    model = keras.models.Sequential([keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape= input_shape),
                                     keras.layers.Conv2D(64, kernel_size=(3, 3),activation='relu'),
                                     keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                     keras.layers.Dropout(0.25),
                                     keras.layers.Flatten(),
                                     keras.layers.Dense(128,activation="relu"),
                                     keras.layers.Dropout(0.5),
                                     keras.layers.Dense(2,activation="softmax")])
    optimizer = keras.optimizers.Adam(learning_rate=lr_train)
    model.compile(loss="sparse_categorical_crossentropy",optimizer = optimizer, metrics = ["accuracy"])
    model.fit(X,y,epochs =epochs,shuffle = False,batch_size = 128)
    return model

model = train(X_train,y_train,11,0.001)
#print("Network summary :")
#model.summary()

def predict(train_model,X):
    """
    Parameters
    ----------
    train_model : Already trained model
    X : independent variables for the predicted label

    Returns
    -------
    y_label : predicted label

    """
    predictions = train_model.predict(X)
    y_label = np.argmax(predictions, axis=-1)
    return y_label



def misclassified(y_train,X_train):
    """
    Parameters
    ----------
    y_train : ------
    X : Independent variables for the train set
    X_train : Target variable for the train set
    epochs : Number of complete pass through the entire training dataset
    Returns
    -------
    new_x_train : misclassified trainset independent variables
    new_y_train : misclassified trainset target variable
        DESCRIPTION.
    accuracy : accuracy.

    """
    y_trainn = y_train.tolist()
    y_train_predict = predict(model,X_train)
    y_train_predict = y_train_predict.tolist()
    correct_count = 0
    wrong_count = 0
    y_train_data_index = np.arange(len(y_train))
    wrong_index = []
    
    
    y_label_and_predictedlabel = list(zip(y_trainn,y_train_predict))
    for i in range(len(y_label_and_predictedlabel)):
       if ((y_label_and_predictedlabel[i][0]) == (y_label_and_predictedlabel[i][1])):
          correct_count += 1
       else:
          wrong_count += 1
          wrong_index.append(y_train_data_index[i])
    
    accuracy = (correct_count/(correct_count+wrong_count))*100
    misclassified_indices_to_select  = wrong_index
    
    
    new_x_train = X_train[misclassified_indices_to_select];  new_y_train  = y_train[misclassified_indices_to_select]
    
  
    return new_x_train , new_y_train, accuracy

new_x_train,new_y_train,accuracy = misclassified(y_train,X_train)
new_y_train[:] = 2


#adding the neuron and layer weights
def add_neuron_and_layer_wt(wt,initial_model):
    """
    Parameters
    ----------
    wt : New weight to added to the reject node
    initial_model : Already trained model
    Returns
    -------
    new_model : The updated model

    """
    new_output_layer = tf.keras.layers.Dense(3, activation='softmax')
    new_model = tf.keras.Sequential(initial_model.layers[:-1] + [new_output_layer])
   
    weights,biases = initial_model.layers[-1].get_weights()
    new_weights = np.hstack((weights, np.full((weights.shape[0], 1), wt)))
    new_biases = np.append(biases,wt)
    new_model.layers[-1].set_weights([new_weights,new_biases])
    return new_model



wt = 0.0
new_model =  add_neuron_and_layer_wt(wt,model)
new_model.summary()

#just to show the weights
output_layer = new_model.layers[-1]
weights, biases = output_layer.get_weights()
weights_for_class = weights[:, 2]
weights_for_class
bias_for_class = biases[2]
bias_for_class






def prediction_accuracy_retrain(new_model,X_train_miss,y_train_miss,X_test,y_test,pre_epochs,reject_label,lr):
    """
    

    Parameters
    ----------
    new_model : The updated model
    X_train_miss : Independent variables for the missclassified train set
    y_train_miss : Target variable for the missclassified train label
    X_test : New independent variable for evaluating the model
    y_test : New dependent variable for evaluating the model
    pre_epochs : for retraining
    reject_label : Value for reject label

    Returns
    -------
    None.

    """
   
    for j in range(pre_epochs):
        optimizer_retrain = keras.optimizers.Adam(learning_rate=lr)
        new_model.compile(loss="sparse_categorical_crossentropy",optimizer =  optimizer_retrain )
        new_model.fit(X_train_miss,y_train_miss,epochs = 1)
        predictions_retrain = new_model.predict(X_test)
        y_test_predict = np.argmax(predictions_retrain, axis=-1)
        
        correct_count = 0
        wrong_count = 0
        reject_count = 0
       
        
        testlabel_and_predictedlabel = list(zip(y_test.tolist(),y_test_predict.tolist()))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][1]) == reject_label:  
                reject_count += 1
            if ((testlabel_and_predictedlabel[i][1]) != reject_label ) and ((testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1])):
                correct_count += 1
            if ((testlabel_and_predictedlabel[i][1]) != reject_label ) and ((testlabel_and_predictedlabel[i][0]) != (testlabel_and_predictedlabel[i][1])):
                wrong_count += 1
        reject_rate = (reject_count/(correct_count+wrong_count+reject_count))*100
        if correct_count+wrong_count == 0:
            print("Iteration "+str(j+1)+" Reject rate : "+str(np.round(reject_rate,4)))
        else:
            accuracy = (correct_count/(correct_count+wrong_count))*100
           # reject_rate = (reject_count/(correct_count+wrong_count+reject_count))*100
            print("Iteration "+str(j+1)+": Test accuracy: "+str(np.round(accuracy,4))+" Reject rate : "+str(np.round(reject_rate,4)))
            print("The number of reject is :",reject_count)
             

prediction_accuracy_retrain(new_model,new_x_train,new_y_train,X_test,y_test,20,2,0.0045)            
            
            
            
            
            










