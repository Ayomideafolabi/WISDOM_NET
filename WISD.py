# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 00:43:26 2023

@author: Ayomide
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
np.random.seed (1)


def normalize (X): 
    max_val = X.max(axis= 0);min_val = X.min(axis= 0) 
    range_ = max_val - min_val
    X_df = (X - min_val)/(range_)
    return X_df

def relabel(g):
    Z =[]
    for  i  in g:
        if i  == "SIRA":
            Z.append(0)
        else:
            Z.append(1)  
    return Z

# data preparation           
data = pd.read_csv("Dry_Bean_Dataset_5_6_original.csv")
X = data[data.columns[0:16]]
X = normalize(X)
#X = normalize(X)
y = list(data["Class"])        
k = relabel(y)    
data ["Class"] = k
y= data["Class"]
X_original = X[0:6182] ; y_original = y[0:6182]


#spliting the previous data
X_train,X_test,y_train,y_test = train_test_split(X_original,y_original ,test_size = 0.1,random_state = 1)
X_trainn = np.array(X_train)
y_trainn = np.array(y_train).reshape(-1,1)
#X_train = np.array(X_train).reshape(-1,1)




#
def train(X,y,epochs):
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
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=[16,1]),
                                     keras.layers.Dense(8,activation="sigmoid"),
                                     keras.layers.Dense(2,activation="softmax")])
    model.compile(loss="sparse_categorical_crossentropy",optimizer = "sgd", metrics = ["accuracy"])
    model.fit(X,y,epochs =epochs)
    return model

model = train(X_trainn,y_trainn,300)
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
    y_train_data_index = y_train.index
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
    
    
    retrain_mask = X_train.index.isin(misclassified_indices_to_select);  yretrain_mask = y_train.index.isin(misclassified_indices_to_select)
    
    new_x_train =  X_train[retrain_mask];  new_y_train  = y_train[yretrain_mask]
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
    new_model.layers[2].set_weights([new_weights,new_biases])
    return new_model



wt = 0.005
new_model =  add_neuron_and_layer_wt(wt,model)
new_model.summary()

#just to show the weights
output_layer = new_model.layers[-1]
weights, biases = output_layer.get_weights()
weights_for_class = weights[:, 2]
weights_for_class
bias_for_class = biases[2]
bias_for_class






def prediction_accuracy_retrain(new_model,X_train_miss,y_train_miss,X_test,y_test,pre_epochs,reject):
   
    for j in range(pre_epochs):
        new_model.compile(loss="sparse_categorical_crossentropy",optimizer = "sgd", )
        new_model.fit(X_train_miss,y_train_miss,epochs = 1)
        predictions_retrain = new_model.predict(X_test)
        y_test_predict = np.argmax(predictions_retrain, axis=-1)
        
        correct_count = 0
        wrong_count = 0
        reject_count = 0
       
        
        testlabel_and_predictedlabel = list(zip(y_test.tolist(),y_test_predict.tolist()))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][1]) == reject:  
                reject_count += 1
            if ((testlabel_and_predictedlabel[i][1]) != reject ) and ((testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1])):
                correct_count += 1
            if ((testlabel_and_predictedlabel[i][1]) != reject ) and ((testlabel_and_predictedlabel[i][0]) != (testlabel_and_predictedlabel[i][1])):
                wrong_count += 1
        accuracy = (correct_count/(correct_count+wrong_count))*100
        reject_rate = (reject_count/(correct_count+wrong_count+reject_count))*100
        print("Iteration "+str(j+1)+": Test accuracy: "+str(accuracy)+" Reject rate : "+str(reject_rate))
        
             

prediction_accuracy_retrain(new_model,new_x_train,new_y_train,X_test,y_test,10,2)            
            
            
            
            
            










