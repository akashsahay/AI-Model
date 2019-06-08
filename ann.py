#ANN


#Data Preprocessing 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 4:7].values
Y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#making ANN

import keras
from keras.models import Sequential  #to initialize Neural network
from keras.layers import Dense #for the layers



#Dropout regularization is the solution for overfitting i.e. when the model learns too much, when overfitting
#occurs there is much higher accuracy in training set than the test set
#high variance means overfitting
#solution for overfitting i.e. when we see high variance


      
#initializing ANN
classifier = Sequential()

#Adding input layer and first hidden layer with dropout
classifier.add(Dense(units = 9, activation = 'relu', input_dim = 3))
# =============================================================================
# #classifier.add(Dropout(p= 0.1))  #output dim or units is the number of nodes in the
# #hidden layer and input_dim is the number of nodes in input layer we need to specify the number of input nodes as
# #the hidden layer do not know how many inputs to expect relu is for rectifier activation function
# #output_dim is most of the times average of all the independent and dependnt variable.


#Adding the  layers
classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) 
classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) 
classifier.add(Dense(units = 9, init = 'uniform', activation = 'relu')) 
classifier.add(Dense(1)) 

#Compiling the ANN



classifier.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'] ) #optimizer takes in the algorithm that we want to use in our ANN
 
 
#fitting the ANN to the training set

classifier.fit(X_train, Y_train, batch_size = 1, epochs = 100)

classifier.fit(X_test, Y_test, batch_size = 1, epochs = 100)
# =============================================================================
# getting 80% accuracy for both training and test dataset
# =============================================================================

#making the prediction

# Predicting the Test set results

Y_pred = classifier.predict(X_test) 
#Y_pred = (Y_pred > 0.5) 

classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("classifier.h5")
