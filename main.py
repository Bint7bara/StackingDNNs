from sklearn import datasets
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from numpy import dstack
from sklearn.linear_model import LogisticRegression
import numpy as np


dataset = np.loadtxt("NewGencode4DLTraining.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:5]
y = dataset[:, 5]

#train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state = 101)

model1 = Sequential()
model1.add(Dense(50,activation = 'relu',input_dim = 5))
model1.add(Dense(25,activation = 'relu'))
model1.add(Dense(1,activation = 'sigmoid'))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def accuracy_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=[accuracy_m])
history = model1.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 100)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy_m'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy_m'])
plt.legend(['loss','accuracy_m',"val_loss",'val_accuracy_m'])
plt.show()

model1.save('model1.h5')

model2 = Sequential()
model2.add(Dense(25,activation = 'relu',input_dim = 5))
model2.add(Dense(25,activation = 'relu'))
model2.add(Dense(10,activation = 'relu'))
model2.add(Dense(1,activation = 'sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=[accuracy_m])
history1 = model2.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 100)

plt.plot(history1.history['loss'])
plt.plot(history1.history['accuracy_m'])
plt.plot(history1.history['val_loss'])
plt.plot(history1.history['val_accuracy_m'])
plt.legend(['loss','accuracy_m',"val_loss",'val_accuracy_m'])
plt.show()

model2.save('model2.h5')

model3 = Sequential()
model3.add(Dense(50,activation = 'relu',input_dim = 5))
model3.add(Dense(25,activation = 'relu'))
model3.add(Dense(25,activation = 'relu'))
model3.add(Dropout(0.1))
model3.add(Dense(10,activation = 'relu'))
model3.add(Dense(1,activation = 'sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=[accuracy_m])
history3 = model3.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 100)

plt.plot(history3.history['loss'])
plt.plot(history3.history['accuracy_m'])
plt.plot(history3.history['val_loss'])
plt.plot(history3.history['val_accuracy_m'])
plt.legend(['loss','accuracy_m',"val_loss",'val_accuracy_m'])
plt.show()

model3.save('model3.h5')

model4 = Sequential()
model4.add(Dense(50,activation = 'relu',input_dim = 5))
model4.add(Dense(25,activation = 'relu'))
model4.add(Dropout(0.1))
model4.add(Dense(10,activation = 'relu'))
model4.add(Dense(1,activation = 'sigmoid'))

model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=[accuracy_m])
history4 = model4.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 100)

plt.plot(history4.history['loss'])
plt.plot(history4.history['accuracy_m'])
plt.plot(history4.history['val_loss'])
plt.plot(history4.history['val_accuracy_m'])
plt.legend(['loss','accuracy_m',"val_loss",'val_accuracy_m'])
plt.show()

model4.save('model4.h5')

dependencies = {
    'accuracy_m': accuracy_m
}

# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'model' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename,custom_objects=dependencies)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

n_members = 4
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat #
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = LogisticRegression() #meta learner
	model.fit(stackedX, inputy)
	return model

model = fit_stacked_model(members, X_test,y_test)

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

# evaluate model on test set
yhat = stacked_prediction(members, model, X_test)
score = accuracy_m(y_test/1.0, yhat/1.0)
print('Stacked F Score:', score)

from sklearn.metrics import accuracy_score

i = 0
for model in members:
    i+=1
    pred = model.predict(X_test)
    score = accuracy_score(y_test,pred.round())
    print('Accuracy of model {} is '.format(i),score)









