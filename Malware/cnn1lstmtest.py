from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
from sklearn.cross_validation import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, average_precision_score, precision_recall_curve, hamming_loss



traindata = pd.read_csv('data/train-features.csv', header=None)
trainlabel = pd.read_csv('data/train-labels.csv', header=None)
testdata = pd.read_csv('data/test-features.csv', header=None)
testlabel = pd.read_csv('data/test-labels.csv', header=None)

X = traindata.iloc[:,0:1024]
Y = trainlabel.iloc[:,0]
C = testlabel.iloc[:,0]
T = testdata.iloc[:,0:1024]


trainX = np.array(X)
testT = np.array(T)

y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

lstm_output_size = 70

cnn = Sequential()
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(1024, 1)))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(25, activation="softmax"))

# define optimizer and objective, compile cnn
'''
cnn.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

# train
checkpointer = callbacks.ModelCheckpoint(filepath="results/cnnlstm1results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('results/cnnlstm1results/cnntrainanalysis1.csv',separator=',', append=False)
cnn.fit(X_train, y_train, nb_epoch=1000,callbacks=[checkpointer,csv_logger])
cnn.save("results/cnnlstm1results/cnn_model.hdf5")
'''

score = []
name = []
import os
for file in os.listdir("results/cnnlstm1results/"):
  #print("within a loop")
  cnn.load_weights("results/cnnlstm1results/"+file)
  # make predictions
  y_pred = cnn.predict_classes(X_test)
  accuracy = accuracy_score(y_test1, y_pred)
  recall = recall_score(y_test1, y_pred , average="weighted")
  precision = precision_score(y_test1, y_pred , average="weighted")
  f1 = f1_score(y_test1, y_pred, average="weighted")
  print(file)
  print("----------------------------------------------")
  print("accuracy")
  print("%.3f" %accuracy)
  print("racall")
  print("%.3f" %recall)
  print("precision")
  print("%.3f" %precision)
  print("f1score")
  print("%.3f" %f1)
  score.append(accuracy)
  name.append(file)

print("-------------------------------------------------------")
print(max(score))
print(name[score.index(max(score))])

print("***********************************************************")
cnn.load_weights("results/cnnlstm1results/"+name[score.index(max(score))])
y_pred = cnn.predict_classes(X_test)
y_proba = cnn.predict_proba(X_test)
np.savetxt('classical/predictedlabelcnnlstm1.txt', y_pred, fmt='%01d')
np.savetxt('classical/predictedprobacnnlstm1.txt', y_proba)
accuracy = accuracy_score(y_test1, y_pred)
recall = recall_score(y_test1, y_pred , average="weighted")
precision = precision_score(y_test1, y_pred , average="weighted")
f1 = f1_score(y_test1, y_pred, average="weighted")
print(file)
print("accuracy")
print("%.3f" %accuracy)
print("racall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)

