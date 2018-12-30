from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import CSVLogger
import keras
import keras.preprocessing.text
import itertools
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import callbacks
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import Convolution1D, MaxPooling1D

trainlabels = pd.read_csv('data/trainlabel.csv', header=None)
trainlabel = trainlabels.iloc[:,0:1]
testlabels = pd.read_csv('data/trainlabel.csv', header=None)
testlabel = testlabels.iloc[:,0:1]

train = pd.read_csv('data/train.txt', header=None)
test = pd.read_csv('data/train.txt', header=None)


X = train.values.tolist()
X = list(itertools.chain(*X))


T = test.values.tolist()
T = list(itertools.chain(*T))


# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X+T)))}

max_features = len(valid_chars) + 1
maxlen = np.max([len(x) for x in X])

# Convert characters to int and pad
X = [[valid_chars[y] for y in x] for x in X]

T = [[valid_chars[y] for y in x] for x in T]


X_train = sequence.pad_sequences(X, maxlen=maxlen)

'''
# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(T)))}

max_features1 = len(valid_chars) + 1
maxlen1 = np.max([len(x) for x in T])

# Convert characters to int and pad
T = [[valid_chars[y] for y in x] for x in T]
'''

X_test = sequence.pad_sequences(T, maxlen=maxlen)


y_train = np.array(trainlabel)
y_test = np.array(testlabel)

print(X_train.shape)
print(y_train.shape)


hidden_dims = 128
nb_filter = 64
filter_length = 5 
embedding_vecor_length = 128
pool_length = 4
lstm_output_size = 70

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

'''
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnnlstm/checkpoint-{epoch:02d}.hdf5", save_best_only=True, monitor='loss')
csv_logger = CSVLogger('logs/cnnlstm/training_set_lstmanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=32, nb_epoch=1000,validation_split=0.33, shuffle=True,callbacks=[checkpointer,csv_logger])
model.save("logs/cnnlstm/coomplemodel.hdf5")
score, acc = model.evaluate(X_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
'''

model.load_weights("logs/cnnlstm/checkpoint-97.hdf5")

y_pred = model.predict_classes(X_test)
y_prob = model.predict_proba(X_test)
#np.savetxt("res/cnn-lstm.txt", y_prob)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="binary")
precision = precision_score(y_test, y_pred , average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("racall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)

