from random import shuffle
import glob
shuffle_data = True  # shuffle the addresses before saving
hdf5_path = '../data/dataset.hdf5'  # address to where you want to save the hdf5 file
cat_dog_train_path = '../data/original/*.jpg'
# read addresses and labels from the 'resized' folder
addrs = glob.glob(cat_dog_train_path)
labels = [1 if 'plastic' in addr else 0 for addr in addrs]  # 1 = plastic, 0 = else
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 80% train, 20% test
train_addrs = addrs[0:int(0.7*len(addrs))]
train_labels = labels[0:int(0.7*len(labels))]
test_addrs = addrs[int(0.7*len(addrs)):]
test_labels = labels[int(0.7*len(labels)):]

import numpy as np
import h5py
data_order = 'th'  # 'th' for Theano, 'tf' for Tensorflow
# check the order of data and chose proper data shape to save images
if data_order == 'th':
    train_shape = (len(train_addrs), 3, 56, 56)
    test_shape = (len(test_addrs), 3, 56, 56)
elif data_order == 'tf':
    train_shape = (len(train_addrs), 56, 56, 3)
    test_shape = (len(test_addrs), 56, 56, 3)
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.int8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
hdf5_file["test_labels"][...] = test_labels


import cv2
# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 100 images
    if i % 100 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(train_addrs))
    # read an image and resize to (256, 256)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (56, 56), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))
for i in range(len(test_addrs)):
    # print how many images are saved every 100 images
    if i % 100 == 0 and i > 1:
        print 'Test data: {}/{}'.format(i, len(test_addrs))
    # read an image and resize to (256, 256)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (56, 56), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["test_img"][i, ...] = img[None]
# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()


import h5py
import numpy as np
hdf5_path = '../data/dataset.hdf5'
subtract_mean = False
# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")
# subtract the training mean
if subtract_mean:
    mm = hdf5_file["train_mean"][0, ...]
    mm = mm[np.newaxis, ...]
# Total number of samples
data_num = hdf5_file["train_img"].shape[0]


import h5py
from sklearn.model_selection import train_test_split
f = h5py.File("../data/dataset.hdf5")
x_train = f['train_img'].value
x_test = f['test_img'].value
y_train = f['train_labels'].value
temp = y_train.tolist()
y_train_new = []
for thing in temp:
    if thing == 0:
        y_train_new.append([0,1])
    else:
        y_train_new.append([1,0])
y_train = np.array(y_train_new)
y_test = f['test_labels'].value
temp = y_test.tolist()
y_test_new = []
for thing in temp:
    if thing == 0:
        y_test_new.append([0,1])
    else:
        y_test_new.append([1,0])
y_test = np.array(y_test_new)


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as K
from keras.models import model_from_json
K.set_image_dim_ordering('th')

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test /= 255


y_train = y_train[:,0]
y_test = y_test[:,0]

print("================================")
print(y_train.shape)
unique, counts = np.unique(y_train, return_counts=True)
print(unique)
print(counts)
print(y_test.shape)
unique, counts = np.unique(y_test, return_counts=True)
print(unique)
print(counts)


model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(3, 56, 56)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

check = ModelCheckpoint("logs/weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
#model.fit(x_train, y_train, batch_size=32, nb_epoch=1000,callbacks=[check],validation_data=(x_test,y_test))

'''
model.load_weights("logs/weights.813-0.82213.hdf5")

#out = model.predict_proba(x_test)
#out = np.array(out)

loss, accuracy = model.evaluate(x_train, y_train)
print(accuracy)

predicted =  model.predict_classes(x_train)
np.savetxt("expected.txt", y_train, fmt="%0d")
np.savetxt("predicted.txt", predicted, fmt="%0d")


threshold = np.arange(0.1,0.9,0.1)

acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])
y_prob = np.array(out)
for j in threshold:
    y_pred = [1 if prob>=j else 0 for prob in y_prob]
    acc.append( matthews_corrcoef(y_test,y_pred))
acc   = np.array(acc)
index = np.where(acc==acc.max()) 
accuracies.append(acc.max()) 
best_threshold[0] = threshold[index[0][0]]
acc = []
print "best thresholds", best_threshold


y_test = y_test.reshape(len(y_test), 1)

y_pred = np.array([[0 if out[i,j]>=best_threshold[j] else 1 for j in range(y_test.shape[1])] for i in range(len(y_test))])

print("-"*40)
print("Matthews Correlation Coefficient")
print("Class wise accuracies")
print(accuracies)


import os
i = 0
for img_file in os.listdir('../data/original/'):
    if i<1000:
            test_img_addr = os.path.join('../data/original/', img_file)
    # read an image and resize to (256, 256)
    # cv2 load images as BGR, convert it to RGB
    #addr = test_addrs[test_img_addr]
            img = cv2.imread(test_img_addr)
            img = cv2.resize(img, (56, 56), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
            if data_order == 'th':
                                   img = np.rollaxis(img, 2)
            img /= 255
            im_test_shape = (1, 3, 56, 56)
            img1 = np.zeros(im_test_shape)
            img1[0, ...] = img
            print("--------------------------------------------------")
            print(img_file, model.predict_classes(img1))
            i = i+1


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

'''
