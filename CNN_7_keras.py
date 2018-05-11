import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
import csv
import sys
def build_model():
	model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', input_shape = (28,28,1)))
	model.add(LeakyReLU(0.2))
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same'))
	model.add(LeakyReLU(0.2))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
	model.add(LeakyReLU(0.2))
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
	model.add(LeakyReLU(0.2))
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation = "softmax"))

	return model


def noisy_data_set(data):
	n = data.shape[0]

	imag = np.reshape(data, (n, 28, 28)).astype(np.float32)

	for j in range(int(n*0.4)):	
		i = np.random.randint(4, 15)
		maxtrix = np.random.randint(2, size = (10, 10)) * 255
		imag[j, i:i+10, i:i+10] = maxtrix
	#	plt.imshow(X_tst[j], cmap=plt.cm.gray)

	for j in range(int(n*0.4), int(n*0.6)):
		matrix = np.int32((np.random.normal(0, 100, (20, 20))))
		imag[j, 4:24, 4:24] = np.clip((imag[j, 4:24, 4:24] + matrix), 0, 255)
	data_n = np.reshape(imag, (n, -1))
	return data_n



"""
#########
First_time = True
for i in range(12):
	X_temp = np.genfromtxt('X_trn_0_denoise_%d.csv' % i, delimiter = ',')
	if First_time:
		X_trn = X_temp
		First_time = False
	else:
		X_trn = np.concatenate((X_trn, X_temp), axis = 0)

Y_trn = np.genfromtxt('Y_trn_0.csv', delimiter = ',')

First_time = True
for i in range(2):
	X_temp = np.genfromtxt('X_tst_denoise_%d.csv' % i, delimiter = ',')
	if First_time:
		X_tst = X_temp
		First_time = False
	else:
		X_tst = np.concatenate((X_tst, X_temp), axis = 0)
Y_tst = np.genfromtxt('Y_tst.csv', delimiter = ',')
########
"""
########
data_path = "./data.mat"
data_raw = loadmat(data_path)
Y_trn = data_raw["train_lbl"]


First_time = True
for j in range(5):
	for i in range(10):
		X_temp = np.genfromtxt('./model/X_trn_%d_denoise_%d.csv' % (j, i), delimiter = ',')
		if First_time:
			X_trn = X_temp
			First_time = False
		else:
			X_trn = np.concatenate((X_trn, X_temp), axis = 0)
for j in range(4):
	Y_temp = data_raw["train_lbl"]
	Y_trn = np.concatenate((Y_trn, Y_temp))
print (X_trn.shape)
print (Y_trn.shape)

First_time = True
for i in range(4):
	X_temp = np.genfromtxt('./model/X_tst_denoise_%d.csv' % i, delimiter = ',')
	if First_time:
		X_tst = X_temp
		First_time = False
	else:
		X_tst = np.concatenate((X_tst, X_temp), axis = 0)

########


X_trn = (X_trn-127.5) / 127.5
X_tst = (X_tst-127.5) / 127.5

X_trn = np.reshape(X_trn, (X_trn.shape[0], 28, 28, 1))
X_tst = np.reshape(X_tst, (X_tst.shape[0], 28, 28, 1))

Y_trn = to_categorical(Y_trn, num_classes = 10)

np.random.seed(2)

random_seed = 2

X_trn, X_val, Y_trn, Y_val = train_test_split(X_trn, Y_trn, test_size = 0.1, random_state=random_seed)

#######################

model = build_model()

#optimizer = Adam(lr = 0.0002, epsilon = 1e-8)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

batch_size = 86
epochs = 30

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

datagen.fit(X_trn)

history = model.fit_generator(datagen.flow(X_trn,Y_trn, batch_size=batch_size), epochs = epochs, validation_data = (X_val,Y_val), verbose = 2, steps_per_epoch=X_trn.shape[0] // batch_size, callbacks=[learning_rate_reduction])


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


Y_pred = model.predict(X_tst)
Y_pred = np.argmax(Y_pred,axis = 1)[:,None]

index = np.arange(1, len(Y_pred)+1)[:, None]
out = np.concatenate((index, Y_pred), axis = 1)
temp = np.array([['ID', 'Prediction']])
out = np.concatenate((temp, out), axis = 0)

with open('./model/out.csv', 'w') as file:
	writer = csv.writer(file)
	writer.writerows(out)







