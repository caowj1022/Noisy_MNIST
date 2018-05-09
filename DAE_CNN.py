import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import csv
from mnist import MNIST
from scipy.signal import medfilt2d

import sys

class Mnist_Dataset(Dataset):

	def __init__(self, X_data, Y_data):

		X_data = (X_data.astype(np.float32) - 127.5) / 127.5
		X_data = np.reshape(X_data, (X_data.shape[0], 1, 28, 28))

		self.X_data = (torch.from_numpy(X_data))
		self.Y_data = torch.from_numpy(Y_data).long().view(-1)
		self.len = self.X_data.size(0)

	def __getitem__(self, index):
		return self.X_data[index], self.Y_data[index]

	def __len__(self):
		return self.len

class Encoder(nn.Module):
	def __init__(self, f_dims = 32):
		super(Encoder, self).__init__()

		self.f_dims = f_dims

		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = self.f_dims, kernel_size = 3, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims),
			nn.Conv2d(in_channels = self.f_dims, out_channels = self.f_dims, kernel_size = 3, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims),
			nn.Conv2d(in_channels = self.f_dims, out_channels = self.f_dims*2, kernel_size = 3, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims*2),
			nn.Conv2d(in_channels = self.f_dims*2, out_channels = self.f_dims*2, kernel_size = 3, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims*2),

			###########
			nn.MaxPool2d(kernel_size = 2, stride = 2)
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels = self.f_dims*2, out_channels = self.f_dims*4, kernel_size = 3, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims*4),
			nn.Conv2d(in_channels = self.f_dims*4, out_channels = self.f_dims*4, kernel_size = 3, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims*4),

			###########
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = self.f_dims*4, out_channels = self.f_dims*8, kernel_size = 3, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU()
		)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.view(x.size(0), -1)
		return x


class Decoder(nn.Module):
	def __init__(self, f_dims = 32):
		super(Decoder,self).__init__()

		self.f_dims = f_dims

		self.layer1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels = self.f_dims*8,out_channels = self.f_dims*4, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims*4),
			nn.ConvTranspose2d(in_channels = self.f_dims*4,out_channels = self.f_dims*4, kernel_size = 3, stride = 1, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims*4),
			nn.ConvTranspose2d(in_channels = self.f_dims*4,out_channels = self.f_dims*2, kernel_size = 3, stride = 1, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims*2),
			nn.ConvTranspose2d(in_channels = self.f_dims*2,out_channels = self.f_dims*2, kernel_size = 3, stride = 1, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims*2)
		)
		self.layer2 = nn.Sequential(
			nn.ConvTranspose2d(in_channels = self.f_dims*2,out_channels = self.f_dims, kernel_size = 3, stride = 1, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims),
			nn.ConvTranspose2d(in_channels = self.f_dims,out_channels = self.f_dims, kernel_size = 3, stride = 1, padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU(),
			nn.BatchNorm2d(self.f_dims),
			nn.ConvTranspose2d(in_channels = self.f_dims,out_channels = 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
			nn.LeakyReLU(0.2),
#			nn.ReLU()
		)
	def forward(self,x):
		x = x.view(x.size(0), self.f_dims*8, 7, 7)
		x = self.layer1(x)
		x = self.layer2(x)
		return x

class Neural_Network(nn.Module):

	def __init__(self, f_dim = 32):
		super(Neural_Network, self).__init__()
		self.f_dim = f_dim

		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = self.f_dim, kernel_size = (5, 5), padding = 2),
			nn.LeakyReLU(0.2),
			nn.Conv2d(in_channels = self.f_dim, out_channels = self.f_dim, kernel_size = (5, 5), padding = 2),
			nn.LeakyReLU(0.2),
			nn.MaxPool2d(kernel_size = (2, 2)),
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels = self.f_dim, out_channels = self.f_dim*2, kernel_size = (3, 3), padding = 1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(in_channels = self.f_dim*2, out_channels = self.f_dim*2, kernel_size = (3, 3), padding = 1),
			nn.LeakyReLU(0.2),
			nn.MaxPool2d(kernel_size = (2, 2)),
		)
		self.layer3 = nn.Linear(7*7*self.f_dim*2, 256)
		self.out_layer = nn.Linear(256, 10)


	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.view(x.size(0), -1)
		x = self.layer3(x)
		x = self.out_layer(x)
		return x

def get_my_noisy_training_set(path1, path2):
	X_trn = np.genfromtxt(path1, delimiter = ',')
	Y_trn = np.genfromtxt(path2, delimiter = ',')	
	return X_trn, Y_trn

def noisy_data_set(data):
	n = data.shape[0]

	imag = np.reshape(data, (n, 28, 28))

	for j in range(int(n*0.4)):	
		i = np.random.randint(4, 15)
		maxtrix = np.random.randint(2, size = (10, 10)) * 255
		imag[j, i:i+10, i:i+10] = maxtrix
	#	plt.imshow(X_tst[j], cmap=plt.cm.gray)

	for j in range(int(n*0.4), int(n*0.6)):
		matrix = np.int32((np.random.normal(0, 100, (20, 20))))
		imag[j, 4:24, 4:24] = np.clip(imag[j, 4:24, 4:24] + matrix, 0, 255)

	data_n = np.reshape(imag, (n, -1))
	return data_n

def denoise_data_set(noisy_data):
	try:
		encoder, decoder = torch.load('./model/DAE.pkl')
		print ("Load encoder and decoder suffessfully")
	except:
		print ("Encoder and decoder are not found")

	noisy_data = np.reshape(noisy_data, (noisy_data.shape[0], 1, 28, 28))
	noisy_data = torch.from_numpy(noisy_data)
	noisy_data = Variable(noisy_data)
	output = encoder(noisy_data)
	output = decoder(output)

	output = np.clip(output.data.numpy(), -1, 1)

	output = np.reshape(output, (output.shape[0], -1))
	return output

def train_CNN(X_trn, Y_trn, batch_size = 8, epochs = 2):
	trn_data = Mnist_Dataset(X_trn, Y_trn)
	train_loader = DataLoader(dataset = trn_data, batch_size = batch_size, shuffle = True)
	Net = Neural_Network()
	print (Net)
	params = Net.parameters()

	lr = 0.0002

	optimizer = Adam(params, lr = lr)
	loss_func = nn.CrossEntropyLoss()


	for epoch in range(epochs):
		for index, (x, y) in enumerate(train_loader):

			var_x = Variable(x)
			var_y = Variable(y)
			output = Net(var_x)
			loss = loss_func(output, var_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print ("epoch [%d/%d] batch [%d/%d], loss: %.8f" % (epoch, epochs, index, trn_data.len/batch_size, loss))
	torch.save(Net.state_dict(), "./model/weight.tar")

def load_CNN():
	Net = Neural_Network()
	Net.load_state_dict(torch.load("./model/weight.tar"))
	return Net

def predict(Net, X_tst):
	var_x = Variable(X_tst)
	output = Net(var_x)
	prob = F.softmax(output, dim = 1)
	return (prob.max(dim = 1)[1]).data.numpy()

def score(X_tst, Y_tst):

	X_tst = (X_tst.astype(np.float32) - 127.5) / 127.5
	X_tst = np.reshape(X_tst, (X_tst.shape[0], 1, 28, 28))
	X_tst = (torch.from_numpy(X_tst))

	Net = load()
	Y_pred = predict(Net, X_tst).data.numpy()[:,None]

def write_data(data, path):
	with open(path, 'w') as file:
		writer = csv.writer(file)
		writer.writerows(data)

X_trn, Y_trn = get_my_noisy_training_set('X_trn_0.csv', 'Y_trn_0.csv')
X_trn = (X_trn.astype(np.float32) - 127.5) / 127.5
X_trn_d = denoise_data_set(X_trn)
X_trn_d = X_trn_d * 127.5 + 127.5
write_data(X_trn_d, 'X_trn_0_denoise.csv')

sys.exit()

X_tst = np.genfromtxt('X_tst_1.csv', delimiter = ',')
Y_tst = np.genfromtxt('Y_tst_1.csv', delimiter = ',')
X_tst = (X_tst.astype(np.float32) - 127.5) / 127.5
X_tst_d = denoise_data_set(X_tst)
X_tst_d = X_tst_d * 127.5 + 127.5
write_data(X_tst_d, 'X_tst_denoise.csv')
sys.exit()

X_trn, Y_trn = get_my_training_set()
X_trn = np.array(X_trn)

X_trn_n = (X_trn_n1.astype(np.float32) - 127.5) / 127.5
X_trn_n = np.reshape(X_trn_n, (X_trn_n.shape[0], 1, 28, 28))[36000:36010, :, :, :]
X_trn_n = (torch.from_numpy(X_trn_n))
X_trn_n = Variable(X_trn_n)

output = encoder(X_trn_n)
output = decoder(output)
out = output.data.numpy()
out = out*127.5 + 127.5
out = np.int32(out)


for i in range(10):
	image = np.reshape(X_trn[36000+i], (28, 28))
	image_n = np.reshape(X_trn_n1[36000+i], (28, 28))
	image_out = np.reshape(out[i], (28, 28))

	plt.imshow(image, cmap=plt.cm.gray)
	plt.show()
	plt.imshow(image_n, cmap=plt.cm.gray)
	plt.show()
	plt.imshow(image_out, cmap=plt.cm.gray)
	plt.show()

