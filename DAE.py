import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

import numpy as np

from mnist import MNIST

import sys

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

class Mnist_Dataset(Dataset):

	def __init__(self, X_data, X_data_n):

		X_data = (X_data.astype(np.float32) - 127.5) / 127.5
		X_data = np.reshape(X_data, (X_data.shape[0], 1, 28, 28))

		X_data_n = (X_data_n.astype(np.float32) - 127.5) / 127.5
		X_data_n = np.reshape(X_data_n, (X_data_n.shape[0], 1, 28, 28))

		self.X_data = (torch.from_numpy(X_data))
		self.X_data_n = (torch.from_numpy(X_data_n))
		self.len = self.X_data.size(0)

	def __getitem__(self, index):
		return self.X_data[index], self.X_data_n[index]

	def __len__(self):
		return self.len


def get_my_training_set():
	mndata = MNIST()
	X_trn, Y_trn = mndata.load_training()
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

	data = np.reshape(imag, (n, -1))
	return data

learning_rate = 0.0002
epochs = 100
batch_size = 64

X_trn, Y_trn = get_my_training_set()
X_trn = np.array(X_trn)
X_trn_n = noisy_data_set(X_trn)

X_trn, Y_trn = get_my_training_set()
X_trn = np.array(X_trn)

trn_data = Mnist_Dataset(X_trn, X_trn_n)
train_loader = DataLoader(dataset = trn_data, batch_size = batch_size, shuffle = True)

encoder = Encoder()
decoder = Decoder()	

parameters = list(encoder.parameters())+ list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = Adam(parameters, lr = learning_rate)

"""
try:
	encoder, decoder = torch.load('./model/DAE.pkl')
	print ("Load encoder and decoder suffessfully")
except:
	print ("Encoder and decoder are not found")
"""


for epoch in range(epochs):
	for index, (x, x_n) in enumerate(train_loader):
		var_x = Variable(x)
		var_x_n = Variable(x_n)
		optimizer.zero_grad()
		output = encoder(var_x_n)
		output = decoder(output)
		loss = loss_func(output, var_x)
		loss.backward()
		optimizer.step()

		print ("epoch [%d/%d] batch [%d/%d], loss: %.8f" % (epoch, epochs, index, trn_data.len/batch_size, loss))
		if (index % 50) == 0:
			torch.save([encoder,decoder],'./model/DAE.pkl')
#		break
torch.save([encoder,decoder],'./model/DAE.pkl')

