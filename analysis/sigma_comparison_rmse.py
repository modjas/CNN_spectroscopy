import pkg_resources
pkg_resources.require("torch==1.0.0")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys
import getopt
from sklearn.model_selection import train_test_split
import os
print("Pytorch version is {}".format(torch.__version__))



#Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is {}".format(device))

#Load data
coulomb = np.absolute(np.load(os.path.join(os.path.dirname(__file__),'../data/coulomb.npz'))['coulomb'])


energies_01 = np.loadtxt(os.path.join(os.path.dirname(__file__),'../data/spectra_sigma0.1.txt'))
energies_03 = np.loadtxt(os.path.join(os.path.dirname(__file__),'../data/spectra_sigma0.3.txt'))
energies_05 = np.load(os.path.join(os.path.dirname(__file__),'../data/spectra.npz'))['spectra']

# train, test, val split
train_split = 0.9
val_test_split = 0.5

coulomb_train_01, coulomb_test_01, energies_train_01, energies_test_01 = train_test_split(coulomb, energies_01, test_size = 1.0-train_split, random_state=0)
coulomb_train_03, coulomb_test_03, energies_train_03, energies_test_03 = train_test_split(coulomb, energies_03, test_size = 1.0-train_split, random_state=0)
coulomb_train_05, coulomb_test_05, energies_train_05, energies_test_05 = train_test_split(coulomb, energies_05, test_size = 1.0-train_split, random_state=0)

coulomb_test_01, coulomb_val_01, energies_test_01, energies_val_01 = train_test_split(coulomb_test_01, energies_test_01, test_size = val_test_split, random_state=0)
coulomb_test_03, coulomb_val_03, energies_test_03, energies_val_03 = train_test_split(coulomb_test_03, energies_test_03, test_size = val_test_split, random_state=0)
coulomb_test_05, coulomb_val_05, energies_test_05, energies_val_05 = train_test_split(coulomb_test_05, energies_test_05, test_size = val_test_split, random_state=0)


coulomb_val_01 = torch.from_numpy(coulomb_val_01).unsqueeze(1).float()
coulomb_val_03 = torch.from_numpy(coulomb_val_03).unsqueeze(1).float()
coulomb_val_05 = torch.from_numpy(coulomb_val_05).unsqueeze(1).float()

energies_val_01 = torch.from_numpy(energies_val_01).float()
energies_val_03 = torch.from_numpy(energies_val_03).float() 
energies_val_05 = torch.from_numpy(energies_val_05).float()  


validation_data_01 = torch.utils.data.TensorDataset(coulomb_val_01, energies_val_01)
validation_data_03 = torch.utils.data.TensorDataset(coulomb_val_03, energies_val_03)
validation_data_05 = torch.utils.data.TensorDataset(coulomb_val_05, energies_val_05)

validation_loader_01 = torch.utils.data.DataLoader(validation_data_01, batch_size = 1)
validation_loader_03 = torch.utils.data.DataLoader(validation_data_03, batch_size = 1)
validation_loader_05 = torch.utils.data.DataLoader(validation_data_05, batch_size = 1)



#Define network
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1_1 = nn.Conv2d(1, 22, 3, padding=1) 
		self.conv1_2 = nn.Conv2d(22, 22, 3, padding=1)
		self.conv1_3 = nn.Conv2d(22, 22, 3, padding=1)
		self.conv2_1 = nn.Conv2d(22, 47, 3, padding=1)
		self.conv2_2 = nn.Conv2d(47, 47, 3, padding=1)
		self.conv2_3 = nn.Conv2d(47, 47, 3, padding=1)
		self.conv3_1 = nn.Conv2d(47, 42, 3, padding=1)
		self.conv3_2 = nn.Conv2d(42, 42, 3, padding=1)
		self.conv3_3 = nn.Conv2d(42, 42, 3, padding=1)
		self.pool = nn.MaxPool2d(2,2)
		self.fc = nn.Linear(42*3*3,300)

	def forward(self, x):
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = self.pool(F.relu(self.conv1_3(x)))
		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = self.pool(F.relu(self.conv2_3(x)))
		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = self.pool(F.relu(self.conv3_3(x)))
		x = x.view(-1, 42*3*3)
		x = self.fc(x)
		return x

def relative_difference(prediction, label):
	dE = 30/prediction.size(-1) #how many eV's one dE is
	numerator = dE*(label-prediction).pow(2).sum()
	denominator = dE*label.sum()

	return 1-torch.sqrt(numerator)/denominator

def kullback_leibler(a, b):

	a = a.numpy()[0,:]
	b = b.numpy()[0,:]
	running_sum = 0
	for i in range(0, len(a)-1):
		if a[i] !=0 and b[i] != 0:
			running_sum += a[i]/b[i]
	return running_sum

def jsd(pred, target):
	M = 0.5*(pred+target)
	return 0.5*torch.nn.KLDivLoss(pred, M)+0.5*torch.nn.KLDivLoss(target, M)

def mse(pred, target):
	return np.average((pred-target).pow(2))

def mae(pred, target):
	return np.average(np.abs(pred-target))

def r_squared(pred, target):
	mean = np.average(pred)
	ssres = (pred-target).pow(2).sum()
	sstot = (pred-mean).pow(2).sum()
	return 1-ssres/sstot

def plot_spectra(output, target):
	fig = plt.figure()
	plt.plot(output.numpy()[0])
	plt.plot(target.numpy()[0])
	fig
	plt.show()

def plot_error(val_error):
	fig = plt.figure()
	plt.plot(val_error)
	fig
	plt.show()

def get_model_error(val_loader):
	val_error_accuracy = []
	val_error_mae = []
	val_error_mse = []
	val_error_r2 = []

	with torch.no_grad():
		for data, target in val_loader:
			output = model(data)
			val_error_accuracy_item = relative_difference(output, target)
			val_error_mae_item = mae(output, target)
			val_error_mse_item = mse(output, target)
			val_error_r2_item = r_squared(output, target)
			
			val_error_accuracy.append(val_error_accuracy_item)
			val_error_mae.append(val_error_mae_item)
			val_error_mse.append(val_error_mse_item)
			val_error_r2.append(val_error_r2_item)

	return np.average(val_error_accuracy), np.sqrt(np.average(val_error_mse)), np.average(val_error_mae), np.average(val_error_r2)


#Load pretrained models for testing
model = Net()

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../results/model_exp18.pth'), map_location='cpu'))
model.eval()
accuracy_01, rmse_01, mae_01, r2_01 = get_model_error(validation_loader_01)

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../results/model_exp19.pth'), map_location='cpu'))
model.eval()
accuracy_03, rmse_03, mae_03, r2_03 = get_model_error(validation_loader_03)

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../results/model_exp17.pth'), map_location='cpu'))
model.eval()
accuracy_05, rmse_05, mae_05, r2_05 = get_model_error(validation_loader_05)

print("Errors when trained with sigma=0.1: Accuracy = {}. RMSE = {}. MAE = {}. R2 = {}.".format(accuracy_01, rmse_01, mae_01, r2_01))
print("Errors when trained with sigma=0.3: Accuracy = {}. RMSE = {}. MAE = {}. R2 = {}.".format(accuracy_03, rmse_03, mae_03, r2_03))
print("Errors when trained with sigma=0.5: Accuracy = {}. RMSE = {}. MAE = {}. R2 = {}.".format(accuracy_05, rmse_05, mae_05, r2_05))


