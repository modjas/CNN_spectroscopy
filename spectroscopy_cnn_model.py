import torch
import torch.nn as nn
import torch.nn.functional as F

#Define network
class Net(nn.Module):
	def __init__(self, output_spectra_length):
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
		self.fc = nn.Linear(42*3*3, output_spectra_length)

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



def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_uniform_(m.weight.data, gain=1)
		nn.init.constant_(m.bias,0)
	if isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight.data, gain=1)
		nn.init.constant_(m.bias,0)

def create_network(output_spectra_length):
	network = Net(output_spectra_length)
	network.apply(weights_init)
	return network
