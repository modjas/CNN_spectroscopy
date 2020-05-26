#Imports
import pkg_resources
import torch
import torch.optim as optim
import sys
import timeit
import numpy as np

#own file imports
import helper_functions
import spectroscopy_cnn_model
import load_data


pkg_resources.require("torch==1.0.0")

#Run with
# python3 spectroscopy_run_experiments.py spectra.npz 10 exp00 rmse 300

#Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is {}".format(device))

#Handle file and output arguments
spectra_file = 'data/' + sys.argv[1]
runtime = int(sys.argv[2])
experiment_number = sys.argv[3]
criterion_string = sys.argv[4]
spectra_name = sys.argv[1][:-4]
output_spectra_length = int(sys.argv[5])
print("Running training on Spectra: {} for {} epochs. Experiment number {}".format(spectra_name, runtime, experiment_number))

#Load data
train_loader, test_loader, validation_loader = load_data.load_experiment_data(spectra_file, 'all')

#Create network
network = spectroscopy_cnn_model.create_network(output_spectra_length)
#model_parameters = filter(lambda p: p.requires_grad, network.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#print("PARAMS", params)
network.to(device) #CUDA or CPU

#Set CUDA/CPU, optimizer and loss function
criterion = helper_functions.set_criterion(criterion_string)
optimizer = optim.Adam(network.parameters(), lr=1e-4)

#Define train and test
train_loss = []
test_losses = []
rmse_test_loss = []

def train(epoch):
	network.train()
	batch_losses = []
	for batch_idx, (data, target) in enumerate(train_loader):
		data = helper_functions.coulomb_shuffle(data)
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = network(data)
		if criterion_string == "cosemb":
			loss = criterion(output, target, torch.ones(target.shape[0]).to(device))
		else:
			loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 100 == 0:
			print("Train Epoch: {} [{}/{}] ({:.0f} %)".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100*batch_idx/len(train_loader)))
		batch_losses.append(loss.item())
	if criterion_string == "rmse":
		train_loss.append(np.sqrt(np.mean(batch_losses)))
	else:
		train_loss.append(np.mean(batch_losses))

def test():
	print("testing")
	network.eval()
	batch_losses = []
	test_rmse = []
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = network(data)
			if criterion_string == "cosemb":
				loss = criterion(output, target, torch.ones(target.shape[0]).to(device))
			else:
				loss = criterion(output, target)
			batch_losses.append(loss.item())

			rmsecrit = helper_functions.set_criterion("rmse")
			rmseloss = rmsecrit(output, target) #Compare loss function to RMSE Loss
			test_rmse.append(rmseloss.item())

		if criterion_string == "rmse":
			test_losses.append(np.sqrt(np.mean(batch_losses)))
		else:
			test_losses.append(np.mean(batch_losses))
		if criterion_string == 'rmse':
			print("Test set: Avg. loss: {}".format(np.sqrt(np.mean(batch_losses))))
		else:
			print("Test set: Avg. loss: {}".format(np.mean(batch_losses)))
		rmse_test_loss.append(np.sqrt(np.mean(test_rmse)))
		print("Test loss", test_losses)
		print("Train loss", train_loss)
		print("RMSE", rmse_test_loss)


def run_train_test(epochs):

	for epoch in range(1, epochs+1):
		print("Epoch {} out of {}".format(epoch, epochs))
		train(epoch)
		test()

		#save model and optimizer and losses.
		if epoch % 10 == 0:
			output_traintest_name = 'results/traintest_loss' + experiment_number + '.npz'
			np.savez(output_traintest_name, train_loss=train_loss, test_loss=test_losses, rmseloss=rmse_test_loss)

		if helper_functions.earlystop_criterion(test_losses) == 0:
			output_model_name = 'results/model_' + experiment_number + '.pth'
			output_optimizer_name = 'results/optimizer_' + experiment_number + '.pth'
			torch.save(network.state_dict(), output_model_name)
			torch.save(network.state_dict(), output_optimizer_name)


		print("Epochs since improved testing error: {} \n".format(helper_functions.earlystop_criterion(test_losses)))
		if helper_functions.earlystop_criterion(test_losses) > 150:
			print("Training ended early because of slow improvement.")
			break

#Run train and test
starttime = timeit.default_timer()
run_train_test(runtime)
print("Training took {} ".format(timeit.default_timer()-starttime))
print("Minimum testing error was {} and occured at epoch {}".format(min(test_losses), np.argmin(test_losses)+1))
