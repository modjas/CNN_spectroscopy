#Imports
import pkg_resources
import torch
import torch.optim as optim
import sys
import timeit

#own file imports
import helper_functions
import spectroscopy_cnn_model
import load_data
import numpy as np

normalize = sys.argv[1]

output_spectra_length = 300
spectra_file = 'spectra.npz'
experiment_model = 'exp42'



train_loader, test_loader, validation_loader = load_data.load_experiment_data(spectra_file, 'all', 1, 1, 1)
model = spectroscopy_cnn_model.create_network(output_spectra_length)

model.load_state_dict(torch.load('results/model_'+experiment_model+'.pth', map_location='cpu'))
model.eval()

predicted_spectra_training = []
predicted_spectra_testing = []
predicted_spectra_validation = []

with torch.no_grad():
	print("Predictions training data")
	
	for data, target in train_loader:
		output = model(data)
		predicted_spectra_training.append(output)

	print("Predicting testing data")
	for data, target in test_loader:
		output = model(data)
		predicted_spectra_testing.append(output)
	
	print("predicting validation data")
	for data, target in validation_loader:
		output = model(data)
		predicted_spectra_validation.append(output)


predicted_spectra_validation = np.array([predicted_spectra_validation[i].squeeze(0).numpy() for i in range(0,len(predicted_spectra_validation)-1)])
predicted_spectra_training = np.array([predicted_spectra_training[i].squeeze(0).numpy() for i in range(0,len(predicted_spectra_training)-1)])
predicted_spectra_testing = np.array([predicted_spectra_testing[i].squeeze(0).numpy() for i in range(0,len(predicted_spectra_testing)-1)])

if normalize == 'normalize':
	pred = []
	train = []
	val = []
	for i in predicted_spectra_validation:
		norm = np.linalg.norm(i)
		i = i/norm
		val.append(i/norm)
	for i in predicted_spectra_training:
		norm = np.linalg.norm(i)
		i = i/norm
		train.append(i/norm)
	for i in predicted_spectra_testing:
		norm = np.linalg.norm(i)
		i = i/norm
		pred.append(i/norm)
	output_filename = 'predictions/'+experiment_model+'_normalized.npz'
	np.savez(output_filename, training_data=np.array(pred), testing_data=np.array(train), validation_data=np.array(val))
else:
	output_filename = 'predictions/'+experiment_model+'.npz'
	np.savez(output_filename, training_data=predicted_spectra_training, testing_data=predicted_spectra_testing, validation_data=predicted_spectra_validation)

