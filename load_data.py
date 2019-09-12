import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
import pkg_resources
pkg_resources.require("torch==1.0.0")

#Load data

def load_experiment_data(spectra_file, data_specified, batch_size_train = 90, batch_size_test = 1000, batch_size_validation=1):
	coulomb = np.absolute(np.load('coulomb.npz')['coulomb'])

	if spectra_file[-4:] == '.txt':
		energies = np.loadtxt(spectra_file)
	if spectra_file[-4:] == '.npz':
		energies = np.load(spectra_file)['spectra']

	# train, test, val split
	train_split = 0.9
	val_test_split = 0.5

	coulomb_train, coulomb_test, energies_train, energies_test = train_test_split(coulomb, energies, test_size = 1.0-train_split, random_state=0)

	coulomb_test, coulomb_val, energies_test, energies_val = train_test_split(coulomb_test, energies_test, test_size = val_test_split, random_state=0)


	coulomb_train = torch.from_numpy(coulomb_train).unsqueeze(1).float()
	energies_train = torch.from_numpy(energies_train).float() 
	coulomb_test = torch.from_numpy(coulomb_test).unsqueeze(1).float()
	energies_test = torch.from_numpy(energies_test).float() 
	coulomb_val = torch.from_numpy(coulomb_val).unsqueeze(1).float()
	energies_val = torch.from_numpy(energies_val).float() 




	train_data = torch.utils.data.TensorDataset(coulomb_train, energies_train)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size_train, shuffle=True)

	test_data = torch.utils.data.TensorDataset(coulomb_test, energies_test)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test)

	validation_data = torch.utils.data.TensorDataset(coulomb_val, energies_val)
	validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = batch_size_validation)

	if data_specified == 'train':
		return train_loader
	if data_specified == 'test':
		return test_loader
	if data_specified == 'validation':
		return validation_loader
	if data_specified == 'all':
		return train_loader, test_loader, validation_loader
	