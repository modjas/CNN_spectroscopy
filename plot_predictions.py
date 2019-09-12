import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import pkg_resources
pkg_resources.require("torch==1.0.0")
import load_data
import helper_functions
import spectroscopy_cnn_model


spectra_predicted = np.load('predictions/exp17.npz')['validation_data']

#Load data with same seed to be able to compare
coulomb = np.absolute(np.load('coulomb.npz')['coulomb'])
energies = np.load('spectra.npz')['spectra']
train_split = 0.9
val_test_split = 0.5
coulomb_train, coulomb_test, energies_train, energies_test = train_test_split(coulomb, energies, test_size = 1.0-train_split, random_state=0)
coulomb_test, coulomb_val, energies_test, energies_val = train_test_split(coulomb_test, energies_test, test_size = val_test_split, random_state=0)

def relative_difference(prediction, label):
	dE = 30/len(prediction) #how many eV's one dE is
	numerator = np.sum(np.power(dE*(label-prediction),2))
	denominator = np.sum(dE*label)

	return 1-np.sqrt(numerator)/denominator

validation_loader = load_data.load_experiment_data('spectra.npz','validation')
model = spectroscopy_cnn_model.create_network(300)


model.load_state_dict(torch.load('results/model_exp17.pth', map_location='cpu'))
model.eval()
rmse_accuracy = helper_functions.get_model_error(validation_loader, model, True)

best_prediction = np.argmin(rmse_accuracy)
worst_prediction = np.argmax(rmse_accuracy)
median_prediction = rmse_accuracy.index(np.percentile(rmse_accuracy, 50, interpolation='nearest'))

index = 1836
index2 = 5670

error = relative_difference(spectra_predicted[index], energies_val[index])
error2 = relative_difference(spectra_predicted[index2], energies_val[index2])
print("LOSS", error)
print("LOSS2", error2)

print("Best index = {}\nWorst index = {}\nMedian index = {}".format(best_prediction, worst_prediction, median_prediction))
print("Best accuracy=", max(rmse_accuracy), "molecule", len(energies_test)+len(energies_train)-1+index)
print("Worst accuracy=", min(rmse_accuracy), "molecule", len(energies_test)+len(energies_train)-1+index2)

ticks = np.linspace(-30, 0, num=300)
fig = plt.figure()
plt.rcParams.update({'font.size':14})

plt.subplot(131)
plt.rcParams.update({'font.size':14})
plt.plot(ticks, energies_val[index], linewidth=3.0, label='Target')
plt.plot(ticks, spectra_predicted[index], linewidth=3.0, label='Prediction')
plt.ylabel('Intensity')
plt.xlabel("E (eV)")
plt.legend()
plt.title('Best prediction, 98.3%')

plt.subplot(132)
plt.rcParams.update({'font.size':14})
plt.plot(ticks, energies_val[index2], linewidth=3.0, label='Target')
plt.plot(ticks, spectra_predicted[index2], linewidth=3.0, label='Prediction')
plt.ylabel('Intensity')
plt.xlabel("E (eV)")
plt.legend()
plt.title('Worst prediction, 82.2%')

bin_width = 0.0025
plt.subplot(133)
plt.rcParams.update({'font.size':14})
plt.hist(rmse_accuracy, bins=np.arange(min(rmse_accuracy), max(rmse_accuracy) + bin_width, bin_width))
plt.title("Prediction accuracies")
plt.xlabel("Accuracy")
plt.ylabel("Occurences")


fig
plt.show()



#Best accuracy= 0.9834389090538025