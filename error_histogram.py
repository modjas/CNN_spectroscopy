import torch
import pkg_resources
pkg_resources.require("torch==1.0.0")

import load_data
import helper_functions
import spectroscopy_cnn_model
import matplotlib.pyplot as plt
import numpy as np

validation_loader = load_data.load_experiment_data('spectra.npz','validation')
model = spectroscopy_cnn_model.create_network(300)



model.load_state_dict(torch.load('results/model_exp17.pth', map_location='cpu'))
model.eval()
rmse_accuracy = helper_functions.get_model_error(validation_loader, model, True)

best_prediction = np.argmin(rmse_accuracy)
worst_prediction = np.argmax(rmse_accuracy)
median_prediction = rmse_accuracy.index(np.percentile(rmse_accuracy, 50, interpolation='nearest'))

print("Best index = {}\nWorst index = {}\nMedian index = {}".format(best_prediction, worst_prediction, median_prediction))
print("Best accuracy=", max(rmse_accuracy))

bin_width = 0.001

plt.hist(rmse_accuracy, bins=np.arange(min(rmse_accuracy), max(rmse_accuracy) + bin_width, bin_width))
plt.title("Prediction accuracies")
plt.xlabel("Accuracy")
plt.ylabel("Occurences")
plt.show()