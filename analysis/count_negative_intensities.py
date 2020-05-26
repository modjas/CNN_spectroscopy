import numpy as np
import os


predictions_rmse = np.load(os.path.join(os.path.dirname(__file__), '../predictions/exp17_normalized.npz'))['validation_data']
predictions_smooth = np.load(os.path.join(os.path.dirname(__file__), '../predictions/exp22_normalized.npz'))['validation_data']
predictions_logcosh = np.load(os.path.join(os.path.dirname(__file__), '../predictions/exp23_normalized.npz'))['validation_data']
predictions_cos = np.load(os.path.join(os.path.dirname(__file__), '../predictions/exp39_normalized.npz'))['validation_data']
predictions_pearson = np.load(os.path.join(os.path.dirname(__file__), '../predictions/exp40_normalized.npz'))['validation_data']

datapoints = len(predictions_rmse)*300

def count_negs(predictions):
	negative_values = 0
	positive_values = 0
	for i in predictions:
		for j in i:
			if j<0:
				negative_values += 1
			else:
				positive_values += 1
	return negative_values, positive_values

def neg_magnitude(predictions):
	neg_vector = 0
	for i in predictions:
		for j in i:
			if j < 0:
				neg_vector += j

	return neg_vector/len(predictions)

neg_rmse, pos_rmse = count_negs(predictions_rmse)
neg_smooth, pos_smooth = count_negs(predictions_smooth)
neg_logcosh, pos_logcosh = count_negs(predictions_logcosh)
neg_cos, pos_cos = count_negs(predictions_cos)
neg_pearson, pos_pearson = count_negs(predictions_pearson)

print(neg_rmse/datapoints, neg_smooth/datapoints, neg_logcosh/datapoints, neg_cos/datapoints, neg_pearson/datapoints)

print(neg_magnitude(predictions_rmse), neg_magnitude(predictions_smooth), neg_magnitude(predictions_logcosh), neg_magnitude(predictions_cos), neg_magnitude(predictions_pearson))

