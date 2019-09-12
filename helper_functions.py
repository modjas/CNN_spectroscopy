import torch
import torch.nn as nn
import numpy as np


def log_cosh_loss(pred, true):
	loss = torch.log(torch.cosh(pred - true))
	return torch.sum(loss)/pred.shape[0] #Keep error comparable over different batch sizes

def cosinemetric(pred, true):
	numerator = (pred*true).sum()
	denominator = torch.sqrt((true*true).sum())*torch.sqrt((pred*pred).sum())
	return 1-numerator/denominator

def cosinetest(pred, true):
	numerator = (pred*true).sum()
	denominator = torch.sqrt((true*true).sum())*torch.sqrt((pred*pred).sum())
	return 1+numerator/denominator

def pearsonmetric(pred, true):
	
	avg_pred = torch.mean(pred)
	avg_true = torch.mean(true)
	
	numerator = ((pred-avg_pred)*(true-avg_true)).sum()
	denominator = torch.sqrt((pred-avg_pred).pow(2).sum())*torch.sqrt((true-avg_true).pow(2).sum())
	
	return 1-numerator/denominator

def pearsontest(pred, true):
	avg_pred = torch.mean(pred)
	avg_true = torch.mean(true)
	numerator = ((pred-avg_pred)*(true-avg_true)).sum()
	denominator = torch.sqrt((pred-avg_pred).pow(2).sum())*torch.sqrt((true-avg_true).pow(2).sum())
	return 1+numerator/denominator


def relative_difference(prediction, label):
	dE = 30/prediction.size(-1) #how many eV's one dE is
	numerator = dE*(label-prediction).pow(2).sum()
	denominator = dE*label.sum()

	return 1-torch.sqrt(numerator)/denominator

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
	return out

def get_model_error(validation_loader, model, full_errors=False, normalized = False):
	val_error_accuracy = []
	val_error_mae = []
	val_error_mse = []
	val_error_r2 = []


	with torch.no_grad():
		for data, target in validation_loader:
			output = model(data)
			if normalized == True:
				norm = torch.norm(output).item()
				output = output/norm
			val_error_accuracy_item = relative_difference(output, target)
			val_error_mae_item = mae(output, target)
			val_error_mse_item = mse(output, target)
			val_error_r2_item = r_squared(output, target)
			
			val_error_accuracy.append(val_error_accuracy_item)
			val_error_mae.append(val_error_mae_item)
			val_error_mse.append(val_error_mse_item)
			val_error_r2.append(val_error_r2_item)
	#return np.average(val_error_accuracy), np.sqrt(np.average(val_error_mse)), np.average(val_error_mae)
	if full_errors == True:
		return [i.item() for i in val_error_accuracy]
	return np.average(val_error_accuracy), np.sqrt(np.average(val_error_mse)), np.average(val_error_mae), np.average(val_error_r2)


def set_criterion(crit):
	if crit == "rmse":
		return nn.MSELoss()
	if crit == "smooth":
		return nn.SmoothL1Loss()
	if crit == "logcosh":
		return log_cosh_loss
	if crit == "cosine":
		return cosinemetric
	if crit == "cosinetest":
		return cosinetest
	if crit == "pearson":
		return pearsonmetric
	if crit == "pearsontest":
		return pearsontest
	if crit == "jsd":
		return jsd
	if crit == "cosemb":
		return nn.CosineEmbeddingLoss()


def coulomb_shuffle(coulomb_matrices):
    batch_size = coulomb_matrices.size()[0]
    coulomb_matrices = coulomb_matrices.squeeze(1)
    row_norms = coulomb_matrices.norm(dim=1)
    row_norms = row_norms + torch.randn_like(row_norms)
    sort_idxs = torch.argsort(row_norms, dim=1, descending=True)
    for idx, sort_idx in enumerate(sort_idxs):
        coulomb_matrices[idx] = coulomb_matrices[idx][sort_idxs[idx]][sort_idxs[idx]]
    coulomb_matrices = coulomb_matrices.unsqueeze(1)
    return coulomb_matrices

def earlystop_criterion(test_losses):
	min_index = np.argmin(test_losses)
	epochs_since_improvement = len(test_losses)-min_index-1
	return epochs_since_improvement