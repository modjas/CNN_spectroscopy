import torch
import pkg_resources
pkg_resources.require("torch==1.0.0")
import sys
sys.path.append('..')
import load_data
import helper_functions
import spectroscopy_cnn_model
import os

validation_loader = load_data.load_experiment_data(os.path.join(os.path.dirname(__file__),'../data/spectra_normalized.npz'),'validation', relative_path=True)
model = spectroscopy_cnn_model.create_network(300)



model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../results/model_exp17.pth'), map_location='cpu'))
model.eval()
rmse_accuracy, rmse_rmse, rmse_mae, rmse_r2 = helper_functions.get_model_error(validation_loader, model, normalized=True)

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../results/model_exp41.pth'), map_location='cpu'))
model.eval()
smooth_accuracy, smooth_rmse, smooth_mae, smooth_r2 = helper_functions.get_model_error(validation_loader, model, normalized=True)

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../results/model_exp42.pth'), map_location='cpu'))
model.eval()
logcosh_accuracy, logcosh_rmse, logcosh_mae, logcosh_r2 = helper_functions.get_model_error(validation_loader, model, normalized=True)

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../results/model_exp39.pth'), map_location='cpu'))
model.eval()
cosine_accuracy, cosine_rmse, cosine_mae, cosine_r2 = helper_functions.get_model_error(validation_loader, model, normalized=True)

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'../results/model_exp40.pth'), map_location='cpu'))
model.eval()
pearson_accuracy, pearson_rmse, pearson_mae, pearson_r2 = helper_functions.get_model_error(validation_loader, model, normalized=True)


print("Errors when trained by RMSE: Accuracy = {:.4f}. RMSE = {:.4f}. MAE = {:.4f}. R2 = {:.4f}.".format(rmse_accuracy, rmse_rmse, rmse_mae, rmse_r2))
print("Errors when trained by Smooth L1: Accuracy = {:.4f}. RMSE = {:.4f}. MAE = {:.4f}. R2 = {:.4f}.".format(smooth_accuracy, smooth_rmse, smooth_mae, smooth_r2))
print("Errors when trained by LogCosh: Accuracy = {:.4f}. RMSE = {:.4f}. MAE = {:.4f}. R2 = {:.4f}.".format(logcosh_accuracy, logcosh_rmse, logcosh_mae, logcosh_r2))
print("Errors when trained by 1-cos: Accuracy = {:.4f}. RMSE = {:.4f}. MAE = {:.4f}. R2 = {:.4f}.".format(cosine_accuracy, cosine_rmse, cosine_mae, cosine_r2))
print("Errors when trained by Pearson: Accuracy = {:.4f}. RMSE = {:.4f}. MAE = {:.4f}. R2 = {:.4f}.".format(pearson_accuracy, pearson_rmse, pearson_mae, pearson_r2))