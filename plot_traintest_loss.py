import numpy as np
import matplotlib.pyplot as plt

losses_1 = np.load('results/traintest_lossexp17.npz')
losses_2 = np.load('results/traintest_lossexp41.npz')
losses_3 = np.load('results/traintest_lossexp42.npz')
losses_4 = np.load('results/traintest_lossexp39.npz')
losses_5 = np.load('results/traintest_lossexp40.npz')





test_loss_1 = losses_1['test_loss']
train_loss_1 = losses_1['train_loss']
test_loss_2 = losses_2['test_loss']
train_loss_2 = losses_2['train_loss']
test_loss_3 = losses_3['test_loss']
train_loss_3 = losses_3['train_loss']
test_loss_4 = losses_4['test_loss']
train_loss_4 = losses_4['train_loss']
test_loss_5 = losses_5['test_loss']
train_loss_5 = losses_5['train_loss']



def plot_spectra():
	fig = plt.figure()
	plt.rcParams.update({'font.size':13})
	plt.subplot(511)
	plt.plot(test_loss_1, label='Test loss', linewidth = 3.0)
	plt.plot(train_loss_1, label='Train loss', linewidth = 3.0)
	plt.ylabel('Loss')
	plt.title('RMSE')
	plt.legend()
	plt.subplot(512)
	plt.plot(test_loss_2, label='Test loss', linewidth = 3.0)
	plt.plot(train_loss_2, label='Train loss', linewidth = 3.0)
	plt.ylabel('Loss')
	plt.title('Smooth L1')
	plt.legend()
	plt.subplot(513)
	plt.plot(test_loss_3, label='Test loss', linewidth = 3.0)
	plt.plot(train_loss_3, label='Train loss', linewidth = 3.0)
	plt.legend()
	plt.ylabel('Loss')
	plt.title('Logcosh')
	plt.subplot(514)
	plt.plot(test_loss_4, label='Test loss', linewidth = 3.0)
	plt.plot(train_loss_4, label='Train loss', linewidth = 3.0)
	plt.legend()
	plt.ylabel('Loss')
	plt.title('1-Cos')
	plt.subplot(515)
	plt.plot(test_loss_5, label='Test loss', linewidth = 3.0)
	plt.plot(train_loss_5, label='Train loss', linewidth = 3.0)
	plt.legend()
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.title('Pearson')

	fig
	plt.show()



#plot_spectra05_and_energies()
plot_spectra()
#print(np.min(test_loss_05))
print(losses_2['test_loss'].shape)
print(test_loss_3)