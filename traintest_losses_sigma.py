import numpy as np
import matplotlib.pyplot as plt

losses_300 = np.load('results/traintest_lossexp17.npz')
losses_500 = np.load('results/traintest_lossexp19.npz')
losses_700 = np.load('results/traintest_lossexp18.npz')




test_loss_300 = losses_300['test_loss']
train_loss_300 = losses_300['train_loss']
test_loss_500 = losses_500['test_loss']
train_loss_500 = losses_500['train_loss']
test_loss_700 = losses_700['test_loss']
train_loss_700 = losses_700['train_loss']



def plot_spectra():
	fig = plt.figure()
	plt.rcParams.update({'font.size':14})
	plt.subplot(311)
	plt.plot(test_loss_300, label='Test loss', linewidth = 3.0)
	plt.plot(train_loss_300, label='Train loss', linewidth = 3.0)
	plt.ylabel('Loss')
	plt.title('$\sigma=0.5$')
	plt.legend()
	plt.subplot(312)
	plt.plot(test_loss_500, label='Test loss', linewidth = 3.0)
	plt.plot(train_loss_500, label='Train loss', linewidth = 3.0)
	plt.ylabel('Loss')
	plt.title('$\sigma=0.3$')
	plt.legend()
	plt.subplot(313)
	plt.plot(test_loss_700, label='Test loss', linewidth = 3.0)
	plt.plot(train_loss_700, label='Train loss', linewidth = 3.0)
	plt.legend()
	plt.ylabel('Loss')
	plt.title('$\sigma=0.1$')
	plt.xlabel('Epoch')

	fig
	plt.show()



#plot_spectra05_and_energies()
plot_spectra()
#print(np.min(test_loss_05))
