import numpy as np
import matplotlib.pyplot as plt
import os

losses_1 = np.load(os.path.join(os.path.dirname(__file__),'../results/traintest_lossexp28.npz'))
losses_2 = np.load(os.path.join(os.path.dirname(__file__),'../results/traintest_lossexp29.npz'))
losses_3 = np.load(os.path.join(os.path.dirname(__file__),'../results/traintest_lossexp38.npz'))
losses_4 = np.load(os.path.join(os.path.dirname(__file__),'../results/traintest_lossexp31.npz'))



test_loss_1 = losses_1['test_loss']/100000
train_loss_1 = losses_1['train_loss']/100000
rmse_loss_1 = losses_1['rmseloss']/100000
test_loss_2 = losses_2['test_loss']/1000
train_loss_2 = losses_2['train_loss']/1000
rmse_loss_2 = losses_2['rmseloss']/1000
test_loss_3 = losses_3['test_loss']/1000
train_loss_3 = losses_3['train_loss']/1000
rmse_loss_3 = losses_3['rmseloss']/1000
test_loss_4 = losses_4['test_loss']/1000
train_loss_4 = losses_4['train_loss']/1000
rmse_loss_4 = losses_4['rmseloss']/1000


def plot_spectra():
	fig = plt.figure()
	plt.subplot(311)
	plt.plot(test_loss_01, label='Test loss')
	plt.plot(train_loss_01, label='Train loss')
	plt.ylabel('Mean squared error')
	plt.title('Test loss with broadening 0.1')
	plt.subplot(312)
	plt.plot(test_loss_03, label='Test loss')
	plt.plot(train_loss_03, label='Train loss')
	plt.ylabel('Mean squared error')
	plt.title('Test loss with broadening 0.3')
	plt.legend()
	plt.subplot(313)
	plt.plot(test_loss_05, label='Test loss')
	plt.plot(train_loss_05, label='Train loss')
	plt.legend()
	plt.ylabel('Mean squared error')
	plt.title('Test loss with broadening 0.5 ')

	fig
	plt.show()

def plot_cos_pearson():
	fig = plt.figure()

	plt.subplot(411)
	plt.plot(test_loss_1, label='Test loss')
	plt.plot(train_loss_1, label='Train loss')
	plt.plot(rmse_loss_1, label='RMSE')
	plt.ylabel('Error')
	plt.title('Loss for training with Pearson')
	plt.legend()
	plt.subplot(412)
	plt.plot(test_loss_2, label='Test loss')
	plt.plot(train_loss_2, label='Train loss')
	plt.plot(rmse_loss_2, label='RMSE')
	plt.ylabel('Error')
	plt.title('Loss for training with "wrong" Pearson')
	plt.legend()
	plt.subplot(413)
	plt.plot(test_loss_3, label='Test loss')
	plt.plot(train_loss_3, label='Train loss')
	plt.plot(rmse_loss_3, label='RMSE')
	plt.ylabel('Error')
	plt.title('Loss for training with Cosine')
	plt.legend()
	plt.subplot(414)
	plt.plot(test_loss_4, label='Test loss')
	plt.plot(train_loss_4, label='Train loss')
	plt.plot(rmse_loss_4, label='RMSE')
	plt.ylabel('Error')
	plt.title('Loss for training with "wrong" Cosine')
	plt.legend()

	fig
	plt.show()	

plot_cos_pearson()
print(np.argmin(rmse_loss_1))
print(np.argmin(rmse_loss_2))
print(np.argmin(rmse_loss_3))
print(np.argmin(rmse_loss_4))
