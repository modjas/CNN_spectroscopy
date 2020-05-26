import numpy as np
import matplotlib.pyplot as plt
import os

losses_rmse = np.load(os.path.join(os.path.dirname(__file__),'../results/traintest_lossexp17.npz'))
losses_smooth = np.load(os.path.join(os.path.dirname(__file__),'../results/traintest_lossexp15.npz'))
losses_logcosh = np.load(os.path.join(os.path.dirname(__file__),'../results/traintest_lossexp16.npz'))




test_loss_rmse = losses_rmse['test_loss']
train_loss_rmse = losses_rmse['train_loss']
test_loss_smooth = losses_smooth['test_loss']
train_loss_smooth = losses_smooth['train_loss']
test_loss_logcosh = losses_logcosh['test_loss']
train_loss_logcosh = losses_logcosh['train_loss']




fig = plt.figure()
plt.subplot(311)	
plt.plot(test_loss_rmse, label='Test loss')
plt.plot(train_loss_rmse, label='Train loss')
plt.ylabel('Loss')
plt.title('RMSE loss')
plt.subplot(312)
plt.plot(test_loss_smooth, label='Test loss')
#plt.plot(train_loss_smooth, label='Train loss')
plt.ylabel('Loss')
plt.title('Smooth L1 Loss')
plt.legend()
plt.subplot(313)
plt.plot(test_loss_logcosh, label='Test loss')
#plt.plot(train_loss_logcosh, label='Train loss')
plt.legend()
plt.ylabel('Loss')
plt.title('Logcosh loss ')

fig
plt.show()

