import numpy as np
import matplotlib.pyplot as plt
import os

spectra_01 = np.loadtxt(os.path.join(os.path.dirname(__file__),'../data/spectra_sigma0.1.txt'))
spectra_03 = np.loadtxt(os.path.join(os.path.dirname(__file__),'../data/spectra_500_sigma0.1.txt'))
spectra_05 = np.loadtxt(os.path.join(os.path.dirname(__file__),'../data/spectra_700_sigma0.1.txt'))





fig = plt.figure()
plt.subplot(311)
plt.plot(spectra_01[2])
plt.title('Spectra with n=300')

plt.subplot(312)
plt.plot(spectra_03[2])
plt.title('Spectra with n=500')

plt.subplot(313)
plt.plot(spectra_05[2])
plt.title('Spectra with n=700')

fig.suptitle('Spectra with varying datapoint density')
fig
plt.show()
