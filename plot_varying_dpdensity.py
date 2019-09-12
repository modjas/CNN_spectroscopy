import numpy as np
import matplotlib.pyplot as plt

spectra_01 = np.loadtxt('spectra_sigma0.1.txt')
spectra_03 = np.loadtxt('spectra_500_sigma0.1.txt')
spectra_05 = np.loadtxt('spectra_700_sigma0.1.txt')

ticks1 = np.linspace(-30, 0, num=300)
ticks2 = np.linspace(-30, 0, num=500)
ticks3 = np.linspace(-30, 0, num=700)



fig = plt.figure()
plt.rcParams.update({'font.size':14})
plt.subplot(321)
plt.plot(ticks1, spectra_01[2], linewidth=3.0)
plt.ylabel('Intensity')
plt.title('Np=300')

plt.subplot(323)
plt.plot(ticks2, spectra_03[2], linewidth=3.0)
plt.ylabel('Intensity')
plt.title('Np=500')

plt.subplot(325)
plt.plot(ticks3, spectra_05[2], linewidth=3.0)
plt.ylabel('Intensity')
plt.xlabel("Energy (eV)")
plt.title('Np=700')

plt.subplot(322)
plt.plot(ticks1, spectra_01[2], linewidth=3.0)
plt.title('Np=300')

plt.subplot(324)
plt.plot(ticks2, spectra_03[2], linewidth=3.0)
plt.title('Np=500')

plt.subplot(326)
plt.plot(ticks3, spectra_05[2], linewidth=3.0)
plt.xlabel("Energy (eV)")
plt.title('Np=700')




fig
plt.show()
