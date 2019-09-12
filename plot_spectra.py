import numpy as np
import matplotlib.pyplot as plt

spectra_01 = np.loadtxt('spectra_sigma0.1.txt')
spectra_03 = np.loadtxt('spectra_sigma0.3.txt')
spectra_05 = np.loadtxt('spectra_sigma0.5.txt')



ticks = np.linspace(-30, 0, num=300)



fig = plt.figure()
plt.rcParams.update({'font.size':14})

plt.subplot(331)
plt.plot(ticks, spectra_01[2], linewidth=3.0)
plt.ylabel('Intensity')
plt.title('$\sigma$=0.1eV')

plt.subplot(332)
plt.plot(ticks, spectra_01[16], linewidth=3.0)
plt.title('$\sigma$=0.1eV')

plt.subplot(333)
plt.plot(ticks, spectra_01[-1], linewidth=3.0)
plt.title('$\sigma$=0.1eV')

plt.subplot(334)
plt.plot(ticks, spectra_03[2], linewidth=3.0)
plt.ylabel('Intensity')
plt.title('$\sigma$=0.3eV')

plt.subplot(335)
plt.plot(ticks, spectra_03[16], linewidth=3.0)
plt.title('$\sigma$=0.3eV')

plt.subplot(336)
plt.plot(ticks, spectra_03[-1], linewidth=3.0)
plt.title('$\sigma$=0.3eV')

plt.subplot(337)
plt.plot(ticks, spectra_05[2], linewidth=3.0)
plt.ylabel('Intensity')
plt.xlabel("Energy (eV)")
plt.title('$\sigma$=0.5eV')

plt.subplot(338)
plt.plot(ticks, spectra_05[16], linewidth=3.0)
plt.xlabel("Energy (eV)")
plt.title('$\sigma$=0.5eV')

plt.subplot(339)
plt.plot(ticks, spectra_05[-1], linewidth=3.0)
plt.xlabel("Energy (eV)")
plt.title('$\sigma$=0.5eV')

plt.subplots_adjust(hspace = 0.4)
fig.suptitle("(a)                                (b)                                 (c)")

fig
plt.show()
