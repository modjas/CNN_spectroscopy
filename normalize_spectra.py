import numpy as np
from sklearn.preprocessing import normalize

spec = np.load('spectra.npz')['spectra']

print(spec[0][0])
print(spec[0][0]/np.linalg.norm(spec[0]))
out = []

for i in spec:
	norm = np.linalg.norm(i)
	out.append(i/norm)


np.savez('spectra_normalized.npz', spectra=out)