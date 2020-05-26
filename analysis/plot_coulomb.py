import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

coulomb = np.absolute(np.load(os.path.join(os.path.dirname(__file__),'../data/coulomb.npz'))['coulomb'])

coulomb17 = coulomb[16]

fig, ax = plt.subplots()
plt.rcParams.update({'font.size':18})
ax.set_xlabel('Atom i', fontsize=16)
ax.set_ylabel('Atom j', fontsize=16)
bars = ["C", "C", "C", "C", "N", "O", "H", "H", "H", "H", "H", "H", "H", "H", "H", "-"]
major_ticks = np.arange(0, 17, 1)

ax.matshow(coulomb17[:15, :15], cmap=plt.cm.gray_r)
ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)
ax.set_xticklabels(bars, fontsize = 12)
ax.set_yticklabels(bars, fontsize = 12)
ax.xaxis.set_label_position('top')
for i in range(16):
	for j in range(16):
		c = coulomb17[j,i]
		if i==j and i<6:
			ax.text(i,j,int(round(c,1)), va='center', ha='center', fontsize=10, color='white')
		else:
			ax.text(i,j,int(round(c,1)), va='center', ha='center', fontsize=10)




plt.show()