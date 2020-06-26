import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=6)

# load the assembly results
with np.load('data/assembled_10.npz') as dct:
    Pjlk = dct['Pjlk']
    rolls = dct['rolls']
before, after = -rolls.min(), rolls.max()
Pjlk = np.pad(Pjlk, pad_width=((0,0), (before, after), (0, 0)))
for k_ in range(Pjlk.shape[2]):
    Pjlk[:, :, k_] = np.roll(Pjlk[:, :, k_], rolls[k_], axis=-1)

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(1.5, 2))
plt.subplots_adjust(bottom=.2, left=.25, right=.99, top=.98, wspace=.05, hspace=.08)
ax[0].imshow(np.flip(Pjlk.sum(axis=1), axis=0), aspect='auto', vmin=0, vmax=1.)
ax[1].imshow(Pjlk.sum(axis=0), aspect='auto', vmin=0, vmax=1.)

ax[0].set_ylabel('$j$-index', labelpad=2)
ax[0].set_xticklabels([])
ax[1].set_ylabel('$l$-index', labelpad=2)
ax[1].set_xlabel('frame number $k$')

plt.savefig('assembly_plot.png', dpi=600)
plt.show()
