import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=6)

# load the assembly results
with np.load('data/assembled_10.npz') as dct:
    Pjlk = dct['Pjlk']
    rolls = dct['rolls']
    Q3 = dct['Q'][0]
before, after = -rolls.min(), rolls.max()
Pjlk = np.pad(Pjlk, pad_width=((0,0), (before, after), (0, 0)))
for k_ in range(Pjlk.shape[2]):
    Pjlk[:, :, k_] = np.roll(Pjlk[:, :, k_], rolls[k_], axis=-1)

# calculate the angular ranges
dq3 = Q3 / Pjlk.shape[0]
G111 = 2.669e10 # 1/m
dtheta = dq3 / G111 / np.pi * 180
dx = 2 * np.pi / Q3
domega = 1/4000/np.pi*180# from the assembly settings and detector geometry

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(1.6, 2))
plt.subplots_adjust(bottom=.2, left=.18, right=.76, top=.98, wspace=.05, hspace=.08)
ax[0].imshow(np.flip(Pjlk.sum(axis=1), axis=0), aspect='auto', vmin=0, vmax=1.)
ax[1].imshow(Pjlk.sum(axis=0), aspect='auto', vmin=0, vmax=1.)

ax[0].set_ylabel('$j$-index', labelpad=2)
ax[0].set_xticklabels([])
ax0_ = ax[0].twinx()
ax0_.set_ylim(np.array((-.5,.5)) * Pjlk.shape[0] * dtheta)
ax0_.set_ylabel('$\\theta$ (degrees)', labelpad=2)

ax[1].set_ylabel('$l$-index', labelpad=2)
ax[1].set_xlabel('frame number $k$')
ax1_ = ax[1].twinx()
ax1_.set_ylim(np.array((-.5,.5)) * Pjlk.shape[1] * domega)
ax1_.set_ylabel('$\\omega$ (degrees)', labelpad=2)

for a_ in (ax[0], ax[1], ax0_, ax1_):
    a_.tick_params(axis='y', which='major', pad=1)

plt.savefig('assembly_plot.png', dpi=600)
plt.show()
