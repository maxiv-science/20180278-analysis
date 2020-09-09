import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.ion()
matplotlib.rc('font', size=6)

STRAIN = '1.00'

# load the simulation input
with np.load('data/simulated_%s.npz'%STRAIN) as dct:
    degs_per_pixel = .025 # 2theta=30deg, d=250mm, psize=55e-6
    omega = dct['rolls'] * degs_per_pixel
    theta = dct['offsets']

# load the assembly results
with np.load('data/assembled_%s.npz'%STRAIN) as dct:
    Pjlk = dct['Pjlk']
    #Pjlk = np.flip(Pjlk, axis=0)
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

fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(3+3/8, 2.))
plt.subplots_adjust(bottom=.18, left=.15, right=.85, top=.98, wspace=.05, hspace=.08)
ax[0, 0].plot(theta, 'k')
ax[0, 1].plot(omega, 'k')
Theta = dtheta * Pjlk.shape[0]
ax[1, 0].imshow(Pjlk.sum(axis=1), aspect='auto', vmin=0, vmax=1., extent=(0,Pjlk.shape[-1],-Theta/2,Theta/2))
Omega = Pjlk.shape[1] * degs_per_pixel
ax[1, 1].imshow(Pjlk.sum(axis=0), aspect='auto', vmin=0, vmax=1., extent=(0,Pjlk.shape[-1],-Omega/2,Omega/2))

Pjk = np.sum(Pjlk, axis=1)
j_inds = np.indices(Pjk.shape)[0]
j_com = np.sum(j_inds * Pjk, axis=0) / np.sum(Pjk, axis=0)
theta_ass = dtheta * (Pjlk.shape[0]/2 - j_com)
Plk = np.sum(Pjlk, axis=0)
l_inds = np.indices(Plk.shape)[0]
l_com = np.sum(l_inds * Plk, axis=0) / np.sum(Plk, axis=0)
omega_ass = degs_per_pixel * (Pjlk.shape[1]/2 - l_com)
#theta_ass = -(np.argmax(np.flip(Pjlk.sum(axis=1)), axis=0) - Pjlk.shape[0]/2) * dtheta
#omega_ass = -(np.argmax(Pjlk.sum(axis=0), axis=0) - Pjlk.shape[1]/2) * degs_per_pixel
ax[2, 0].plot(theta_ass - theta)
ax[2, 1].plot(omega_ass - omega)
#ax[2, 0].set_ylim([-.15, .15])
#ax[2, 1].set_ylim([-.05, .05])

ax[0, 1].yaxis.tick_right()
ax[1, 1].yaxis.tick_right()
ax[2, 1].yaxis.tick_right()
ax[0, 1].yaxis.set_label_position("right")
ax[1, 1].yaxis.set_label_position("right")
ax[2, 1].yaxis.set_label_position("right")
ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])
ax[1, 0].set_xticklabels([])
ax[1, 1].set_xticklabels([])
ax[1, 0].set_xticks(list(ax[1,0].get_xticks())[:-1])
ax[0, 0].set_ylabel('$\\delta\\theta$ (deg)')
ax[0, 1].set_ylabel('$\\delta\\omega$ (deg)')
ax[1, 0].set_ylabel('$\\delta\\theta$ (deg)')
ax[1, 1].set_ylabel('$\\delta\\omega$ (deg)')
ax[2, 0].set_ylabel('residual')
ax[2, 1].set_ylabel('residual')
ax[1, 0].set_xlabel('frame number $k$').set_x(1.0)
ax[1, 1].set_ylim(ax[0, 1].get_ylim())
ax[1, 0].set_ylim(ax[0, 0].get_ylim())
ax[2, 1].set_ylim(np.array(ax[0, 1].get_ylim())/5)
ax[2, 0].set_ylim(np.array(ax[0, 0].get_ylim())/5)

#ax[1, 0].plot(theta, 'r')

plt.savefig('assembly_plot.png', dpi=600)
