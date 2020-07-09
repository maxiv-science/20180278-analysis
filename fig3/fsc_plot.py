"""
Fourier shell correlation of reconstructed subsets.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=6)
import numpy as np

with np.load('data/validation.npz') as dct:
    q = dct['q']
    c = dct['fsc']
    c_cut = dct['fsc_cut']
    nri = dct['nri']

# half-bit threshold
# effective number of pixels
D = 60e-9
L = 100 * 4.8128e-09
neri = nri * (D / L)**2
neri[neri < 1] = 1
T = (0.2071 + 1.9102 / np.sqrt(neri)) / (1.2071 + 0.9102 / np.sqrt(neri))

# one-bit threshold
#T = (.5 + 2.4142 / np.sqrt(nri)) / (1.5 + 1.4124 / np.sqrt(nri))

fig, a1 = plt.subplots(figsize=(1.5,1.5))
fig.subplots_adjust(left=.25, bottom=.25, right=.97, top=.8)

a1.plot(q, np.real(c))
a1.plot(q, np.real(c_cut))
a1.plot(q, T, 'k--')
a1.set_xlabel('q = $2\pi/\Delta r$  [nm-1]', labelpad=2)
a1.set_ylabel('Fourier shell correlation', labelpad=2)
a1.set_xlim(-.05, 1.37)
a1.set_ylim(0, 1.1)
#a1.axhline(.5, color='k', linestyle=':')
a2 = a1.twiny()
a2.set_xlabel('Full-period resolution $\Delta r$  [nm]', labelpad=3).set_x(.45)
resolutions = (5, 6, 8, 10, 15, 30, 90)
a2.set_xticks([2*np.pi/dr for dr in resolutions])
a2.set_xticklabels(resolutions)
a2.set_xlim(a1.get_xlim())
a1.tick_params('x', pad=2)
a1.tick_params('y', pad=1)
a2.tick_params('x', pad=0)

plt.savefig('fsc_plot.png', dpi=600)

plt.show()