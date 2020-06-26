import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=6)

STRAINLIM = 8e-4

# load and prepare the data
with np.load('data/rectified.npz') as dct:
    data = np.load('data/rectified.npz')['data']
    psize = np.load('data/rectified.npz')['psize']
m, n = 43, 57
data = data[m:n, m:n, m:n]
data[np.abs(data) < np.abs(data).max()/4] = np.nan

offsets = np.arange(0, data.shape[0], 1)
fig, ax = plt.subplots(nrows=1, ncols=len(offsets), figsize=(4.58, .6))
plt.subplots_adjust(wspace=0, hspace=0, left=.01, right=.99, bottom=.2, top=.9)

dx = (offsets - 7) * psize
G = 2.669e10 # 1/m
for i, offset in enumerate(offsets):
    phase = np.angle(data[offset])
    strain = -np.diff(phase, axis=0) / G / psize
    w = data.shape[-1] * psize
    ax[i].imshow(np.flip(strain, axis=[0,1]), cmap='jet', vmin=-STRAINLIM, vmax=STRAINLIM,
        extent=[-w/2, w/2, -w/2, w/2])
    plt.setp(ax[i], 'xticks', [], 'yticks', [])
    sign = {True: '+', False:'-'}[dx[i] >= 0]
    ax[i].set_xlabel(sign + ('%d'%round(abs(dx[i])*1e9)), labelpad=0)
    ax[i].set_frame_on(False)

# make 60 nm marks for the inkscape scalebar
ax[0].plot([-30e-9, 30e-9], [0, 0], 'k,')

plt.savefig('slice_plot.png', dpi=600)
plt.show()
