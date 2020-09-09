import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.ion()
matplotlib.rc('font', size=6)

# load the simulation output
with np.load('data/simulated_0.10.npz') as dct:
    frames = dct['frames']

w, n= 3+3/8, 8
fig, ax = plt.subplots(ncols=n, nrows=1, figsize=(w, w/n))
plt.subplots_adjust(bottom=.02, left=.02, right=.98, top=.98, wspace=.05, hspace=.08)
for i, a_ in enumerate(ax):
    a_.set_yticks([])
    a_.set_xticks([])
    a_.set_frame_on(False)
    frame_ = frames[::frames.shape[0]//n][i]
    a_.imshow(np.log10(frame_), vmin=0, vmax=np.log10(frames.max()))

plt.savefig('frames_plot.png', dpi=600)
