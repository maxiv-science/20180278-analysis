"""
Provides a function which takes a 3d complex array, and replaces
the phases where the amplitude is below a cutoff with the phase
of the nearest point with amplitude above the cutoff. This has the
effect of inflating the phases near the curoff surface outwards,
and helps give a better view of surface phases without having to
erode the particle.
"""

from scipy.interpolate import griddata
import numpy as np

def inflate_phase(arr, cutoff):
    ii, jj, kk = np.indices(arr.shape)
    ijk = np.stack([l.flatten() for l in (ii, jj, kk)], axis=1)
    arr_ = arr.flatten()
    outside = (np.abs(arr_) < cutoff)
    inside = ~outside
    missing_phases = griddata(points=ijk[inside],
                              values=np.angle(arr_[inside]),
                              xi=ijk[outside],
                              method='nearest')
    print(missing_phases)
    print(outside.shape)
    arr_[outside] = np.abs(arr_[outside]) * np.exp(1j * missing_phases)
    arr = np.reshape(arr_, arr.shape)
    return arr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    data = np.load('data/rectified.npz')['data']
    infl = inflate_phase(data, np.abs(data).max()/4)
    plt.imshow(np.angle(infl[infl.shape[0]//2]), vmin=-.2, vmax=.2, cmap='jet')
