import numpy as np
import sys
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow
from silx.gui.colors import Colormap
from bcdiass.utils import rectify_sample
import matplotlib.pyplot as plt
import h5py
import matplotlib
matplotlib.rc('font', size=6)

INPUT, OUTPUT = range(2)
STRAIN = '1.00'
EXTRA_ROLL = 0 #-1

input_amplitudes, output_amplitudes, input_phases, output_phases = [], [], [], []

for WHAT in (INPUT, OUTPUT,):

    # load the data
    theta = 15.0
    if WHAT == INPUT:
        # input
        data = np.load('data/simulated_%s.npz'%STRAIN)['particle'][5:25, 64-10:64+10, 64-10:64+10]
        data = np.flip(data, axis=0)
        # these numbers are known from the simulation but weren't recorded:
        dr = 5e-09, 4.57e-09, 4.40e-09
#        print('*** dr = ', dr)
    elif WHAT == OUTPUT:
        Q = np.load('data/assembled_%s.npz'%STRAIN)['Q']
        sh = np.load('data/assembled_%s.npz'%STRAIN)['W'].shape
        dQ = Q / np.array(sh)
        costheta = np.cos(theta / 180 * np.pi)
        with h5py.File('data/modes_%s.h5'%(STRAIN.replace('.','')), 'r') as fp:
            data = fp['entry_1/data_1/data'][0]
#            print(data.shape)
        sh = np.load('data/prepared_%s.npz'%(STRAIN.replace('.','')))['data'].shape
        Q = dQ * np.array(sh)
        # this is the half-period resolution, the real-space pixel size:
        dr = 2 * np.pi / Q / np.array((costheta, costheta, 1))
        dr = np.array(dr)
#        print('*** dr = ', dr)
        # the pixel size along the third dimension Dmax is an input parameter,
        # adjusting here for comparison with the ground truth.
        dr[0] = dr[0] * .9
    print(data.shape)
    data, psize = rectify_sample(data, dr, theta, interp=5, find_order=False)
#    print('psize = ', psize)

    # center the particle and shift the phase
    tmp = (np.abs(data) > np.abs(data).max() / 3).astype(int)
    com = np.sum(tmp * np.indices(data.shape), axis=(1,2,3)) / np.sum(tmp)
    com = com.astype(int)
    shift = np.array(data.shape)//2 - com
    data = np.roll(data, shift, axis=(0,1,2))
    if WHAT == OUTPUT:
        av = np.mean(np.angle(data[com[0]-2:com[0]+2, com[1]-2:com[1]+2, com[2]-2:com[2]+2, ]))
        data[:] *= np.exp(-1j * av)
        data = np.roll(data, EXTRA_ROLL, axis=0)

    ### silx 3D plot
    if True:
        # Create a SceneWindow widget in an app
        app = qt.QApplication([])
        window = SceneWindow()

        # Get the SceneWidget contained in the window and set its colors
        widget = window.getSceneWidget()
        widget.setBackgroundColor((1., 1., 1., 1.))
        widget.setForegroundColor((1., 1., 1., 1.)) # the box color
        widget.setTextColor((0, 0, 0, 1.))

        # change the camera angle, there are no silx API calls for this...
        # direction is the line of sight of the camera and up is the direction
        # pointing upward in the screen plane 
        # from experimentation these axes are are [y z x]
        widget.viewport.camera.extrinsic.setOrientation(direction=[0.9421028 , 0.0316029 , 0.33383173],
                                                        up=[-0.03702362,  0.99926555,  0.00988633])
        widget.centerScene()

        # add the volume, which will be made complex based on data.dtype
        volume = widget.addVolume(data)
        volume.setScale(*(1e9*psize,)*3)

        # our array is indexed as x z y, with z indexed backwards
        # they expect the data as z y x.
        # it might be flipped somehow but the axes are correct now
        widget._sceneGroup.setAxesLabels(xlabel='Y', ylabel='Z', zlabel='X')

        # add and manipulate an isosurface, of type ComplexIsosurface
        volume.addIsosurface(np.abs(data).max()/4, 'r')
        iso = volume.getIsosurfaces()[0]
        #iso.setComplexMode('phase')
        #iso.setColormap(Colormap('jet', vmin=-.5, vmax=.5))

        # disable the default cut plane
        cuts = volume.getCutPlanes()
        [cut.setVisible(False) for cut in cuts]

        window.show()

        # Display exception in a pop-up message box
        sys.excepthook = qt.exceptionHandler

        # Run Qt event loop
        app.exec_()

    ### normal slice plot

    # cut out the good part
    N = 18
    a, b, c = data.shape
    data = data[a//2-N//2:a//2+N//2, b//2-N//2:b//2+N//2, c//2-N//2:c//2+N//2, ]

    # threshold the volume
    dmax = np.abs(data).max()
#    print(dmax)
    #data[np.abs(data) < dmax / 3] = np.nan
    mask = (np.abs(data) > dmax / 5) * 1.
    mask[np.where(mask < .5)] = np.nan

    w, n= 3+3/8, 4
    depths = [-6, -4, -2, 0, 2, 4, 6]
    fig, ax = plt.subplots(ncols=len(depths), nrows=2, figsize=(w, w/n))
    #fig, ax = plt.subplots(ncols=len(depths), nrows=2, figsize=(8, 3))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=.01, left=.05, right=.85, top=.85)
    vmax = float(STRAIN) / 2
    vmin = -vmax
    print('the extent of each image is ', psize*data.shape[-1])
    cax = fig.add_axes([.87, .05, .02, .38])
    a = psize * data.shape[-1] / 2
    exts = [-a, a, -a, a]
    for i, offset in enumerate(depths):
        ampl = np.abs((data*mask)[N//2+offset])
        phase = np.angle((data*mask)[N//2+offset])
        if WHAT == INPUT:
            input_amplitudes.append(np.abs((data)[N//2+offset]))
            input_phases.append(np.angle((data)[N//2+offset]))
        else:
            output_amplitudes.append(np.abs((data)[N//2+offset]))
            output_phases.append(np.angle((data)[N//2+offset]))

        diff = np.diff(phase, axis=0)
        aim = ax[0, i].imshow(ampl, vmax=dmax*.7, vmin=0, cmap='viridis', extent=exts)
        pim = ax[1, i].imshow(phase, cmap='jet', vmin=vmin, vmax=vmax, extent=exts)
        #ax[2, i].imshow(diff, cmap='jet', vmin=vmin/2, vmax=vmax/2)
        sign = {True:'+', False:'-'}[offset>=0]
        ax[0, i].set_title(sign + '%.1f nm'%abs(offset*psize*1e9),
                           pad=1,
                           fontdict={'fontsize':6})
        ax[0, i].set_xticks([]); ax[0, i].set_yticks([])
        ax[1, i].set_xticks([]); ax[1, i].set_yticks([])

    plt.colorbar(cax=cax, mappable=pim)
    cax.set_ylabel('rad.', fontdict={'fontsize':6}, labelpad=0)
    cax.yaxis.set_label_position("right")
    cax.tick_params(axis="y", labelsize=4, pad=2)

    ax[0, 0].set_ylabel('ampl.', fontdict={'fontsize':6})
    ax[1, 0].set_ylabel('phase')
    # for inkscaping the scalebar later:
    ax[0, -1].set_yticks([-25e-9, 25e-9])
    ax[0, -1].set_yticklabels([])
    ax[0, -1].yaxis.tick_right()
    plt.savefig('particle_plot_%s.png'%{True:'input',False:'output'}[WHAT==INPUT], dpi=600)


# now make difference maps
fig, ax = plt.subplots(nrows=2, ncols=len(input_phases))
for i in range(len(input_phases)):
    output_amplitudes[i] = np.roll(output_amplitudes[i], shift=(1,1), axis=(0,1))
    ax[0, i].imshow(output_amplitudes[i]/1000-input_amplitudes[i], vmin=-.2, vmax=.2)
    ax[1, i].imshow(output_phases[i]-input_phases[i], vmin=vmin, vmax=vmax, cmap='jet')

plt.show()
