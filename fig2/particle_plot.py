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

INPUT, OUTPUT, REGULAR = range(3)
STRAIN = '1.00'

input_amplitudes, output_amplitudes, input_phases, output_phases = [], [], [], []
regular_amplitudes, regular_phases = [], []

for WHAT in (INPUT, OUTPUT, REGULAR):#(INPUT, OUTPUT, REGULAR,):

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
    elif WHAT == REGULAR:
        with h5py.File('data/modes_%sregular.h5'%STRAIN, 'r') as fp:
            data = fp['entry_1/data_1/data'][0]
        data = np.flip(data, axis=0)
        data = np.pad(data, 10, mode='constant', )
        dr = np.array((5e-09, 4.57e-09, 4.40e-09))
        dr[0] = dr[0] * .85
    print(data.shape)
#    print('psize = ', psize)

    # manually remove phase ramps
    if WHAT == OUTPUT:
        inds = np.indices(data.shape)
        slopes = np.array([.025, -.015, 0])
        #gradient = slope * np.arange(data.shape[0]).reshape((-1,1,1))
        gradient = np.sum(inds * slopes.reshape((3, 1, 1, 1)), axis=0)
        data *= np.exp(1j * gradient)
    elif WHAT == REGULAR:
        inds = np.indices(data.shape)
        slopes = np.array([-.025, -.02, -.01])
        #gradient = slope * np.arange(data.shape[0]).reshape((-1,1,1))
        gradient = np.sum(inds * slopes.reshape((3, 1, 1, 1)), axis=0)
        data *= np.exp(1j * gradient)

    # center the particle and shift the phase
    tmp = (np.abs(data) > np.abs(data).max() / 3).astype(int)
    com = np.sum(tmp * np.indices(data.shape), axis=(1,2,3)) / np.sum(tmp)
    com = com.astype(int)
    shift = np.array(data.shape)//2 - com
    print(com)
    data = np.roll(data, shift, axis=(0,1,2))
    if WHAT in (OUTPUT, INPUT, REGULAR):
        av = np.mean(np.angle(data[com[0], com[1], com[2]]))
        data[:] *= np.exp(-1j * av)
        av = np.mean(np.angle(data[com[0]-2:com[0]+2, com[1]-2:com[1]+2, com[2]-2:com[2]+2, ]))        
        data[:] *= np.exp(-1j * av)

    # manually shift the particle if needed
    if WHAT == OUTPUT:
        data = np.roll(data, axis=(1,2), shift=(0,-1))
        data *= np.exp(1j * .1)
    elif WHAT == REGULAR:
        data = np.roll(data, axis=(0,1,2), shift=(-1,-1,-1))

    # now resample on the orthogonal grid
    data, psize = rectify_sample(data, dr, theta, interp=1, find_order=False)

    # cut out the good part
    N = 18
    a, b, c = data.shape
    data = data[a//2-N//2:a//2+N//2, b//2-N//2:b//2+N//2, c//2-N//2:c//2+N//2, ]

    ### silx 3D plot
    if False:
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
        widget.viewport.camera.extrinsic.setOrientation(#direction=[-0.9421028 , 0.0316029 , 0.33383173],
                                                        direction=[0 , 0 , 1],
                                                        #up=[-0.03702362,  0.99926555,  0.00988633])
                                                        up=[0, 1, 0])
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

        # clean up some crap
        volume.setBoundingBoxVisible(False)
        group = volume.parent()
        group.setBoundingBoxVisible(False)
        widget.setOrientationIndicatorVisible(False)


        window.show()

        # Display exception in a pop-up message box
        sys.excepthook = qt.exceptionHandler

        # Run Qt event loop
        app.exec_()

    ### normal slice plot

    # threshold the volume
    dmax = np.abs(data).max()
#    print(dmax)
    #data[np.abs(data) < dmax / 3] = np.nan
    mask = (np.abs(data) > dmax / 3) * 1.
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
        m_ = mask[N//2+offset]
        if WHAT == INPUT:
            input_amplitudes.append(np.abs((data)[N//2+offset]))
            input_phases.append(m_ * np.angle((data)[N//2+offset]))
        elif WHAT == OUTPUT:
            output_amplitudes.append(np.abs((data)[N//2+offset]))
            output_phases.append(np.angle((data)[N//2+offset]))
        else:
            regular_amplitudes.append(np.abs((data)[N//2+offset]))
            regular_phases.append(np.angle((data)[N//2+offset]))

        diff = np.diff(phase, axis=0)
        aim = ax[0, i].imshow(ampl, vmax=dmax*.7, vmin=0, cmap='viridis', extent=exts)
        pim = ax[1, i].imshow(phase, cmap='jet', vmin=vmin, vmax=vmax, extent=exts)
        #ax[2, i].imshow(diff, cmap='jet', vmin=vmin/2, vmax=vmax/2)
        sign = {True:'+', False:'-'}[offset>=0]
        ax[0, i].set_title(sign + '%.1f nm'%abs(offset*psize*1e9),
                           pad=1,
                           fontdict={'fontsize':5.5})
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
    plt.savefig('particle_plot_%s.png'%{INPUT:'input',OUTPUT:'output',REGULAR:'regular'}[WHAT], dpi=600)

# now make difference maps
ext = psize * data.shape[-1] * 1e9

def diff_map(ampl, phase, label):
    fig, ax = plt.subplots(nrows=2, ncols=len(input_phases), figsize=(6,1.5))
    pcax = fig.add_axes([.88, .1, .02, .3])
    acax = fig.add_axes([.88, .6, .02, .3])
    plt.subplots_adjust(left=.1, right=.86, bottom=.05, top=.95)
    for i in range(len(input_phases)):
        aim = ax[0, i].imshow(ampl[i], vmin=-.5, vmax=.5, extent=(-ext/2, ext/2, -ext/2, ext/2))
        pim = ax[1, i].imshow(phase[i], vmin=vmin, vmax=vmax, cmap='jet', extent=(-ext/2, ext/2, -ext/2, ext/2))
    plt.colorbar(aim, cax=acax)
    plt.colorbar(pim, cax=pcax)
    pcax.set_ylabel('(rad.)')
    for a in ax.flatten():
        a.set_xticks([])
        a.set_yticks([])
    for a in ax[:, 0]:
        plt.setp(a, 'yticks', [-30, 30], 'yticklabels', ['0 nm', '60 nm'])
    plt.savefig('si/%s.pdf'%label)

diff_map(np.array(output_amplitudes)/1000-np.array(input_amplitudes),
         np.array(output_phases)-np.array(input_phases),
         'output_vs_input')

diff_map(np.array(regular_amplitudes)/4000-np.array(input_amplitudes),
         np.array(regular_phases)-np.array(input_phases),
         'regular_vs_input')

plt.show()
