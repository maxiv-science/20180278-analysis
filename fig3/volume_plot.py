import numpy as np
import sys
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow
from silx.gui.colors import Colormap
from inflate_phase import inflate_phase

CUTOFF = 50.
WHAT = 'solid'
WHAT = 'slice'
STRAINLIMS = [-8e-4, 8e-4]
PLANE = 'along'
PLANE = 'across'

# load the data and calculate the strain
data = np.load('data/rectified.npz')['data']
psize = np.load('data/rectified.npz')['psize']

# crop
N = data.shape[0]
n = 8
#data = data[N//2-9:N//2+9, N//2-6:N//2+8, N//2-8:N//2+8]
data = data[N//2-9:N//2+9, N//2-8:N//2+10, N//2-9:N//2+9]

# (optionally) convert phase to duz/dz
phase = np.angle(data)
strain = np.zeros_like(phase)
for j in range(1, strain.shape[1]-1):
    strain[:, j, :] = -(phase[:, j+1, :] - phase[:, j-1, :]) / 2
G = 2.669e10 # 1/m
strain = strain / G / psize
data = np.abs(data) * np.exp(1j * strain)

if WHAT == 'solid':
    # optionally extrapolate the phases outisde a stricter cutoff to their
    # nearest point within this cutoff.
    data = inflate_phase(data, 200)
elif WHAT == 'slice':
    # set pixels outside the cutoff to phase -3, so they fall below
    # vmin and are excluded in the cut slice
    mask = (np.abs(data) >= CUTOFF).astype(int)
    data = mask * data + (1 - mask) * np.abs(data) * np.exp(-3j)
    
# Create a SceneWindow widget in an app
app = qt.QApplication([])
window = SceneWindow()

# Get the SceneWidget contained in the window and set its colors
widget = window.getSceneWidget()
widget.setBackgroundColor((1., 1., 1., 1.))
widget.setForegroundColor((.5, .5, .5, 1.)) # the box color
widget.setTextColor((0, 0, 0, 1.))

# change the camera angle, there are no silx API calls for this...
# direction is the line of sight of the camera and up is the direction
# pointing upward in the screen plane 
# from experimentation these axes are are [y z x]
# you can get them with
# widget.viewport.camera.extrinsic.direction
# widget.viewport.camera.extrinsic.up
camd, camu = [1.,0,.6], [0, 1, 0]
camd, camu = [.86, -.27, .41], [.24, .96, .14]
widget.viewport.camera.extrinsic.setOrientation(direction=camd, up=camu)
widget.centerScene()

# add the volume, which will be made complex based on data.dtype
volume = widget.addVolume(data)
volume.setScale(*(1e9*psize,)*3)

# clean up some crap
volume.setBoundingBoxVisible(False)
group = volume.parent()
group.setBoundingBoxVisible(False)
widget.setOrientationIndicatorVisible(False)

# our array is indexed as x z y, with z indexed backwards
# they expect the data as z y x.
# it might be flipped somehow but the axes are correct now
widget._sceneGroup.setAxesLabels(xlabel='Y', ylabel='Z', zlabel='X')

# add and manipulate an isosurface, of type ComplexIsosurface
if WHAT == 'solid':
    iso = volume.addIsosurface(CUTOFF, color=[1,0,0,1.])
    iso.setComplexMode('phase')
    iso.setColormap(Colormap('jet', vmin=STRAINLIMS[0], vmax=STRAINLIMS[1]))
else:
    iso = volume.addIsosurface(CUTOFF, color=[1,0,0,.4])

# modify the default cut plane
cuts = volume.getCutPlanes()
if WHAT == 'solid':
    [cut.setVisible(False) for cut in cuts]
else:
    cut = cuts[0]
    cut.setComplexMode(cut.ComplexMode.PHASE)
    if PLANE == 'across':
        cut.setParameters([0,0,1,-8.5])
        print('top edge of cut plane is %d nm' % round(data.shape[-1] * psize * 1e9))
    elif PLANE == 'along':
        cut.setParameters([1,0,0,-9])
    cut.setDisplayValuesBelowMin(False)
    cut.setColormap(Colormap('jet', vmin=STRAINLIMS[0], vmax=STRAINLIMS[1]))

# save and show
window.show()
#im = widget.grabGL()
#im.save('volume_%s.png'%WHAT)

# Display exception in a pop-up message box
sys.excepthook = qt.exceptionHandler


####### make a colorbar in a normal mpl window
import matplotlib.pyplot as plt
import matplotlib
plt.ion()
matplotlib.rc('font', size=6)
fig = plt.figure(figsize=(.8,.4))
im = plt.imshow(10000*np.linspace(STRAINLIMS[0], STRAINLIMS[1], 100).reshape((10,10)), cmap='jet')
plt.gca().set_visible(False)
cax = plt.axes([.1,.8,.8,.2])
cbar = plt.colorbar(mappable=im, cax=cax, orientation='horizontal')
cax.set_xlabel('strain (1e-4)', labelpad=1)
plt.savefig('volume_colorbar.png', dpi=600)


###### make a cartoon in another mpl window
from nmutils.utils.bodies import TruncatedOctahedron
import mpl_toolkits.mplot3d as a3
# make a particle of diameter=1, lying on its xy plane
scale = .72
trunc = TruncatedOctahedron(scale=scale)
trunc.shift([-.5, -.5, -.5])
trunc.rotate('z', 45)
trunc.rotate('y', 109.5/2)
trunc.scale(1/scale)

_xy = np.sqrt(camd[0]**2 + camd[1]**2)
el = 7
el = 15
az = 70
print(el, az)
ax = a3.Axes3D(plt.figure(figsize=(3,3)), azim=az, elev=el)
ax.axis('off')
faces = trunc.faces()
for face in faces:
    poly = a3.art3d.Poly3DCollection([face])
    poly.set_color((1,1,1,.8))
    poly.set_edgecolor('k')
    ax.add_collection3d(poly)
    lims = (-.5, .5)
    plt.setp(ax, 'xlim', lims, 'ylim', lims, 'zlim', lims)
plt.savefig('volume_cartoon.png', dpi=600)

# Run Qt event loop
app.exec_()
