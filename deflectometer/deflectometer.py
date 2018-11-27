"""
Deflectometer main File
"""

import numpy as np
import os
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

# setup plotting window
plt.close('all')
try:
    cam.release()
except:
    pass
plt.ion()
# mpl.rcParams['toolbar'] = 'None'
mpl.rcParams['axes.facecolor'] = 'black'
mpl.rc('axes.spines', top=False, bottom=False, left=False, right=False)


class Camera():
    """basic camera data"""

    def __init__(self, camID=0, dist_map=False):
        self.camID = camID
        self.cam = cv.VideoCapture(camID)
        self.pol_cam_meta()
        if dist_map is not False:
            self.dist_map = self.distortion_map(dist_map)
        else:
            self.dist_map = np.ones(self.shape)
        self.calc_noise()

    def poll_cam_meta(self):
        """Returns a dict of the current base camera properties"""

        self.shape = (int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT)), int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH)))
        self.fps = self.cam.get(cv.CAP_PROP_FPS)
        self.brightness = self.cam.get(cv.CAP_PROP_BRIGHTNESS)
        self.gain = self.cam.get(cv.CAP_PROP_GAIN)
        self.exposure = self.cam.get(cv.CAP_PROP_EXPOSURE)
        self.frame_count = self.cam.get(cv.CAP_PROP_FRAME_COUNT)

    def distortion_map(self, dmap):
        pass

    def ifov(self, ffov=30, orientation='width'):
        """ function to add ifov to cam object. Needed to cal part location"""
        ffov *= np.pi / 180  # convert to radians
        self.poll_cam_meta()
        if orientation == 'width':
            self.ifov = ffov / self.shape[1]
        elif orientation == 'height':
            self.ifov = ffov / self.shape[0]
        xang = (np.arange(0, self.shape[1]) - self.shape[1] / 2) * self.ifov
        yang = (np.linspace(0, self.shape[0]) - self.shape[1] / 2) * self.ifov
        self.fov_array = np.meshgrid(xang, yang)

    def calc_noise(self, nframes=30):
        """ Collect dark noise for camera """
        self.avg_noise = 10

class Part(object):
    """basic part properties"""

    def __init__(self, radius=np.inf, k=0, sdia=1):
        self.radius = radius
        self.k = k
        self.sdia = sdia

    def gen_sagmap(self, camera):
        xv = np.arange(-camera.shape[1] // 2, camera.shape[1] // 2)
        yv = np.arange(-camera.shape[0] // 2, camera.shape[0] // 2)
        xx, yy = np.meshgrid(xv, yv)
        rr = xx**2 + yy**2
        if self.radius == 0:
            self.sag_map = np.zeros(camera.shape)
        else:
            self.sag_map = (rr / self.radius) / (1 + np.sqrt(1 - (1 - self.k) * (rr / self.radius**2)))



def create_window():
    """Opens a figure window of size and returns linear dimension in inches"""
    scanner_fig = plt.figure()
    # scan_mng = plt.get_current_fig_manager()
    # scan_mng.full_screen_toggle()
    scanner_size = scanner_fig.get_size_inches() * scanner_fig.dpi
    return scanner_fig, scanner_size


# def start_cam(camID=0):
#     cam_obj = cv.VideoCapture(camID)
#     # ret, frame = cam_obj.read()
#     # Our operations on the frame come here
#     # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     # cv.imshow('frame',gray)
#     return cam_obj


def calc_separation(fig, ar, cam, frame, lit_pix=(0, 0), screen_cam=(-150, 0), part_dist=1000):
    """units are in mm

    Two lateral separations are required for the deflectomer (lit Screen-lit part, lit part-camera)

    --

    Math:
    L = distance from cam to screen
    Z = axial distance between the part and the camera-screen aparatus
    IFOV = Camera IFOV
    campix = lit camera pixel
    centerpix = central pixel that notes boresight of the camera ie angle of zero-zero

    theta = IFOV * (campix-centerpix)  -- the angle of the illuminated point on the part as seen by the camera
    xpc = Z * tan(theta)  -- the linear distance between the camera and the illuminated point on the part
    xsp = L - xpc  -- the linear distance between the screen's lit pixel and the illuminated point on the part


    Coordinate frame notes: it may help to treat the screen as being a negative distance away from the camera with smaller pixels indicies being closer to the camera


    Parameters
    -----------
    fig : matplotlib.pyplot.figure object

    ar : ndarray (int)  -- pixelated screen projection array

    lit_x : tuple(int) -- lit pixel of screen in (Y, X) i.e. (Row, Col) C-style format

    screen_cam : tuple(int) -- distance from cam to center of screen in form (Y, X). Coordinate frame zero is defined as center of camera 'pinhole'

    Returns
    -----------

    x_sep : tuple (float) -- 2 term tuple of the x separation arrays of (screen-part, part-cam)

    y_sep : tuple (float) -- 2 term tuple of the y separation arrays of (screen-part, part-cam)
    """
    # import pdb; pdb.set_trace()

    # find center of screen
    centerx = ar.shape[1] // 2
    centery = ar.shape[0] // 2
    screen_size = fig.get_size_inches() * 25.4  # in mm

    # Calculate separation of lit pix from camera (L from equations above)
    screencamx_sep = (lit_pix[1] - centerx) / ar.shape[1] * screen_size[0] + screen_cam[1]
    screencamy_sep = (centery - lit_pix[0]) / ar.shape[0] * screen_size[1] + screen_cam[0]

    frame_x, frame_y = find_in_frame(cam, frame, threshold=150)

    # Calculate xpc and xsp for the X and Y vectors
    camx_sep = np.tan(frame_x) * part_dist
    camy_sep = np.tan(frame_y) * part_dist

    # Calculate xsp for the X and Y vectors and combine into tuple with xpc
    x_sep = (camx_sep, screencamx_sep - camx_sep)
    y_sep = (camy_sep, screencamy_sep - camy_sep)
    return x_sep, y_sep


def slope_val(pixloc, partloc, camloc, zscreen_part, zcam_part, partsag=0):
    """ Simplified slope vals... needs update for fast parts!"""

    # dscreen_part = np.sqrt(zscreen_part**2 + (pixloc-partloc)**2)
    # dcam_part = np.sqrt(zscreen_part**2 + (camloc-partloc)**2)
    t1 = (partloc - pixloc) / zscreen_part
    t2 = (partloc - camloc) / zcam_part
    t3 = 1
    t4 = 1
    return (t1 + t2) / (t3 + t4)


def find_in_frame(cam, frame, threshold=150):
    """
    Calculates points illuminated in frame based on centroiding method

    Parameters
    ----------
    slope : ndarray (float)  - Array containing measured slopes

    frame  : ndarray  - Camera frame

    avg_ noise : int, float, or ndarray (of same size as frame) - noise floor or NUC map

    threshold : minimum value to be considered "illuminated"

    Returns
    ---------
    angle_ar_x : ndarray (float)  -- array of x angles corresponding to lit distances on part

    angle_ar_y : ndarray (float)  -- array of y angles corresponding to lit distances on part
    """

    # remove noise
    frame[frame < (cam.avg_noise + threshold)] = 0
    angle_ar = np.zeros(frame.shape)
    angle_ar[frame > 0] = 1
    angle_ar_x = angle_ar * cam.fov_array[0]
    angle_ar_y = angle_ar * cam.fov_array[1]
    return angle_ar_x, angle_ar_y

def combine_maps(ar1, ar2=None):
    """combines two image arrays into single output. Indicies that have non-zero values are averaged"""
    if ar2 is None:
        return ar1
    else:
        val_mask = np.logical_and(ar1 != 0, ar2 != 0)
        outarray = ar1 + ar2
        outarray[val_mask] /= 2
        return outarray

def scranner(screen_ar, barsize=10, dist=1000, camh=150):
    """ Primary function for capturing deflectometer data"""
    camfig = plt.figure()
    mgr = plt.get_current_fig_manager()
    mgr.window.setGeometry(900, 30, 320, 240)
    camax = plt.gca()
    # plt.axes([0, 0, 1, 1], frameon=False)
    camax.set_axis_off()
    try:
        win = camfig.canvas.manager.window
    except AttributeError:
        win = camfig.canvas.window()
    toolbar = win.findChild(QtWidgets.QToolBar)
    toolbar.setVisible(False)

    fig = plt.figure(facecolor='black')
    mgr = plt.get_current_fig_manager()
    mgr.window.setGeometry(10, 30, 650, 510)
    try:
        win = fig.canvas.manager.window
    except AttributeError:
        win = fig.canvas.window()
    toolbar = win.findChild(QtWidgets.QToolBar)
    toolbar.setVisible(False)

    plt.axes([0, 0, 1, 1], frameon=False)
    ax = plt.gca()

    # start cam
    cam_obj = Camera(camID=0)
    ret, background = cam_obj.cam.read()
    bkg_gray = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    camimg = camax.imshow(bkg_gray, cmap='gray')
    # setup and plot initial array
    # screen_ar[0, :] = 255
    screen_ar[0, :] = 0
    img = ax.imshow(screen_ar, cmap='gray', vmin=0, vmax=255)

    # setup output array with same resolution as camera
    x_slopemap = np.zeros(cam_obj.shape)
    y_slopemap = np.zeros(cam_obj.shape)
    # test variable. Remove cam_frame for real use
    cam_frame = x_slopemap.copy()
    for cdx in range(0, screen_ar.shape[1], barsize):
        # import pdb; pdb.set_trace()

        if cdx >= barsize:
            screen_ar[:, cdx - barsize] = 0
        screen_ar[:, cdx] = 255
        img.set_data(screen_ar)
        plt.pause(0.001)
        ret, frame = cam_obj.cam.read()
        # Our operations on the frame come here
        gray = np.abs(cv.cvtColor(frame, cv.COLOR_BGR2GRAY) - bkg_gray)
        # Display the resulting frame - background frame
        camimg.set_data(gray)
        camfig.canvas.draw()
        # plt.pause(0.001)

        # calculate separation value for X
        xsep, _ = calc_separation(fig, screen_ar, lit_x=cdx, lit_y=0, cam_x=0, cam_y=camh, )
        # import pdb; pdb.set_trace()
        x_slopemap_frame = pix_slope(slope=calc_slope(xsep, dist=dist), frame=gray)
        x_slopemap = combine_maps(x_slopemap, x_slopemap_frame)
        print(cdx, ' - ', xsep)

    screen_ar[:, cdx] = 0

    for rdy in range(0, screen_ar.shape[0], barsize):
        if rdy >= barsize:
            screen_ar[rdy - barsize:rdy, :] = 0
        screen_ar[rdy, :] = 255
        img.set_data(screen_ar)
        plt.pause(0.001)

        ret, frame = cam.read()
        # Our operations on the frame come here
        gray = np.abs(cv.cvtColor(frame, cv.COLOR_BGR2GRAY) - bkg_gray)
        # Display the resulting frame - background frame
        camimg.set_data(gray)
        camfig.canvas.draw()

        _, ysep = calc_separation(fig, screen_ar, lit_x=0, lit_y=rdy, cam_x=0, cam_y=camh, )
        y_slopemap_frame = pix_slope(slope=calc_slope(ysep, dist=dist), frame=gray)
        y_slopemap = combine_maps(y_slopemap, y_slopemap_frame)

        print(rdy, ' - ', ysep)

    # blank screen
    screen_ar = np.zeros(screen_ar.shape)
    plt.imshow(screen_ar, cmap='gray')
    cam.release()

    return x_slopemap, y_slopemap

if __name__ == "__main__":
    # fig, scansize = create_window()
    screen = np.zeros((48, 64))
    img = scranner(screen, barsize=1)
    dir = r'/usr/share/opencv/samples/data'
    # img = cv.imread(os.path.join(dir, 'messi5.jpg'), 0)
    # cv.namedWindow('image', cv.WINDOW_NORMAL & cv.WINDOW_GUI_EXPANDED)
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(img[0])
    ax1.set_title('X Slope')
    ax2.imshow(img[1])
    ax2.set_title('Y Slope')
