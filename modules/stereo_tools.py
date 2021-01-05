import numpy as np
import cv2
import configparser
import math
import argparse
from modules.util import readXML


##############################################################################
class Resolution :
    def __init__(self,width, height):
        self.width = int(width)
        self.height = int( height)
##############################################################################


##############################################################################
# FIVE BASIC ARGUMENTS TO STEREO.PY
# --adjust : adjust image quality : escape by pressing "q", follow instruction on terminal MAKE SURE TO SELECT THE WINDOW WITH THE MOUSE
# --resolution : string (VGA,HD,FHD,...)
# --cam_ids : 0,1,2,3,4 .... (USB port linux_id)
# --depthmap : if want to see depth map instead of disparity
# --distance : if want to see display distance instead of depth

def parse_args():
    parser = argparse.ArgumentParser(description='stereo')

    parser.add_argument('--adjust', action='store_true', default=False,help='adjust image quality')
    parser.add_argument('--resolution', type=str, default='HD',help='Resolution : VGA, HD, FHD ,2K')
    parser.add_argument('--cam_ids', type=str, default=0,help='List of camera USB port id : 0,1,2,3,4...')
    parser.add_argument('--depthmap', action='store_true', default=False,help='view depth map instead of disparity map')
    parser.add_argument('--distance', action='store_true', default=False,help='display distance instead of depth')


    args = parser.parse_args()

    # Examples in command line for two USB inputs
    # python stereo.py --resolution='HD' --adjust --cam_ids=0,1

    # Examples in command line for one USB input
    # python stereo.py --resolution='HD' --adjust --cam_ids=0 --depth_map --distance

    return args


##############################################################################
def get_image_dimension_from_resolution(resolution):

    if resolution== 'VGA' :
 #       width=672
#        height=376
        width=800
        height=480
    elif resolution== 'HD' :
        width=1280
        height=720
    elif resolution== 'FHD' :
        width=1920
        height=1080
    elif resolution== '2K' :
        width=2208
        height=1242
    return width,height
##############################################################################



##############################################################################
# THIS FONCTION READS CALIBRATION FILE AND OUTPUTS MATRICES AND COEFFICIENTS
## SEE EXAMPLE SN20716499.conf FOR CORRECT FORMAT

def init_calibration(left_xml, right_xml, image_size, resolution) :


    # READ .XML FILE -------
    cameraMatrix_left,distCoeffs_left, _, _ ,_, E, F = readXML(left_xml) # left
    cameraMatrix_right,distCoeffs_right, R, T ,_, _, _ = readXML(right_xml) # right

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y, Q

###############################################################################






################################################################################
## THIS FONCTION RECTIFIES LEFT AND RIGHT FRAMES

def get_rectified_left_right(left_frame,right_frame, map_left_x, map_left_y, map_right_x, map_right_y):

    left_frame_rect = cv2.remap(left_frame, map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
    right_frame_rect = cv2.remap(right_frame, map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)

    return left_frame_rect,right_frame_rect
##############################################################################




#############################################################################
# THIS FONCTION DISPLAYS DISTANCES OR DEPTH ON MOUSE CLICK

def on_mouse_display_depth_value(event, x, y, flags, params):

    if (event == cv2.EVENT_LBUTTONDOWN):

        distance,cloud,frame,windowName = params
        dist=np.round(np.linalg.norm(cloud[y, x,:]),2)
        depth=np.round(cloud[y,x,2],2)

        if distance:
            display =dist
            disp_string='distance'
        else:
            display=depth
            disp_string='depth'

        cv2.line(frame[0],(x - 20, y), (x + 20, y), (255, 255, 255), 2)
        cv2.line(frame[0], (x,y - 20),(x, y + 20), (255, 255, 255), 2)
        cv2.putText(frame[0], str(display),(x+30,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(windowName,np.concatenate((frame[0],frame[1]),1))

        key = cv2.waitKey(100)
        print(disp_string + ' = ' + str(np.round(display,2)))

#############################################################################




#########################################################################
## ACTIVATES TRACK BAR FOR STEREO ADJUSTMENTS

def create_tackbar(windowNameD,num_disp,min_disp,wsize,lambd,sigma,width,height,left_matcher,wls_filter):


    def on_trackbar_set_min_disparities(value):
        left_matcher.setMinDisparity(max(1, value * 1))


    def on_trackbar_set_disparities(value):
        left_matcher.setNumDisparities(max(16, value * 16))

    def on_trackbar_set_blocksize(value):
        if not(value % 2):
            value = value + 1
        left_matcher.setBlockSize(max(3, value))

    def on_trackbar_set_speckle_range(value):
        left_matcher.setSpeckleRange(value)

    def on_trackbar_set_speckle_window(value):
        left_matcher.setSpeckleWindowSize(value)

    def on_trackbar_set_setDisp12MaxDiff(value):
        left_matcher.setDisp12MaxDiff(value)

    def on_trackbar_set_setP1(value):
        left_matcher.setP1(value)

    def on_trackbar_set_setP2(value):
        left_matcher.setP2(value)

    def on_trackbar_set_setPreFilterCap(value):
        left_matcher.setPreFilterCap(value)

    def on_trackbar_set_setUniquenessRatio(value):
        left_matcher.setUniquenessRatio(value)

    def on_trackbar_set_wlsLmbda(value):
        wls_filter.setLambda(value)

    def on_trackbar_set_wlsSigmaColor(value):
        wls_filter.setSigmaColor(value * 0.1)

    def on_trackbar_null(value):
        return


    cv2.createTrackbar("Min Disparity(x 16): ", windowNameD, 0, 16, on_trackbar_set_min_disparities)
    cv2.createTrackbar("Max Disparity(x 16): ", windowNameD, int(num_disp/16), 16, on_trackbar_set_disparities)
    cv2.createTrackbar("Window Size: ", windowNameD, wsize, 50, on_trackbar_set_blocksize)
    cv2.createTrackbar("Speckle Window: ", windowNameD, 0, 200, on_trackbar_set_speckle_window)
    cv2.createTrackbar("LR Disparity Check Diff:", windowNameD, 0, 25, on_trackbar_set_setDisp12MaxDiff)
    cv2.createTrackbar("Disparity Smoothness P1: ", windowNameD, 0, 4000, on_trackbar_set_setP1)
    cv2.createTrackbar("Disparity Smoothness P2: ", windowNameD, 0, 16000, on_trackbar_set_setP2)
    cv2.createTrackbar("Pre-filter Sobel-x- cap: ", windowNameD, 0, 5, on_trackbar_set_setPreFilterCap)
    cv2.createTrackbar("Winning Match Cost Margin %: ", windowNameD, 0, 20, on_trackbar_set_setUniquenessRatio)
    cv2.createTrackbar("Speckle Size: ", windowNameD, math.floor((width/2 * height) * 0.0005), 10000, on_trackbar_null)  #* 0.0005), 10000,
    cv2.createTrackbar("Max Speckle Diff: ", windowNameD, 16, 2048, on_trackbar_null)
    cv2.createTrackbar("WLS Filter Lambda: ", windowNameD, lambd, 10000, on_trackbar_set_wlsLmbda)
    cv2.createTrackbar("WLS Filter Sigma Color (x 0.1): ", windowNameD, math.ceil(sigma / 0.1), 50, on_trackbar_set_wlsSigmaColor) #0.1
################################################################################










#####################################################################
opencv_cam_settings = cv2.CAP_PROP_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
################################################################################





################################################################################
# DISPLAY IMAGE AND CALL IMAGE ADJUST FUNCTION

def opencv_adjust(cap,windowName):

    print_help()
    key = 1000
    while key != 113:  # for 'q' key

        if len(cap)==1:
            ret,frame = cap[0].read()

        else:
            ret,left_frame = cap[0].read()
            ret,right_frame = cap[1].read()
            frame=np.concatenate((left_frame,right_frame),1)

        cv2.imshow(windowName,frame )
        key = cv2.waitKey(1)
        opencv_camera_settings(key, cap)

################################################################################




################################################################################
def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Quit:                               q\n")
################################################################################



################################################################################
# SET CAMERA SETTINGS (BRIGHTNESS, CONTRAST ETC.) FOR LEFT AND RIGHT EQUALY
def opencv_camera_settings(key, cap):
    step_camera_settings = 1

    if key == 115:  # for 's' key
        switch_opencv_camera_settings()
    elif key == 43:  # for '+' key
        current_value = cap[0].get(opencv_cam_settings)
        cap[0].set(opencv_cam_settings, current_value + step_camera_settings)
        if len(cap)==2:
            cap[1].set(opencv_cam_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cap[0].get(opencv_cam_settings)
        if current_value >= 1:
            cap[0].set(opencv_cam_settings, current_value - step_camera_settings)
            if len(cap)==2:
                cap[1].set(opencv_cam_settings, current_value + step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        cap[0].set(cv2.CAP_PROP_BRIGHTNESS, -1)
        cap[0].set(cv2.CAP_PROP_CONTRAST, -1)
        cap[0].set(cv2.CAP_PROP_HUE, -1)
        cap[0].set(cv2.CAP_PROP_SATURATION, -1)
        cap[0].set(cv2.CAP_PROP_SHARPNESS, -1)
        cap[0].set(cv2.CAP_PROP_GAIN, -1)
        cap[0].set(cv2.CAP_PROP_EXPOSURE, -1)
        cap[0].set(cv2.CAP_PROP_WB_TEMPERATURE, -1)

        if len(cap)==2:
            cap[1].set(cv2.CAP_PROP_BRIGHTNESS, -1)
            cap[1].set(cv2.CAP_PROP_CONTRAST, -1)
            cap[1].set(cv2.CAP_PROP_HUE, -1)
            cap[1].set(cv2.CAP_PROP_SATURATION, -1)
            cap[1].set(cv2.CAP_PROP_SHARPNESS, -1)
            cap[1].set(cv2.CAP_PROP_GAIN, -1)
            cap[1].set(cv2.CAP_PROP_EXPOSURE, -1)
            cap[1].set(cv2.CAP_PROP_WB_TEMPERATURE, -1)


        print("Camera settings: reset")
################################################################################




################################################################################
# SWITCH BETWEEN SETTINGS
def switch_opencv_camera_settings():
    global opencv_cam_settings
    global str_camera_settings
    if opencv_cam_settings == cv2.CAP_PROP_BRIGHTNESS:
        opencv_cam_settings = cv2.CAP_PROP_CONTRAST
        str_camera_settings = "Contrast"
        print("Camera settings: CONTRAST")
    elif opencv_cam_settings == cv2.CAP_PROP_CONTRAST:
        opencv_cam_settings = cv2.CAP_PROP_HUE
        str_camera_settings = "Hue"
        print("Camera settings: HUE")
    elif opencv_cam_settings == cv2.CAP_PROP_HUE:
        opencv_cam_settings = cv2.CAP_PROP_SATURATION
        str_camera_settings = "Saturation"
        print("Camera settings: SATURATION")
    elif opencv_cam_settings == cv2.CAP_PROP_SATURATION:
        opencv_cam_settings = cv2.CAP_PROP_SHARPNESS
        str_camera_settings = "Sharpness"
        print("Camera settings: Sharpness")
    elif opencv_cam_settings == cv2.CAP_PROP_SHARPNESS:
        opencv_cam_settings = cv2.CAP_PROP_GAIN
        str_camera_settings = "Gain"
        print("Camera settings: GAIN")
    elif opencv_cam_settings == cv2.CAP_PROP_GAIN:
        opencv_cam_settings = cv2.CAP_PROP_EXPOSURE
        str_camera_settings = "Exposure"
        print("Camera settings: EXPOSURE")
    elif opencv_cam_settings == cv2.CAP_PROP_EXPOSURE:
        opencv_cam_settings = cv2.CAP_PROP_WB_TEMPERATURE
        str_camera_settings = "White Balance"
        print("Camera settings: WHITEBALANCE")
    elif opencv_cam_settings == cv2.CAP_PROP_WB_TEMPERATURE:
        opencv_cam_settings = cv2.CAP_PROP_BRIGHTNESS
        str_camera_settings = "Brightness"
        print("Camera settings: BRIGHTNESS")
################################################################################
