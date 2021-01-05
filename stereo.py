################################################################################
import cv2
import numpy as np
import time
import nanocamera as nano

from modules.stereo_tools import *
#########################################################

################################################################################
## PATH TO CALIBRATION FILE : SEE EXAMPLE FOR FORMAT
left_xml='cam1.xml'
right_xml='cam2.xml'
#####################################################################


##############################################################################
## GET ARGUMENT VALUES
args = parse_args()
cam_ids=args.cam_ids
cam_ids = args.cam_ids.split(',')
resolution=args.resolution
adjust=args.adjust
depthmap=args.depthmap
distance=args.distance
###############################################################################


#############################################################
## DEFINE RESOLUTION AND GET IMAGE SIZE
width,height = get_image_dimension_from_resolution(resolution)
class Resolution :
    width = int( width)
    height = int( height)
image_size = Resolution()
#####################################################################



##############################################################################
# INITIALIZE CAMERAS
if len(cam_ids)==1:

    ## for single USB (left and rigth frames are "glued together )
    cap = cv2.VideoCapture(int(cam_ids[0]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
else:
    ### for double USB
    # cap_left = cv2.VideoCapture(int(cam_ids[0]))
    # cap_right = cv2.VideoCapture(int(cam_ids[1]))
    # cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # nano.Camera()
    cap_left = nano.Camera(device_id=0,flip=2,width=800,height=480,fps=30)
    cap_right = nano.Camera(device_id=1,flip=2,width=800,height=480,fps=30)

########################################################################



#############################################################################
## GET ALL CAMERA MATRICES AND COEFFICIENTS
camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y, Q = init_calibration(left_xml, right_xml,image_size, resolution)
################################################################################



##############################################################################
# INITIALIZE IMAGE WINDOWS
windowName = "Live Camera Input"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowName, width, height)
################################################################################



################################################################################
# IF WANT TO ADJUST IMAGE QUALITY : FOLLOW INSTRUCTION ON TERMINAL RIGHT CAMERA
# WILL TAKE SAME VALUES AS LEFT CAMERA
# MAKE SURE TO SELECT THE WINDOW WITH THE MOUSE
if adjust:
    if len(cam_ids)==2:
        opencv_adjust([cap_left,cap_right],windowName)
    else:
        opencv_adjust([cap],windowName)
################################################################################




##############################################################################
# INITIALIZE DISPARITY WINDOWS
windowNameD = "Stereo Disparity"
cv2.namedWindow(windowNameD, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowNameD, width, 2*height)
cv2.moveWindow(windowNameD,1000,0)  # Move it to (40,30)
################################################################################



################################################################################
# SETUP DEFAULT PARMATERS FOR DISPARITY COMPUTATION
num_disp = 80
min_disp=1
wsize = 5

stereoProcessor = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities = num_disp,
        blockSize=wsize,
        P1=24*wsize*wsize,
        P2=96*wsize*wsize,  #96
#       disp12MaxDiff=1,
        preFilterCap=63,    #63
#        disp12MaxDiff=1,
#        uniquenessRatio=15,
#        speckleWindowSize=0,
#        speckleRange=2,
#        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

left_matcher = stereoProcessor
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)


lambd = 8000
sigma = 5
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lambd)
wls_filter.setSigmaColor(sigma)
################################################################################



###############################################################################
## BEGIN COMPUTATIONS
create_tackbar(windowNameD,num_disp,min_disp,wsize,lambd,sigma,width,height,left_matcher,wls_filter)
keep_processing = True
startime = time.time()
ff=0

while (keep_processing):


#   GET FRAME
    if len(cam_ids)==1:
        ret,frame=cap.read()
        frames = np.split(frame, 2, axis=1)
        frameL=frames[0]
        frameR=frames[1]
    else:
        frameL=cap_left.read()
        frameR=cap_right.read()


#    RECTIFIY
    grayL,grayR=\
    get_rectified_left_right(frameL,frameR, map_left_x, map_left_y, map_right_x, map_right_y)

#   STORE FOR DISPLAY LATER
    frame=[grayL,grayR]


    # GRAY SCALE
    grayL = cv2.cvtColor(grayL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(grayR,cv2.COLOR_BGR2GRAY)


##    DOWNSCALE
    downscale=2
    new_num_disp = int(num_disp / downscale)
    n_width = int(grayL.shape[1] * 1/downscale)
    n_height = int(grayR.shape[0] * 1/downscale)
    grayL_down = cv2.resize(grayL, (n_width, n_height))
    grayR_dowm = cv2.resize(grayR,(n_width, n_height))


#   SMOOTH
    grayL_down = cv2.medianBlur(grayL_down,3)
    grayR_dowm = cv2.medianBlur(grayR_dowm,3)


    # COMPUTE AND FILTER DISPARITY
    displ = left_matcher.compute(cv2.UMat(grayL_down),cv2.UMat(grayR_dowm))
    dispr = right_matcher.compute(cv2.UMat(grayR_dowm),cv2.UMat(grayL_down))
    displ = np.int16(cv2.UMat.get(displ))
    dispr = np.int16(cv2.UMat.get(dispr))
    disparity = wls_filter.filter(displ, grayL, None, dispr)

    # FILTER SPECKLES DONE SEPARALTY AS POST PROCESSING FROM VALUE ON TRACK BAR
    speckleSize = cv2.getTrackbarPos("Speckle Size: ", windowNameD)
    maxSpeckleDiff = cv2.getTrackbarPos("Max Speckle Diff: ", windowNameD)
    cv2.filterSpeckles(disparity, 0, speckleSize, maxSpeckleDiff)

    # FORMAT DISPARITY
    disparity =disparity.astype(np.float32) / 16.0

    # COMPUTE X,Y Z COORDINATES
    # DEFINE POINT CLOUD AND DEPTH MAP (Z COORDINATE)
    cloud = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    depth_map=cv2.UMat(cloud[:,:,2])

    if depthmap:
        displaymap=depth_map
    else:
        minDisp = np.min(disparity)
        maxDisp = np.max(disparity)
        disparity_to_display = ((disparity - minDisp) / (maxDisp - minDisp) * 255.).astype(np.uint8)
        # disparity_to_display = (disparity * (256. / num_disp)).astype(np.uint8)
        displaymap=disparity_to_display



    # DRAW TARGET ON IMAGE AND DEPTH MAP OR DISPARITY MAP
    cv2.line(displaymap, (int(width / 2) - 20, int(height / 2)), (int(width / 2) + 20, int(height / 2)), (255, 255, 255), 2)
    cv2.line(displaymap, (int(width / 2), int(height / 2) - 20), (int(width / 2), int(height / 2) + 20), (255, 255, 255), 2)
    cv2.line(frame[0], (int(width / 2) - 20, int(height / 2)), (int(width / 2) + 20, int(height / 2)), (255, 255, 255), 2)
    cv2.line(frame[0], (int(width / 2), int(height / 2) - 20), (int(width / 2), int(height / 2) + 20), (255, 255, 255), 2)


    # DISPLAY DISTANCE OF DEPTH ON IMAGE
    center = cloud[int(height / 2), int(width / 2),:]  # center pixel x,y,z values
    if distance:
        display= np.round(np.linalg.norm(center),2)  # DISTANCE
    else:
        display=center[2]  # DEPTH
    text = str(np.round(display,2))
    cv2.putText(frame[0], text,(230,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)


    # SHOW DEPTH MAP
    cv2.imshow(windowNameD, displaymap)


    # SHOW FRAMES
    cv2.imshow(windowName,np.concatenate((frame[0],frame[1]),1))

    key = cv2.waitKey(1)

    ff=ff+1
    elapsed = time.time() - startime
    fps = int(ff/elapsed)
    print("Frame rate : " + str(fps) + " FPS"+ "\n")

    cv2.setMouseCallback(windowName,on_mouse_display_depth_value, (distance,cloud,frame,windowName))

    if key == 27 :
        break
################################################################################

cap.release()
