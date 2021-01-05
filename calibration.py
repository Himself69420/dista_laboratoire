from modules.StereoCalibration import StereoCalibration
from modules.util import *

# ================ PARAMETRES ========================

patternSize=(10,8)
squaresize=2e-2
# single_path='stereo/'
stereo_path='captures/captures_calibration/'
single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'
# ====================================================

obj = StereoCalibration(patternSize, squaresize)
obj.calibrateStereo(stereo_path, stereo_detected_path, single_detected_path, fisheye=False, calib_2_sets=False)
obj.saveResultsXML()
# obj.reprojection('output/reprojection/')
