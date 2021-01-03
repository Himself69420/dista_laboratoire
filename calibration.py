from StereoCalibration import StereoCalibration

# ================ PARAMETRES ========================

patternSize=(9,6)
squaresize=3.64e-2
# single_path='stereo/'
stereo_path='stereo/'
# single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'

# ====================================================

obj = StereoCalibration(patternSize, squaresize)

obj.calibrateStereo(stereo_path, stereo_detected_path,fisheye=False, calib_2_sets=False)
obj.saveResultsXML()
obj.reprojection('output/reprojection/')
