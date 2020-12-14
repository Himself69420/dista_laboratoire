from StereoCalibration import StereoCalibration

# ================ PARAMETRES ========================

patternSize=(9,6)
squaresize=3.64e-2
single_path='images/'
stereo_path='images/'
single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'

# ====================================================

obj = StereoCalibration(patternSize, squaresize, single_path, stereo_path, single_detected_path, stereo_detected_path)

obj.calibrateStereo(fisheye=False, calib_2_sets=False)
obj.saveResultsXML()
obj.saveConf('conf')
