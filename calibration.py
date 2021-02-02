from modules.StereoCalibration import StereoCalibration

# ================ PARAMETRES ========================
patternSize=(9,6)
squaresize=3.62e-2
images_path='captures/captures_calibration/'
single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'
# ====================================================

obj = StereoCalibration(patternSize, squaresize)
obj.calibrate(images_path, stereo_detected_path, single_detected_path)
obj.saveResultsXML()
