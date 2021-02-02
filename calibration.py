"""
Auteur: Marianne Lado-Roy
Fichier exécutable pour effectuer la calibration avec la classe StereoCalibration
"""

from modules.StereoCalibration import StereoCalibration

# ================ PARAMETRES ========================
patternSize=(9,6) # Motif du damier
squaresize=3.62e-2 # Taille des carrés du damier
images_path='captures/captures_calibration/'
single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'
# ====================================================

obj = StereoCalibration(patternSize, squaresize)
obj.calibrate(images_path, stereo_detected_path, single_detected_path)
obj.saveResultsXML()
