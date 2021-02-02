"""
Auteur: Marianne Lado-Roy
Fichier exécutable pour effectuer la calibration avec la classe StereoCalibration
"""
import argparse

from modules.StereoCalibration import StereoCalibration

# ================ FOLDERS ==========================
images_path='captures/'
single_detected_path='output/singles_detected/'
stereo_detected_path='output/stereo_detected/'
# ====================================================

# Parser -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# arguments obligatoires
parser.add_argument("points_per_row", type=int,
                    help="Nombre de coins interne du damier sur une ligne")
parser.add_argument("points_per_col", type=int,
                    help="Nombre de coins interne du damier sur une colone")
parser.add_argument("squareSize", type=float,
                    help="squareSize: taille d'un carrés du damier en mètre")
# arguments optionnels
parser.add_argument("--images", type=str, help="images: path au dossier contenant les images à analyser")

args = parser.parse_args()

patternSize=(args.points_per_row, args.points_per_col)
squareSize=args.squareSize
images=args.images

if images is not None:
    images_path=images
# ------------------------------------------------------------------------------

# Faire la calibration ---------------------------------------------------------
obj = StereoCalibration(patternSize, squareSize)
obj.calibrate(images_path, stereo_detected_path, single_detected_path)
obj.saveResultsXML()
# ------------------------------------------------------------------------------
