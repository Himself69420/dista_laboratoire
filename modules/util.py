"""
Petites fonctions utiles
Auteur: Marinouille
"""

import os
import cv2 as cv
import numpy as np


def coins_damier(patternSize,squaresize):
    objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
    #
    objpy = np.flip(objp[:,1])
    objp[:,1]=objpy
    #
    # objpx = np.flip(objp[:,0])
    # objp[:,0]=objpx
    #
    objp*=squaresize
    return objp

def find_corners(fname,patternSize):
    color = cv.imread(fname)
    # Transformation de l'image en nuance de gris pour analyse
    gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    # On cherche les coins sur le damier
    ret, corners = cv.findChessboardCorners(gray, patternSize, None)
    return ret, corners, color


def clean_folders(output_paths):
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for file in os.scandir(path):
                if file.name.endswith(".jpg"):
                    os.unlink(file.path)

def draw_reprojection(color, objPoints, imgPoints, cameraMatrix, distCoeffs, patternSize, squaresize, folder, i):
    """ Pour une image, reprojeter des points et les axes"""
    # Vérification de la calibration de la caméra en reprojection:
    # Montrer axes
    ret, rvecs, tvecs = cv.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    img	=cv.drawFrameAxes(color.copy(), cameraMatrix, distCoeffs, rvecs, tvecs, 3*squaresize, 5)
    cv.imwrite('{}reprojection_axes_{}.jpg'.format(folder, i), img) #Z est en bleu, Y en vert, X en rouge
    pts, jac = cv.projectPoints(objPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
    img = cv.drawChessboardCorners(color.copy(), patternSize, pts, 1)
    cv.imwrite('{}reprojection_points_{}.jpg'.format(folder, i), img)
    return img

def readXML(fname):
    s = cv.FileStorage()
    s.open(fname, cv.FileStorage_READ)
    K=s.getNode('K').mat()
    R=s.getNode('R').mat()
    t=s.getNode('t').mat()
    D=s.getNode('coeffs').mat()
    imageSize=s.getNode('imageSize').mat()
    imageSize=(int(imageSize[0][0]),int(imageSize[1][0]))
    E=s.getNode('E').mat()
    F=s.getNode('F').mat()
    s.release()
    return K,D,R,t,imageSize, E, F
