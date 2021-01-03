"""
Petites fonctions utiles
Auteur: Marinouille
"""

import os, cv2
import numpy as np


def coins_damier(patternSize,squaresize):
    objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
    objp*=squaresize
    return objp

def centres_damier(patternSize,squaresize):
    objp = np.zeros(((patternSize[0]-1)*(patternSize[1]-1), 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0]-1, 0:patternSize[1]-1].T.reshape(-1, 2)
    objp*=squaresize
    objp+=np.array([0.5,0.5,0])*squaresize
    return objp

def read_images(image):
    color=cv2.imread(image)
    gray=cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray

# draw the provided lines on the image
def drawlines(img,lines,pts):
    ''' img - image on which we draw the epilines for the points in other image
    '''
    _,c,_ = img.shape
    img_=img.copy()
    for r,pt in zip(lines,pts):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img_ = cv.line(img_, (x0,y0), (x1,y1), color,1)
        img_ = cv.circle(img_,tuple(pt[0]),5,color,-1)
    return img_



def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def combine(left, right):
    im = np.concatenate((left,right), axis=1)
    return im


def find_corners(fname,patternSize):
    color = cv2.imread(fname)
    # Transformation de l'image en nuance de gris pour analyse
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # On cherche les coins sur le damier
    ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
    return ret, corners, color

def refine_corners(patternSize, objpoints, imgpoints, objp, corners, color, criteria, detected_path, view, i, p=False):
    # Quand les coins sont trouvés et rafinés on les ajoute au tableau de point 3D
    if objpoints != None:
        objpoints.append(objp)
    # On ajoute les points dans le tableau 2D
    corners2= cv2.cornerSubPix(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), corners, (11, 11),(-1, -1), criteria)
    imgpoints.append(corners2)
    # On dessine et affiche les coins sur l'image
    if p:
        _ = cv2.drawChessboardCorners(color, patternSize, corners2, True)
        fname='{}{:03d}.jpg'.format(view, i)
        cv2.imwrite(detected_path + fname, color)


def clean_folders(output_paths):
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for file in os.scandir(path):
                if file.name.endswith(".jpg"):
                    os.unlink(file.path)

def draw_reprojection(color, objectPoints, imagePoints, cameraMatrix, distCoeffs, patternSize, squaresize, folder, i):
    """ Pour une image, reprojeter des points et les axes"""
    def draw(img, origin, imgpts):
        # BGR
        img = cv2.line(img, tuple(origin[0].ravel()), tuple(imgpts[0].ravel()), (255,0,0), 5) #X
        img = cv2.line(img, tuple(origin[0].ravel()), tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, tuple(origin[0].ravel()), tuple(imgpts[2].ravel()), (0,255,255), 5) # Z en jaune
        return img

    objPoints=objectPoints
    imgPoints=imagePoints
    # Vérification de la calibration de la caméra en reprojection:
    # Montrer axes
    axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)*3*squaresize
    ret, rvecs, tvecs = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    axisProj, jac = cv2.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeffs)
    origin = np.float32([[0,0,0]]).reshape(-1,1)
    originProj , jac = cv2.projectPoints(origin, rvecs, tvecs, cameraMatrix, distCoeffs)
    img = draw(color.copy(), originProj[0], axisProj)
    cv2.imwrite('{}reprojection_axes_{}.jpg'.format(folder, i), img)
    pts, jac = cv2.projectPoints(objPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
    img = cv2.drawChessboardCorners(color.copy(), patternSize, pts, 1)
    cv2.imwrite('{}reprojection_points_{}.jpg'.format(folder, i), img)
    return img


def readXML(fname):
    s = cv2.FileStorage()
    s.open(fname, cv2.FileStorage_READ)
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
