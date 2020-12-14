import numpy as np
import cv2
import glob, os

from util import find_corners, refine_corners, draw_reprojection, outputClean


# Classe contenant les fonctions de calibration
class StereoCalibration():

    """
    patternSize = (nb points per row, nb points per col)
    squaresize : taille damier en mètres
    """
    # Paramètres D'initialisation
    def __init__(self, patternSize, squaresize, single_path, stereo_path, single_detected_path, stereo_detected_path):
        # Critères de terminaison
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        self.patternSize=patternSize
        self.squaresize=squaresize #utiliser des unités SI svp
        # Déclaration des points 3D et taille du damier 7x7
        self.objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
        # Taille des carreaux sur le damier en mètres
        self.objp*=self.squaresize

        # Folders contenant les images à analyser
        self.single_path=single_path
        self.stereo_path=stereo_path

        # Folders ou enregistrer les images détectées
        self.single_detected_path=single_detected_path
        self.stereo_detected_path=stereo_detected_path

        # Créer/Vider les folders ou enregistrer les coins détectés
        outputClean([single_detected_path,stereo_detected_path])

        # Déclaration des variables
        self.imageSize1, self.imageSize2 = None, None
        self.err1, self.M1, self.d1, self.r1, self.t1 = None, None, None, None, None
        self.err2, self.M2, self.d2, self.r2, self.t2 = None, None, None, None, None
        self.errStereo, self.R, self.T = None, None, None,

        # Flags:
        self.not_fisheye_flags = cv2.CALIB_FIX_K3|cv2.CALIB_ZERO_TANGENT_DIST
        self.fisheye_flags=cv2.CALIB_RATIONAL_MODEL|cv2.CALIB_FIX_K5|cv2.CALIB_FIX_K6|cv2.CALIB_ZERO_TANGENT_DIST



    def read_single(self, view):
        """
        single_path  (str): './<folder>/'
        view  (str): 'left' ou 'right'
        """
        images = np.sort(glob.glob(self.single_path + '{}*.jpg'.format(view)))
        assert len(images) != 0, "Images pas trouvées. Vérifier le path"

        # Parcours le dossier pour trouver le damier sur les images
        objpoints=[]; imgpoints=[]
        for i in range(len(images)):
            ret, corners, color = find_corners(images[i], self.patternSize)
            if ret==True:
                refine_corners(self.patternSize, objpoints, imgpoints, self.objp, corners, color, self.criteria, self.single_detected_path, view, i )
            image_shape=color.shape
        return objpoints, imgpoints, image_shape[:2]


    def read_stereo(self):
        images_right = np.sort(glob.glob(self.stereo_path + 'left*.jpg'))
        images_left = np.sort(glob.glob(self.stereo_path + 'right*.jpg'))
        assert (len(images_right) != 0 or len(images_left) != 0  ), "Images pas trouvées. Vérifier le path"

        # Parcours le dossier pour trouver le damier sur les images
        objpoints=[]; imgpoints_l=[]; imgpoints_r=[]
        for i in range(len(images_right)):
            ret_l, corners_l, gray_l = find_corners(images_left[i], self.patternSize)
            ret_r, corners_r, gray_r = find_corners(images_right[i], self.patternSize)

            # Si le damier dans les images gauche et droite correspondante est détecté
            if ret_l*ret_r==1:
                # Quand les coins sont trouver et rafiné on les ajoutes au tableau de point 3D
                objpoints.append(self.objp)

                refine_corners(self.patternSize, None, imgpoints_l, self.objp, corners_l, gray_l, self.criteria, self.stereo_detected_path, 'left',i )

                refine_corners(self.patternSize, None, imgpoints_r, self.objp, corners_r, gray_r, self.criteria, self.stereo_detected_path, 'right',i )

        return objpoints, imgpoints_l, imgpoints_r

    def calibrateSingle(self, fisheye=False, skipdetection=False):
        """On calibre les deux caméras indépendament et sauvegarde les valeurs de calibration
        """
        if not skipdetection:
            self.objpoints_l, self.imgpoints_l, self.imageSize1 = self.read_single('left')
            self.objpoints_r, self.imgpoints_r, self.imageSize2 = self.read_single('right')

        if fisheye==True:
            self.err1, self.M1, self.d1, self.r1, self.t1, stdDeviationsIntrinsics1, stdDeviationsExtrinsics1, self.perViewErrors1 = cv2.calibrateCameraExtended(self.objpoints_l, self.imgpoints_l, self.imageSize1, None, None, flags=self.fisheye_flags)

            self.err2, self.M2, self.d2, self.r2, self.t2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, self.perViewErrors2 = cv2.calibrateCameraExtended(self.objpoints_r, self.imgpoints_r, self.imageSize2, None, None, flags=self.fisheye_flags)

        else:
            self.err1, self.M1, self.d1, self.r1, self.t1, stdDeviationsIntrinsics1, stdDeviationsExtrinsics1, self.perViewErrors1 = cv2.calibrateCameraExtended(self.objpoints_l, self.imgpoints_l, self.imageSize1, None, None, flags=self.not_fisheye_flags)

            self.err2, self.M2, self.d2, self.r2, self.t2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, self.perViewErrors2 = cv2.calibrateCameraExtended(self.objpoints_r, self.imgpoints_r, self.imageSize2, None, None, flags=self.not_fisheye_flags)

        # Print erreur de reprojection
        print('Erreur de reprojection RMS calibration individuelle')
        print(self.err1, self.err2)


    def calibrateStereo(self, fisheye=False, calib_2_sets=False):
        """
        calib_2_sets (Bool)
        True: pour utiliser des sets de photos différents
            - un pour la calibration individuelle des caméras
            - un pour la calibration stéréo
        False: pour utiliser un seul set de photo pour la calibration individuelle et la calibration stéréo
        """

        # faire calibration individuelle avant
        if self.err1==None or self.err2==None:
            self.calibrateSingle(fisheye)

        # deux sets ou un set
        if calib_2_sets:
            flags = cv2.CALIB_FIX_INTRINSIC
            objpoints, imgpoints_l, imgpoints_r = self.read_stereo()
        else:
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            if len(self.imgpoints_l) == len(self.imgpoints_r):
                # Si le même nombre d'images sont détectées à gauche et à droite, on assume que ce sont les même et ça évite de re-détecter des points (à surveiller quand même)
                objpoints, imgpoints_l, imgpoints_r = self.objpoints_l, self.imgpoints_l, self.imgpoints_r
            else:
                objpoints, imgpoints_l, imgpoints_r = self.read_stereo()

        # caméra fisheye
        if fisheye:
            flags += self.fisheye_flags
        else:
            flags += self.not_fisheye_flags

        # calculs
        self.errStereo, _, _, _, _, self.R, self.T, self.E, self.F= cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, self.M1, self.d1, self.M2,self.d2, self.imageSize1 ,criteria=self.criteria_cal, flags=flags)

        # Print erreur de reprojection
        print('Erreur de reprojection RMS calibration stereo')
        print(self.errStereo)


    def saveResultsXML(self):

        # Enregistrer caméra 1:
        s = cv2.FileStorage()
        s.open('cam1.xml', cv2.FileStorage_WRITE)
        s.write('K', self.M1)
        s.write('R', np.eye(3))
        s.write('t', np.zeros((1,3)))
        s.write('coeffs', self.d1)
        s.write('imageSize', self.imageSize1)
        s.write('E', self.E)
        s.write('F', self.F)
        s.release()

        # Enregistrer caméra 2:
        s = cv2.FileStorage()
        s.open('cam2.xml', cv2.FileStorage_WRITE)
        s.write('K', self.M2)
        s.write('R', self.R)
        s.write('t', self.T)
        s.write('coeffs', self.d2)
        s.write('imageSize', self.imageSize2)
        s.write('E', self.E)
        s.write('F', self.F)
        s.release()

    def saveTXT(self, fname):
        """ enregistrer en format .txt """

        # Sauvegarde des données dans un fichier
        f = open("{}.conf".format(fname), "w+")
        f.write("[LEFT_CAM_HD] \n")
        f.write("K1 = {}\n".format(self.M1))
        f.write("d1 = {}\n".format(self.d1))
        f.write("\n")
        f.write("[RIGHT_CAM_HD] \n")
        f.write("K2 = {}\n".format(self.M2))
        f.write("d2 = {}\n".format(self.d2))
        f.write("[STEREO] \n")
        f.write("R = {}\n".format(self.R))
        f.write("T = {}\n".format(self.T))
        f.write("Baseline = {} mm".format(self.T[0][0]*1e3)) #mm
        f.write("\n\n")
        f.write("ERREURS\n")
        f.write('Erreur cam 1: {}\n'.format(self.err1))
        f.write('Erreur cam 2: {}\n'.format(self.err2))
        f.write('Erreur stéréo: {}\n'.format(self.errStereo))
        f.close()

    def saveConf(self, fname):

        # On extrait les valeurs qui nous intéresse dans la matrice gauche
        fx_l = self.M1[0][0]
        cx_l = self.M1[0][2]
        fy_l = self.M1[1][1]
        cy_l = self.M1[1][2]
        k1_l = self.d1[0][0]
        k2_l = self.d1[0][1]
        k3_l = self.d1[0][4]
        p1_l = self.d1[0][2]
        p2_l = self.d1[0][3]

        # On extrait les valeurs qui nous intéresse dans la matrice droite
        fx_r = self.M2[0][0]
        cx_r = self.M2[0][2]
        fy_r = self.M2[1][1]
        cy_r = self.M2[1][2]
        k1_r = self.d2[0][0]
        k2_r = self.d2[0][1]
        k3_r = self.d2[0][4]
        p1_r = self.d2[0][2]
        p2_r = self.d2[0][3]

        # On extrait la distance entre les deux lentilles
        baseline = self.T[0][0]*1e3 #mm
        #baseline = baseline * squaresize * 10

        # Sauvegarde des données dans un fichier
        f = open("{}.conf".format(fname), "w+")
        f.write("[LEFT_CAM_VGA] \n")
        f.write("fx=" + str(fx_l) + "\n")
        f.write("fy=" + str(fy_l) + "\n")
        f.write("cx=" + str(cx_l) + "\n")
        f.write("cy=" + str(cy_l) + "\n")
        f.write("k1=" + str(k1_l) + "\n")
        f.write("k2=" + str(k2_l) + "\n")
        f.write("k3=" + str(k3_l) + "\n")
        f.write("p1=" + str(p1_l) + "\n")
        f.write("p2=" + str(p2_l) + "\n")
        f.write("\n")
        f.write("[RIGHT_CAM_VGA] \n")
        f.write("fx=" + str(fx_r) + "\n")
        f.write("fy=" + str(fy_r) + "\n")
        f.write("cx=" + str(cx_r) + "\n")
        f.write("cy=" + str(cy_r) + "\n")
        f.write("k1=" + str(k1_r) + "\n")
        f.write("k2=" + str(k2_r) + "\n")
        f.write("k3=" + str(k3_r) + "\n")
        f.write("p1=" + str(p1_r) + "\n")
        f.write("p2=" + str(p2_r) + "\n")
        f.write("\n")
        f.write("[STEREO] \n")
        f.write("Baseline=" + str(baseline) + "\n")
        f.write("\n")
        f.close()

    def reprojection(self, folder):
        """ Dessiner la reprojection """
        # outputClean([folder])

        images_left = np.sort(glob.glob(self.single_path + 'left*.jpg'))
        images_right = np.sort(glob.glob(self.single_path + 'right*.jpg'))
        assert (len(images_right) != 0 or len(images_left) != 0  ), "Images pas trouvées. Vérifier le path"

        for i in range(len(self.imgpoints_l)):
            draw_reprojection(cv2.imread(images_left[i]), self.objpoints_l[i], self.imgpoints_l[i], self.M1, self.d1, self.patternSize, self.squaresize, folder, i)

        # for j in range(len(self.imgpoints_r)):
        #     draw_reprojection(cv2.imread(images_right[j]), self.objpoints_r[j], self.imgpoints_r[j], self.M2, self.d2, self.patternSize, self.squaresize, folder, j)
