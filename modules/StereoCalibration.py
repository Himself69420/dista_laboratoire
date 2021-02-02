import numpy as np
import cv2 as cv
import glob, os

from modules.util import draw_reprojection, clean_folders, coins_damier, find_corners


# Classe contenant les fonctions de calibration
class StereoCalibration():

    def __init__(self, patternSize, squaresize):
        """
        || Constructeur ||
        patternSize = (nb points per row, nb points per col)
        squaresize : taille damier en mètres
        """

        # Damier ---------------------------------------------------------------
        self.patternSize=patternSize
        self.squaresize=squaresize
        self.objp = coins_damier(patternSize,squaresize)
        # ----------------------------------------------------------------------

        # Critères -------------------------------------------------------------
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        # ----------------------------------------------------------------------

        # Flags de calibration -------------------------------------------------
        self.flags_nano = cv.CALIB_FIX_K3|cv.CALIB_ZERO_TANGENT_DIST
        self.color_flag=cv.COLOR_RGB2GRAY
        self.winSize=(5,5)
        # ----------------------------------------------------------------------


        # Déclaration des attributs --------------------------------------------
        # Folders contenant les images à analyser
        self.images_path=None

        # Folders ou enregistrer les images détectées
        self.single_detected_path=None
        self.stereo_detected_path=None

        # Variables pour la calibration à garder en mémoire
        self.imageSize1, self.imageSize2 = None, None
        self.err1, self.M1, self.d1, self.r1, self.t1 = None, None, None, None, None
        self.err2, self.M2, self.d2, self.r2, self.t2 = None, None, None, None, None
        self.errStereo, self.R, self.T = None, None, None
        self.stdDeviationsIntrinsics1, self.stdDeviationsExtrinsics1 = None, None
        self.stdDeviationsIntrinsics2, self.stdDeviationsExtrinsics2 = None, None

        # Pour que dessine les coins détectés
        self.draw=True
        # ----------------------------------------------------------------------

    def __read_images(self, images_path):
        """
        ||Private method||
        """
        # Lire les images ------------------------------------------------------
        images_left = np.sort(glob.glob(images_path + 'left*.jpg'))
        images_right = np.sort(glob.glob(images_path + 'right*.jpg'))
        assert (len(images_right) != 0 or len(images_left) != 0  ), "Images pas trouvées. Vérifier le path"
        # ----------------------------------------------------------------------

        # Déclaration des listes de points par image ---------------------------
        # Points de la calibration individuelle
        self.objpoints_l, self.objpoints_r, self.imgpoints_l, self.imgpoints_r= [],[],[],[]
        # Points de la calibration stéréo
        self.objpoints, self.imgpoints_left, self.imgpoints_right = [],[],[]
        # ----------------------------------------------------------------------

        # Détection des points -------------------------------------------------
        for i in range(len(images_left)):

            # lire image de gauche ---------------------------------------------
            color_l = cv.imread(images_left[i])
            gray_l = cv.cvtColor( color_l, self.color_flag)
            ret_l, corners_l = cv.findChessboardCorners(gray_l, self.patternSize, None)

            if ret_l==True:
                corners2_l= cv.cornerSubPix(gray_l, corners_l, self.winSize ,(-1, -1), self.criteria)

                self.objpoints_l.append(self.objp)
                self.imgpoints_l.append(corners2_l)

                # Dessiner le chessboard
                if self.draw==True:
                    _ = cv.drawChessboardCorners(color_l, self.patternSize, corners2_l, True)
                    fname='{}{:03d}.jpg'.format('left', i+1)
                    cv.imwrite(self.single_detected_path + fname, color_l)
            self.imageSize1=gray_l.shape
            # ------------------------------------------------------------------

            # lire image de droite ---------------------------------------------
            color_r = cv.imread(images_right[i])
            gray_r = cv.cvtColor( color_r, self.color_flag)
            ret_r, corners_r = cv.findChessboardCorners(gray_r, self.patternSize, None)

            if ret_r==True:
                corners2_r= cv.cornerSubPix(gray_r, corners_r, self.winSize ,(-1, -1), self.criteria);

                self.objpoints_r.append(self.objp)
                self.imgpoints_r.append(corners2_r)

                # Dessiner le chessboard
                if self.draw==True:
                    _ = cv.drawChessboardCorners(color_r, self.patternSize, corners2_r, True)
                    fname='{}{:03d}.jpg'.format('right', i+1)
                    cv.imwrite(self.single_detected_path + fname, color_r)
            self.imageSize2=gray_r.shape
            # ------------------------------------------------------------------

            # Si détection dans les deux images --------------------------------
            if ret_l*ret_r==1:
                self.objpoints.append(self.objp)
                self.imgpoints_left.append(corners2_l)
                self.imgpoints_right.append(corners2_r)

                # Dessiner les chessboard
                if self.draw==True:
                    _ = cv.drawChessboardCorners(color_l, self.patternSize, corners2_l, True)
                    fname='{}{:03d}.jpg'.format('left', i+1)
                    cv.imwrite(self.stereo_detected_path + fname, color_l)
                    _ = cv.drawChessboardCorners(color_r, self.patternSize, corners2_r, True)
                    fname='{}{:03d}.jpg'.format('right', i+1)
                    cv.imwrite(self.stereo_detected_path + fname, color_r)
            # ------------------------------------------------------------------
        # ----------------------------------------------------------------------

        return None

    def calibrate(self, images_path, stereo_detected_path, single_detected_path):
        """
        ||Public method||
        Calibration individuelle de 2 caméras et calibration stéréo simulatémement

        Args:
            images_path (str): "path_to_images/"
            single_detected_path (str): "path_to_single_images_detected/"
            stereo_detected_path (str): "path_to_stereo_images_detected/"

        """
        # Folders
        self.images_path=images_path
        self.single_detected_path=single_detected_path
        self.stereo_detected_path=stereo_detected_path
        clean_folders([stereo_detected_path])
        clean_folders([single_detected_path])

        # Lire les images et détecter les coins des damier
        self.__read_images(images_path)

        # Calibration individuelle des caméra:
        flags=self.flags_nano
        self.__calibrate_intrinsics(flags)

        # Calibration stéréo des caméras:
        # flags+= cv.CALIB_USE_INTRINSIC_GUESS
        flags+= cv.CALIB_FIX_INTRINSIC
        self.__calibrate_extrinsics(flags)




    def __calibrate_intrinsics(self, flags):

        """
        ||Private method||
        Calibration intrinsèques de 2 caméras : gauche et droite
        """

        print('Calibration individuelle:')
        # GAUCHE
        self.err1, self.M1, self.d1, self.r1, self.t1, self.stdDeviationsIntrinsics1, self.stdDeviationsExtrinsics1, self.perViewErrors1 = cv.calibrateCameraExtended(self.objpoints_l, self.imgpoints_l, self.imageSize1, None, None, flags=flags)

        # DROITE
        self.err2, self.M2, self.d2, self.r2, self.t2, self.stdDeviationsIntrinsics2, self.stdDeviationsExtrinsics2, self.perViewErrors2 = cv.calibrateCameraExtended(self.objpoints_r, self.imgpoints_r, self.imageSize2, None, None, flags=flags)

        # Enlever les outliers -------------------------------------------------
        # gauche
        indices=np.indices(self.perViewErrors1.shape)[0]
        self.indexes_l=indices[self.perViewErrors1<self.err1*1]
        if len(self.indexes_l)>0:
            o_l=np.array(self.objpoints_l)[self.indexes_l]
            i_l=np.array(self.imgpoints_l)[self.indexes_l]
            self.err1, self.M1, self.d1, self.r1, self.t1, self.stdDeviationsIntrinsics1, self.stdDeviationsExtrinsics1, self.perViewErrors1 = cv.calibrateCameraExtended(o_l, i_l, self.imageSize1, None, None, flags=flags)

        # droite
        indices=np.indices(self.perViewErrors2.shape)[0]
        self.indexes_r=indices[self.perViewErrors2<self.err2*1]
        if len(self.indexes_r)>0:
            o_r=np.array(self.objpoints_r)[self.indexes_r]
            i_r=np.array(self.imgpoints_r)[self.indexes_r]
            self.err2, self.M2, self.d2, self.r2, self.t2, self.stdDeviationsIntrinsics2, self.stdDeviationsExtrinsics2, self.perViewErrors2 = cv.calibrateCameraExtended(o_r, i_r, self.imageSize2, None, None, flags=flags)
        # ----------------------------------------------------------------------

        # print('Ecart-type paramètres intrinsèques')
        # print(self.stdDeviationsIntrinsics1,self.stdDeviationsIntrinsics2)
        # Print erreur de reprojection
        print('Erreur de reprojection RMS calibration individuelle')
        print(self.err1, self.err2)


    def __calibrate_extrinsics(self, flags):
        """
        ||Private method||
        """
        print('Calibration stéréo')
        self.errStereo, _, _, _, _, self.R, self.T, self.E, self.F, self.perViewErrors = cv.stereoCalibrateExtended(self.objpoints, self.imgpoints_left, self.imgpoints_right, self.M1, self.d1, self.M2,self.d2, self.imageSize1, self.R, self.T, flags=flags)

        # Enlever les outliers -------------------------------------------------
        indices=np.indices(self.perViewErrors.shape)[0]
        b=self.perViewErrors<self.errStereo*1
        B=b[:,0]*b[:,1]
        indexes=indices[B][:,0]
        if len(indexes)>0:
            o=np.array(self.objpoints)[indexes]
            i_l=np.array(self.imgpoints_left)[indexes]
            i_r=np.array(self.imgpoints_right)[indexes]
            # re-calculs
            self.errStereo, _, _, _, _, self.R, self.T, self.E, self.F, self.perViewErrors = cv.stereoCalibrateExtended(o, i_l, i_r, self.M1, self.d1, self.M2,self.d2, self.imageSize1, self.R, self.T, flags=flags)
        # ----------------------------------------------------------------------

        # Print nombre images valides
        print("Nombre d'images valides")
        print(len(self.perViewErrors))
        # Print erreur de reprojection
        print('Erreur de reprojection RMS calibration stereo')
        print(self.errStereo)


    def saveResultsXML(self, left_name='cam1', right_name='cam2'):

        # Enregistrer caméra 1:
        s = cv.FileStorage()
        s.open('{}.xml'.format(left_name), cv.FileStorage_WRITE)
        s.write('K', self.M1)
        s.write('R', np.eye(3))
        s.write('t', np.zeros((1,3)))
        s.write('coeffs', self.d1)
        s.write('imageSize', self.imageSize1)
        s.write('E', self.E)
        s.write('F', self.F)
        s.release()

        # Enregistrer caméra 2:
        s = cv.FileStorage()
        s.open('{}.xml'.format(right_name), cv.FileStorage_WRITE)
        s.write('K', self.M2)
        s.write('R', self.R)
        s.write('t', self.T)
        s.write('coeffs', self.d2)
        s.write('imageSize', self.imageSize2)
        s.write('E', self.E)
        s.write('F', self.F)
        s.release()

    def reprojection(self, folder, number=None):
        """ Dessiner la reprojection """
        clean_folders([folder])

        # Lire les images
        images_left = np.sort(glob.glob(self.images_path + 'left*.jpg'))
        images_right = np.sort(glob.glob(self.images_path + 'right*.jpg'))
        assert (len(images_right) != 0 or len(images_left) != 0  ), "Images pas trouvées. Vérifier le path"

        # Nombre d'images
        if number is not None:
            images_left = images_left[:number]
            images_right = images_right[:number]

        # Reprojection caméra de gauche
        index=0
        for i in range(len(images_left)):
            ret_l, corners_l, gray_l = find_corners(images_left[i], self.patternSize)
            if ret_l==True:
                draw_reprojection(cv.imread(images_left[i]), self.objpoints_l[index], self.imgpoints_l[index], self.M1, self.d1, self.patternSize, self.squaresize, folder, "left_{}".format(i))
                index+=1

        # Reprojection caméra de droite
        jndex=0
        for j in range(len(images_right)):
            ret_r, corners_r, gray_r = find_corners(images_right[j], self.patternSize)
            if ret_r==True:
                draw_reprojection(cv.imread(images_right[j]), self.objpoints_r[jndex], self.imgpoints_r[jndex], self.M2, self.d2, self.patternSize, self.squaresize, folder, "right_{}".format(j))
                jndex+=1
