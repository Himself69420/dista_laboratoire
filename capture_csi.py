##############################################################################################################
# Projet : Integration cours initiation                                                                      #
# Description Script : Effectuer une capture synchrone de deux camera                                        #
# Date : 29 aout 2020                                                                                        #
# Auteur : Carl Beaulieu                                                                                     #
##############################################################################################################

import cv2
import nanocamera as nano

# Variables textuel
text_title="Assistant de calibration"
text_progress=""

# Nombres de captures prises
capture_taken=0
# Nombres de captures à prendre
capture_qty=40

# Déclaration du périphérique de capture
nano.Camera()
cap_left = nano.Camera(device_id=0,flip=2,width=800,height=480,fps=30)
cap_right = nano.Camera(device_id=1,flip=2,width=800,height=480,fps=30)

while(True):
    if capture_taken==0:
        text_progress="Appuyez sur 'C' pour capturer les images"
    elif capture_taken==capture_qty:
        text_progress="Termine, maintenir 'Esc' pour quitter."
    else: 
        text_progress="Il reste " + str(capture_qty-capture_taken)+ " image(s) a prendre."
    
    # Acquisition du flux video
    left_frame_color = cap_left.read()
    right_frame_color = cap_right.read()
        
    
    #Transformation de l'image en nuances de gris
    left_frame = cv2.cvtColor(left_frame_color, cv2.COLOR_BGR2GRAY)
    right_frame = cv2.cvtColor(right_frame_color, cv2.COLOR_BGR2GRAY)

    # Enregistrement des deux frames sans écritures
    if cv2.waitKey(1) == ord('c'):
        if capture_taken<capture_qty:
            capture_taken = capture_taken + 1
            text_progress=""
            filename_l="left" + str(capture_taken) + ".jpg"
            filename_r="right" + str(capture_taken) + ".jpg"
            cv2.imwrite(filename_l, left_frame)
            cv2.imwrite(filename_r, right_frame)
        
    # Affichage du contenu
    left_view = left_frame_color #cv2.resize(left_frame_color,(480,270))
    right_view = right_frame_color #cv2.resize(right_frame_color,(480,270))
    
    
    left_view=cv2.flip(left_view,1)    
    right_view=cv2.flip(right_view,1)    
    
    
    cv2.putText(left_view, text_progress, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(right_view, text_progress, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow(text_title + ' Camera Gauche',left_view)
    cv2.imshow(text_title + ' Camera Droite',right_view)
        
    if cv2.waitKey(1) == 27:
        break
    
# On relache la connexion
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()