import cv2
import math
import numpy as np

#Ouverture du flux video
cap = cv2.VideoCapture("../Projet2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
diff = 0.0
prev_hist = None
#cap = cv2.VideoCapture(0)

ret, frame = cap.read() # Passe à l'image suivante
index = 0

while(ret):
    ret, frame = cap.read()
    yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)

    hist = cv2.calcHist([yuv], [1,2], None, [256,256], [0,255,0,255])

    if prev_hist is not None:
        #diff = cv2.compareHist(hist,prev_hist,cv2.HISTCMP_BHATTACHARYYA)
        diff = np.sum(np.abs(hist.astype(np.float32) - prev_hist.astype(np.float32)))


    hist_norm = cv2.GaussianBlur(hist,(5,5),cv2.BORDER_DEFAULT)
    hist_norm = ((hist_norm*255.0)/np.amax(hist_norm)).astype(np.uint8)
    cv2.imshow('Image',frame)
    hist_display = cv2.applyColorMap(hist_norm,cv2.COLORMAP_JET)
    cv2.imshow('Histogramme (u,v)',hist_display)
    
    prev_hist = hist.copy()
    
    print("Frame %d, Différence : %f"%(index,diff))
    index += 1

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame)
        cv2.imwrite('Hist_UV_%04d.png'%index,hist_display)


cap.release()
cv2.destroyAllWindows()
