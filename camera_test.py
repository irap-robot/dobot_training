from ImageProc import arucoProjection
import cv2 
import numpy as np 

cam = arucoProjection(4)

threshold_value = 120

kernel = np.ones((5,5),dtype=np.uint8)

while True : 

    projection_image ,marker_image = cam.get_projection_image()
    gray_image = cv2.cvtColor(projection_image, cv2.COLOR_BGR2GRAY ) 

    ret ,blue_mask = cv2.threshold(gray_image,threshold_value,255,cv2.THRESH_BINARY_INV)
    blue_mask = cv2.erode(blue_mask,kernel,iterations=1)
    blue_mask = cv2.dilate(blue_mask,kernel,iterations=1)
    blue_contours ,hierarchy = cv2.findContours(blue_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.imshow('projection', gray_image) 
    cv2.imshow('blue_mask', blue_mask) 
    cv2.imshow('marker', marker_image) 

    key = cv2.waitKey(1) & 0xFF 

    if key == ord('q') : 
        break

cam.stop_camera() 
cv2.destroyAllWindows()