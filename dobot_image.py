import cv2
import numpy as np
from ImageProc import arucoProjection

def detect_object(projection_image,threshold_value) : 
    gray_image = cv2.cvtColor(projection_image,cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image,threshold_value,255,cv2.THRESH_BINARY_INV)
    countours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    output_image = projection_image.copy()
    object_points = []
    for c in countours : 
        x,y,w,h = cv2.boundingRect(c)
        area = w*h 
        if area > 5000 : 
            center_pts = np.array([(x+x+w)/2,(y+y+h)/2])
            output_image = cv2.rectangle(output_image,(x,y),(x+w,y+h),(0,0,255),3)
            output_image = cv2.circle(output_image,(int(center_pts[0]),int(center_pts[1])),5,(0,0,255),-1)

            object_points.append(center_pts)

    return output_image, np.array(object_points)

cam = arucoProjection(0)

while True : 
    raw_image = cam.get_image()
    projection_image, marker_image = cam.get_projection_image()
    # gray_image = cv2.cvtColor(projection_image,cv2.COLOR_BGR2GRAY)
    # ret, threshold_image = cv2.threshold(gray_image,140,255,cv2.THRESH_BINARY_INV)
    # countours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # vis_image = projection_image.copy()
    # for c in countours :
    #     x,y,w,h = cv2.boundingRect(c)
    #     area = w*h 
    #     # print(area)
    #     if area > 5000 : 
    #         vis_image = cv2.rectangle(vis_image,(x,y),(x+w,y+h),(0,0,255),3)
    #         center_pts = np.array([(x+x+w)/2,(y+y+h)/2])
    #         vis_image = cv2.circle(vis_image,(int(center_pts[0]),int(center_pts[1])),5,(0,0,255),-1)
    detected_image, object_pts = detect_object(projection_image,140) 


    cv2.imshow("process image",detected_image)
    cv2.imshow("projection image",projection_image)
    cv2.imshow("marker image",marker_image)

    k = cv2.waitKey(1) 
    if k == ord('q') : 
        break 

cam.stop_camera()
cv2.destroyAllWindows()
