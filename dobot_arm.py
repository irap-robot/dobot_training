from DobotDriver import DobotDriver

import cv2 
import numpy as np 
from ImageProc import arucoProjection 

from tensorflow import keras 
import keras 

model = keras.models.load_model("/home/first/dobot_training/can_classification_model.keras")
class_list = ['cube','plain']

def detect_object(projection_image,threshold_value) : 
    gray_image = cv2.cvtColor(projection_image,cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image,threshold_value,255,cv2.THRESH_BINARY_INV)
    countours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    output_image = projection_image.copy()
    object_points = []
    object_class = []
    for c in countours : 
        x,y,w,h = cv2.boundingRect(c)
        area = w*h 
        if area > 5000 : 
            center_pts = np.array([(x+x+w)/2,(y+y+h)/2])
            crop_image = np.copy(projection_image[y:y+h,x:x+w])
            predict_image = cv2.resize(crop_image, (320,320))
            predict_image = predict_image/255.0
            predict_image = np.expand_dims(predict_image,axis=0)
            predict_value = model.predict(predict_image)
            predict_class = np.argmax(predict_value)
            # crop_image = cv2.putText(crop_image,f"{class_list[predict_class]}",(10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            output_image = cv2.rectangle(output_image,(x,y),(x+w,y+h),(0,0,255),3)
            output_image = cv2.circle(output_image,(int(center_pts[0]),int(center_pts[1])),5,(0,0,255),-1)
            output_image = cv2.putText(output_image,f"{class_list[predict_class]}",(x+10,y+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            object_class.append(predict_class)
            object_points.append(center_pts)

    return output_image, np.array(object_points)/2000.0, np.array(object_class)

dobot_arm = DobotDriver()
cam = arucoProjection(0)

dobot_arm.suction_off(wait=True)
dobot_arm.move_on_robot_coordinate(0.205,0.0,0.15,0.0,wait=True)

while True :
    projection_image, marker_image = cam.get_projection_image()
    detected_image, object_points, object_class = detect_object(projection_image,120)

    cv2.imshow("projection image",projection_image)
    cv2.imshow("marker image",marker_image)
    cv2.imshow("detected image",detected_image)

    k = cv2.waitKey(1)
    if k == ord('q') : 
        break 
    elif k == ord('r') :
        for i,pt in enumerate(object_points) :
            print(object_class[i])
            if object_class[i] == 1 : 
                dobot_arm.move_on_marker_coordinate(pt[1],pt[0],-0.05,0.0,wait=True)
                dobot_arm.move_on_marker_coordinate(pt[1],pt[0],-0.075,0.0,wait=True)
                dobot_arm.suction_on(wait=True)
                dobot_arm.move_on_marker_coordinate(pt[1],pt[0],-0.05,0.0,wait=True)
                dobot_arm.move_on_robot_coordinate(0.14727243,0.10148502,-0.04,0.0,wait=True)
                dobot_arm.suction_off(wait=True)
                dobot_arm.move_on_robot_coordinate(0.205,0.0,0.15,0.0,wait=True)

cam.stop_camera()
cv2.destroyAllWindows()

# dobot_arm.set_home()
# dobot_arm.suction_off(wait=True)
# dobot_arm.move_on_robot_coordinate(0.205,0.0,0.15,0.0,wait=True)
# print(dobot_arm.get_endEffectorPose())
# dobot_arm.move_on_robot_coordinate(0.22573332,-0.09276942,-0.04103613,0.0,wait=True)
# dobot_arm.move_on_robot_coordinate(0.22573332,-0.09276942,-0.07103613,0.0,wait=True)
# dobot_arm.suction_on(wait=True)
# dobot_arm.move_on_robot_coordinate(0.22573332,-0.09276942,-0.04103613,0.0,wait=True)
# dobot_arm.move_on_robot_coordinate(0.205,0.0,0.15,0.0,wait=True)
# dobot_arm.move_on_robot_coordinate(0.22573332,-0.09276942,-0.04103613,0.0,wait=True)
# dobot_arm.move_on_robot_coordinate(0.22573332,-0.09276942,-0.07103613,0.0,wait=True)
# dobot_arm.suction_off(wait=True)
# dobot_arm.move_on_robot_coordinate(0.22573332,-0.09276942,-0.04103613,0.0,wait=True)
# dobot_arm.move_on_robot_coordinate(0.205,0.0,0.15,0.0,wait=True)