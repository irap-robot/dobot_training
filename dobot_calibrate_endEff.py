from DobotDriver import DobotDriver 
from ImageProc import arucoProjection
import cv2 

dobot_arm = DobotDriver()
cam = arucoProjection(0)

while True : 

    projection_image ,marker_image = cam.get_projection_image()
    (w,h,_) = projection_image.shape
    projection_image =  cv2.circle(projection_image,(int(h/2),int(w/2)),10,(0,0,255),-1)

    cv2.imshow('projection', projection_image) 
    cv2.imshow('marker', marker_image) 

    key = cv2.waitKey(1) & 0xFF 

    if key == ord('q') : 
        break

cam.stop_camera() 
cv2.destroyAllWindows()
        

end_effector_pose = dobot_arm.get_endEffectorPose()
print(end_effector_pose[:3])
dobot_arm.write_marker_transformation(end_effector_pose[:3])

dobot_arm.move_on_robot_coordinate(0.205,0.0,0.15,0.0,wait=True)
print("Set Home")

dobot_arm.set_home()
print("Move to Origin")

dobot_arm.move_on_robot_coordinate(0.205,0.0,0.15,0.0,wait=True)
dobot_arm.move_on_robot_coordinate(end_effector_pose[0],end_effector_pose[1],end_effector_pose[2]+0.01,0.0,wait=True)




