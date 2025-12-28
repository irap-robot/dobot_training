import pydobot 
import numpy as np 
import math as m 
import os 

class DobotDriver() : 
    def __init__(self,device="/dev/ttyUSB0") : 
        self.device = device 
        self.dobot = pydobot.Dobot(port=self.device)
        self.dobot.speed(50,50)

        self.marker_transformation = self.read_marker_transformation()
       
        # from center to origin
        # note translation change depend on workspace size in ImageProc.py 
        self.tf_center2board = np.array([[-1., 0. ,0. ,0.10], 
                                         [ 0.,-1. ,0. ,0.09],
                                         [ 0., 0. ,1. ,0.  ],
                                         [ 0., 0. ,0. ,1.  ]])
        
        self.tf_robot2board = np.matmul(self.marker_transformation,self.tf_center2board)

    def get_endEffectorPose(self) : 
        pose = self.dobot.get_pose()
        position = pose.position

        return np.array([position.x/1000,position.y/1000,position.z/1000,m.radians(position.r)])  
    
    def move_on_robot_coordinate(self,x,y,z,r,wait=False) : 

        cmd = self.dobot.move_to(x*1000,y*1000,z*1000,m.degrees(r))
        # print(cmd)
        # cmd = self.dobot._set_ptp_cmd(x*1000,y*1000,z*1000,m.degrees(r),mode=MODE_PTP.MOVL_XYZ)

        if wait : 
            self.wait_cmd(cmd)

        return cmd
    
    def move_on_marker_coordinate(self,x,y,z,r,wait=False) :
        goal_pts = np.array([x,y,0,1])

        goal_pts = np.matmul(self.tf_robot2board,goal_pts)
        self.move_on_robot_coordinate(goal_pts[0],goal_pts[1],z,r,wait=wait)
        # print(goal_pts[:3])
        
    def set_home(self) : 
        cmd = self.dobot._set_home_cmd()
        # print(cmd)
        return cmd 
    
    def suction_on(self,wait=False) : 
        cmd = self.dobot.suck(True)

        if wait : 
            self.wait_cmd(cmd)

        return cmd 
    
    def suction_off(self,wait=False) : 
        cmd = self.dobot.suck(False)

        if wait : 
            self.wait_cmd(cmd)

        return cmd 
    
    def wait_cmd(self,id) : 
        while True :
            # print(self.dobot._get_queued_cmd_current_index(),id)
            if self.dobot._get_queued_cmd_current_index() >= id : 
                break 
    
    def read_marker_transformation(self) : 
        main_dir = os.path.dirname(os.path.abspath(__file__))

        with open(f"{main_dir}/config/marker_transformation.txt","r") as file :
            lines = file.readlines()

            transform0 = lines[0].split()
            transform1 = lines[1].split()
            transform2 = lines[2].split()
            transform3 = lines[3].split()

            transform = np.array([[float(transform0[0]),float(transform0[1]),float(transform0[2]),float(transform0[3])],
                                  [float(transform1[0]),float(transform1[1]),float(transform1[2]),float(transform1[3])],
                                  [float(transform2[0]),float(transform2[1]),float(transform2[2]),float(transform2[3])],
                                  [float(transform3[0]),float(transform3[1]),float(transform3[2]),float(transform3[3])]])

            file.close()

        return transform 

    def write_marker_transformation(self,position) :
        main_dir = os.path.dirname(os.path.abspath(__file__))

        with open(f"{main_dir}/config/marker_transformation.txt","w") as file :
            file.write(f"1.0  0.0  0.0 {position[0]}\n")
            file.write(f"0.0  1.0  0.0 {position[1]}\n")
            file.write(f"0.0  0.0  1.0 {position[2]}\n")
            file.write(f"0.0  0.0  0.0 1.0\n")     
            file.close()