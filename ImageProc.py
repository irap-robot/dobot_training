import cv2 
import threading
import os 
import numpy as np 
import time

# from tensorflow import keras
# import keras

class Camera() : 
    def __init__(self,video_id):
        self.video_id = video_id 
        self.camera = cv2.VideoCapture(self.video_id)
        self.image_thread = threading.Thread(target=self.get_raw_image)
        self.close = False 

        self.image = None 
        self.image_thread.start()
        self.camera_matrix ,self.distortion = self.read_camera_matrix()

    def get_raw_image(self) : 
        try : 
            while not self.close : 
                ret , frame = self.camera.read()
                self.image = frame 
        
        except Exception as e : 
            print(f"{str(e)}")

    def stop_camera(self) : 
        self.close = True 
        self.camera.release()

    def read_camera_matrix(self) : 
        main_dir = os.path.dirname(os.path.abspath(__file__))
         
        with open(f"{main_dir}/config/camera_matrix.txt","r") as file :
            lines = file.readlines()
            camMat1 = lines[1].split()
            camMat2 = lines[2].split()
            camMat3 = lines[3].split()
            distort = lines[6].split()

            camera_matrix = np.array([[float(camMat1[0]),float(camMat1[1]),float(camMat1[2])],
                                      [float(camMat2[0]),float(camMat2[1]),float(camMat2[2])],
                                      [float(camMat3[0]),float(camMat3[1]),float(camMat3[2])]])

            distortion = np.array([float(distort[0]),float(distort[1]),float(distort[2]),float(distort[3])])
            file.close()
            
        return camera_matrix ,distortion
    
    def wait_camera_open(self) : 
        while self.image is None :
            # print("waiting image...")
            pass
        
        print("camera is opened!!")
        time.sleep(1)
        
        
class arucoProjection() :

    def __init__(self,video_id):
        self.camera = Camera(video_id)
        self.pts_workspace = np.array([[0.    ,0.20 ],
                                       [0.18  ,0.20 ],
                                       [0.    ,0.   ],
                                       [0.18  ,0.0  ]])
        
        self.offset = np.array([0.097,-0.00])
        self.resolution = 2000
        # self.model = keras.models.load_model("can_classification_model.keras")
        # self.class_list = ['cube', 'plain']

    def get_image(self) : 
        if self.camera.image is not None : 
            return self.camera.image 
        
        else : 
            return np.zeros((480,640,3),dtype=np.uint8)
    
    def get_image_with_marker(self) : 
        if self.camera.image is not None : 
            # print(self.camera.image)
            image_with_marker ,_ = self.detect_marker(self.camera.image)
            return image_with_marker
        else : 
            return np.zeros((480,640,3),dtype=np.uint8) 
        
    def get_projection_image(self) : 
        if self.camera.image is not None :
            image_with_marker ,pts_src = self.detect_marker(self.camera.image)
            
            if pts_src is not None : 
                homoMat ,ret = cv2.findHomography(pts_src,self.pts_workspace*self.resolution)
                projection_image = cv2.warpPerspective(self.camera.image,homoMat,(int(self.pts_workspace[1][0]*self.resolution),int(self.pts_workspace[1][1]*self.resolution)))
                
                return projection_image ,image_with_marker
            else : 
                return np.zeros((480,640,3),dtype=np.uint8) ,self.camera.image 
        else : 
            return np.zeros((480,640,3),dtype=np.uint8) ,np.zeros((480,640,3),dtype=np.uint8) 
    
    def stop_camera(self) : 
        self.camera.stop_camera()

    def detect_marker(self,image) : 
       
        camera_matrix = self.camera.camera_matrix
        distortion_coefficients = self.camera.distortion
        
        rvec = np.zeros((3,1),dtype=np.float64)
        tvec = np.zeros((3,1),dtype=np.float64)

        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
        arucoParams = cv2.aruco.DetectorParameters_create()

        markerLength = 0.0310
        markerSeperation = 0.001 
        
        gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0) 
        # thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

        corners, ids, rejected = cv2.aruco.detectMarkers(gray_img, arucoDict, parameters=arucoParams)
        board = cv2.aruco.GridBoard_create(3, 3, markerLength, markerSeperation, arucoDict)
        cv2.aruco.refineDetectedMarkers(gray_img, board, corners, ids, rejected)

        vis_img = np.copy(image)

        if ids is not None and len(ids) > 0: 
            img_with_marker_board = cv2.aruco.drawDetectedMarkers(vis_img, corners, ids, (0,255,0))
            ret ,rvec ,tvec = cv2.aruco.estimatePoseBoard(corners,ids,board,camera_matrix,distortion_coefficients,rvec,tvec)
            zero_vector = np.zeros((3, 1), np.float32)

            if ret : 
                transformMatrix = self.getTransformMatrix(rvec,tvec)

                workspace_pts = np.copy(self.pts_workspace) 
                workspace_pts[:,0] = workspace_pts[:,0] + self.offset[0]
                workspace_pts[:,1] = workspace_pts[:,1] + self.offset[1]
                top_left_transform = self.relativeTransform([0,0,0],[workspace_pts[0,0],workspace_pts[0,1],0.0])
                top_right_transform = self.relativeTransform([0,0,0],[workspace_pts[1,0],workspace_pts[1,1],0.0])
                bottom_left_transform = self.relativeTransform([0,0,0],[workspace_pts[2,0],workspace_pts[2,1],0.0])
                bottom_right_transform = self.relativeTransform([0,0,0],[workspace_pts[3,0],workspace_pts[3,1],0.0])
                center_transform = self.relativeTransform([0,0,0],[(workspace_pts[1,0]+workspace_pts[2,0])/2,(workspace_pts[1,1]+workspace_pts[2,1])/2,0])
                # top_left_transform = np.array([self.pts_workspace[0,0],self.pts_workspace[0,1],0.])
                # print(top_right_transform)
                
                top_left_transform = np.matmul(transformMatrix,top_left_transform)
                top_right_transform = np.matmul(transformMatrix,top_right_transform)
                bottom_left_transform = np.matmul(transformMatrix,bottom_left_transform)
                bottom_right_transform = np.matmul(transformMatrix,bottom_right_transform)
                center_transform = np.matmul(transformMatrix,center_transform)

                
                top_left_point ,_ = cv2.projectPoints(top_left_transform[:3,3:],zero_vector ,zero_vector ,camera_matrix ,distortion_coefficients)
                top_right_point ,_ = cv2.projectPoints(top_right_transform[:3,3:],zero_vector ,zero_vector ,camera_matrix ,distortion_coefficients)
                bottom_left_point ,_ = cv2.projectPoints(bottom_left_transform[:3,3:],zero_vector ,zero_vector ,camera_matrix ,distortion_coefficients)
                bottom_right_point ,_ = cv2.projectPoints(bottom_right_transform[:3,3:],zero_vector ,zero_vector ,camera_matrix ,distortion_coefficients)

                img_with_marker_board = cv2.drawFrameAxes(img_with_marker_board, camera_matrix, distortion_coefficients, top_left_transform[:3,:3], top_left_transform[:3,3:], 0.05) # TOP LEFT
                img_with_marker_board = cv2.drawFrameAxes(img_with_marker_board, camera_matrix, distortion_coefficients, top_right_transform[:3,:3], top_right_transform[:3,3:], 0.05) # TOP RIGHT
                img_with_marker_board = cv2.drawFrameAxes(img_with_marker_board, camera_matrix, distortion_coefficients, bottom_left_transform[:3,:3], bottom_left_transform[:3,3:], 0.05) # BOTTOM LEFT
                img_with_marker_board = cv2.drawFrameAxes(img_with_marker_board, camera_matrix, distortion_coefficients, bottom_right_transform[:3,:3], bottom_right_transform[:3,3:], 0.05) # BOTTOM RIGHT
                img_with_marker_board = cv2.drawFrameAxes(img_with_marker_board, camera_matrix, distortion_coefficients, center_transform[:3,:3],center_transform[:3,3:],0.05) # CENTER
                
                # img_with_marker_board = cv2.drawFrameAxes(img_with_marker_board, camera_matrix, distortion_coefficients, rvec, tvec, 0.05)

                projection_points = np.array([top_left_point[0,0],
                                              top_right_point[0,0],
                                              bottom_left_point[0,0],
                                              bottom_right_point[0,0]])
            
            return img_with_marker_board ,projection_points
        
        else : 
            return vis_img ,None 
        
    def getTranslationMatrix(self, tvec):
        T = np.identity(n=4)
        T[0:3, 3] = tvec.flatten()

        return T
        
    def getTransformMatrix(self, rvec, tvec):
        mat = self.getTranslationMatrix(tvec)
        R,  _ = cv2.Rodrigues(rvec)
        mat[:3, :3] = R

        return mat
    
    def relativeTransform(self, rotation, translation):
        xC, xS = np.cos(rotation[0]), np.sin(rotation[0])
        yC, yS = np.cos(rotation[1]), np.sin(rotation[1])
        zC, zS = np.cos(rotation[2]), np.sin(rotation[2])
        dX = translation[0]
        dY = translation[1]
        dZ = translation[2]

        Translate_matrix = np.array([[1, 0, 0, dX],
                                    [0, 1, 0, dY],
                                    [0, 0, 1, dZ],
                                    [0, 0, 0, 1]])
        Rotate_X_matrix = np.array([[1, 0, 0, 0],
                                    [0, xC, -xS, 0],
                                    [0, xS, xC, 0],
                                    [0, 0, 0, 1]])
        Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                                    [0, 1, 0, 0],
                                    [-yS, 0, yC, 0],
                                    [0, 0, 0, 1]])
        Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                                    [zS, zC, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
    
        return np.matmul(Rotate_Z_matrix, np.matmul(Rotate_Y_matrix, np.matmul(Rotate_X_matrix, Translate_matrix)))
    
    def get_center(self,contour) :
        M = cv2.moments(contour)
        cX = int(M["m01"] / M["m00"])
        cY = int(M["m10"] / M["m00"])
        return cX ,cY

    def computeContours(self,contours,area_threshold=5000) : 
        areas = []
        pts = []

        for i,c in enumerate(contours) : 
            area = cv2.contourArea(c)
            # print(area)
            if area > area_threshold : 
                cx ,cy = self.get_center(c)
                areas.append(area)
                pts.append(np.array([cx/self.resolution,cy/self.resolution]))

        return np.asarray(pts) ,np.asarray(areas)
    
    def wait_camera_open(self) : 
        self.camera.wait_camera_open()

    # def predict_from_model(self,image) :
    #     predict_image = image.copy()
    #     resized_img = cv2.resize(predict_image, (320, 320))
    #     # print(resized_img.shape)
    #     img_norm = resized_img / 255.0
    #     img_norm = np.expand_dims(img_norm, axis=0)

    #     predictions = self.model.predict(img_norm)
    #     predicted_class = np.argmax(predictions)

    #     cv2.putText(predict_image, f"Predicted: {self.class_list[predicted_class]}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 2)

    #     print(f"Predicted Class: {self.class_list[predicted_class]}")
    #     print(f"Confidence Scores: {predictions[0]}") 

    #     return predict_image, self.class_list[predicted_class]