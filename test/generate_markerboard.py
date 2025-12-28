import cv2

# 1. Define dictionary and board parameters

MARKER_TYPE = cv2.aruco.DICT_4X4_50 # cv2.aruco.DICT_5X5_50 
arucoParams = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(MARKER_TYPE)
markerLength = 0.030
markerSeperation = 0.002
board = cv2.aruco.GridBoard_create(1, 1, markerLength, markerSeperation, aruco_dict)

# 2. Draw the board
img = board.draw((800, 800))  # size in pixels (height, width)

# 3. Save the image
cv2.imwrite("board1x1_4x4_30mm.png", img)
cv2.imshow("ArUco Board", img)
cv2.waitKey(0)
cv2.destroyAllWindows()