"""
Created on 30/07/2020

@author: Mireia Graells, ABB Future Labs Switzerland
@email: mireia.graells-vilella@ch.abb.com

"""

import cv2
import numpy as np
from hand_tracker_MediaPipeModels import HandTracker

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark_3d.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_MIDDLE_COLOR = (0, 255, 0) #GREEN (BGR COLORS)
POINT_CLOSE_COLOR = (0, 0, 255) #RED
POINT_FAR_COLOR = (255, 255, 0) #LIGHT BLUE
CONNECTION_COLOR = (255, 0, 0)  #DARK BLUE
THICKNESS = 2

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    if points is not None:
        if points.shape[1] == 3: # if dimension is 3
            minZ = min(points[:,2])
            maxZ = max(points[:,2])
            rangeZ = maxZ - minZ
            for point in points:
                x, y, z = point
                #The more negative the Z value is, the closer to the camera the hand is
                if z < minZ + rangeZ/3:
                    cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_CLOSE_COLOR, THICKNESS)
                elif z < minZ + 2*rangeZ/3:
                    cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_MIDDLE_COLOR, THICKNESS)
                else:
                    cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_FAR_COLOR, THICKNESS)
            points = np.delete(points, 2, axis = 1)
        else: # if dimension is 2
            for point in points:
                x, y = point
                cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_MIDDLE_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
