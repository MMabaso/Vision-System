#!/usr/bin/env python

# Anchor detection: a node for detecting anchors
# Matsilele Mabaso 2019

import numpy as np
import rospy
import cv2 # OpenCV module
import cv2.aruco as aruco

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import math

from sensor_msgs.msg import Image, CameraInfo
import pyzbar.pyzbar as pyzbar
import message_filters
import time
from ros_essentials_cpp.msg import AnchorPosition


# Initializing the node
rospy.init_node('anchor_detection', anonymous=True)

# Bridge to convert ROS Image type to OpenCV Image type
bridge = CvBridge()

# Get the camera calibration parameter for rgb image
msg = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo, timeout=None) 
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
fx = msg.P[0]
fy = msg.P[5]
cx = msg.P[2]
cy = msg.P[6]

# Publisher for publishing anchor positions
topic = '/Anchor_position_topic'
Position_pub = rospy.Publisher(topic, AnchorPosition, queue_size=10)

# Parameters used to detect a circle
Circle_thresh = 14

def image_callback(rgb_data, depth_data):
  #print 'got an image'
  global bridge
  #convert ros_image into an opencv-compatible image
  try:
    cv_image = bridge.imgmsg_to_cv2(rgb_data, "bgr8")
    cv_image_depth = np.array(bridge.imgmsg_to_cv2(depth_data, '32FC1'), dtype = np.float32)
  except CvBridgeError as e:
      print(e)
  # Number of contours in 
  cnts = find_red_maker(cv_image)
  predictedPosition = {}
  keys = predictedPosition
  #Dictionar for anchors true positions, 
  truePosition = {'Anchor 1': (344, 126), 'Anchor 2': (342, 115), 'Anchor 3': (342, 115), 'Anchor 4': (296, 202)}
  
  if len(cnts) > 0:
    c = max(cnts, key = cv2.contourArea)

    # xp and yp are the centers of the eclosing circle, 
    # radius is teh radius of the circle.
    ((xp, yp), radius) = cv2.minEnclosingCircle(c)
    print(radius)
    M = cv2.moments(c)
    #center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    
    #Get the depth value from the depth image
    zc = cv_image_depth[int(yp)][int(xp)]/1000
    if not math.isnan(cv_image_depth[int(yp)][int(xp)]) and cv_image_depth[int(yp)][int(xp)] > 0.1 and cv_image_depth[int(yp)][int(xp)] < 10.0:
      zc = cv_image_depth[int(yp)][int(xp)]/1000

    # only proceed if the radius meets a minimum size. 
    # Correct this value for your obect's size
    if radius > Circle_thresh:
      #barCode = code(cv_image)
      name =  aruco_detect(cv_image) 
      colors = {'yellow':(0, 255, 217)}

      # Draw the bounding circle
      cv2.circle(cv_image, (int(xp), int(yp)), int(radius),(0, 255, 217), 2)
      cv2.putText(cv_image, name, (int(xp-radius), int(yp-radius)), cv2.FONT_HERSHEY_SIMPLEX,  0.6,(0, 0, 0), 2)
      
      xc,yc,xn, yn = getXYZ(xp, yp, zc, fx,fy,cx,cy)
      #predictedPosition[name] = center
      moved = matching(truePosition, predictedPosition, 40)

      # Anchor position object
      position = AnchorPosition()
      position.anchorName = name
      position.xc = xc
      position.yc = yc
      position.zc = zc
      position.moved_or_not = AnchorMove(moved)
      Position_pub.publish(position)
  
  cv2.imshow("Frame", cv_image)
  cv2.waitKey(50)
  #time.sleep(1)


def find_red_maker(cv_image):
  
  hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
  # Define the range of red color in HSV
  lower_red = np.array([170, 50, 50])
  upper_red = np.array([180, 255, 255])

  # Threshold the HSV image to get only red colors
  kernel = np.ones((1, 1), np.uint8)
  mask = cv2.inRange(hsv_image, lower_red, upper_red)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  #cv2.imshow('Mask', mask)

  #Find the contours in the mask 
  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
  return cnts

def AnchorMove(move):
  if len(move) == 0:
    return "NotMoved"
  else:
    return "AnchorMoved"

def HSVObjectDetector(cv_image):
  hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

  #Define the color red color range in HSV
  lower_yellow = np.array([149, 158, 84])
  upper_yellow = np.array([179, 255, 255])

  #THreshold the HSV image to get only the red color
  mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
  mask_eroded = cv2.erode(mask, None, iterations = 2)

  mask_eroded_dilated = cv2.dilate(mask_eroded, None, iterations = 30)
  showImage(cv_image, mask_eroded, mask_eroded_dilated)
  image, contours, hierachy = cv2.findContours(mask_eroded_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  return contours, mask_eroded_dilated

def showImage(cv_image, mask_erode_image, mask_image):
  res = cv2.bitwise_and(cv_image, cv_image, mask = mask_image)

  #cv2.line(cv_image, (320, 325), (320, 245), (255, 0, 0))
  #cv2.line(cv_image, (325, 240), (315, 240), (255, 0, 0))
  #img_pub1.publish(cv_bridge.cv2_to_imgmsg(cv_image, encoding="passthrough"))

def getXYZ(xp, yp, zc, fx,fy,cx,cy):
    ## 
    xn = (xp - cx) / fx
    yn = (yp - cy) / fy
    xc = xn * zc
    yc = yn * zc
    return (xc,yc,xn, yn)

# A function to read a qr_code
def qr_read_decoded(cv_image):
  im = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
  decodedObjects = pyzbar.decode(im)
  return decodedObjects

def code(cv_image):
  decodedObjects = qr_read_decoded(cv_image)
  for decodedObject in decodedObjects:
    points = decodedObject.polygon

    if len(points) > 4:
      hull = cv2.convexHull(np.array([point for point in points], dtype = np.float32))
      hull = list(map(tuple, np.squeeze(hull)))

    else:
      hull = points;

    n = len(hull)

    x = decodedObject.rect.left
    y = decodedObject.rect.top

    barCode1 = str(decodedObject.data)
    barCode = barCode1[0:9]
    if not barCode:
      return "No_name"
    else:
      return barCode

def matching(truePosition, predictedPosition, threshold):
  keys = truePosition.keys()
  moved = []

  for key in keys:
    if key in truePosition and key in predictedPosition:
      point1 = truePosition[key]
      point2 = predictedPosition[key]
      distx = (point1[0] - point2[0])
      disty = (point1[1] - point2[1])
      dist = math.sqrt(distx*distx + disty*disty)

      if dist > threshold:
        moved.append(key)

  return moved

def aruco_detect(image):
  aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
  param = aruco.DetectorParameters_create()
  corners, ids, rejectedImgPoints = aruco.detectMarkers(image, aruco_dict, parameters = param)
  if ids != None:
    name = ids[0]
    name1 = name[0]
    name2 = str(name1)
    name3 = 'Anchor '
    name4 = name3 + name2
    print(name4)
    return name4

  else:
    return str(None)

def main(args):
  # Subscribe to both RGB and Depth images with a Synchronizer
  image_sub = message_filters.Subscriber("camera/rgb/image_rect_color",Image)
  depth_sub = message_filters.Subscriber("camera/depth/image_raw",Image)
  ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.5)
  ts.registerCallback(image_callback)

  rospy.spin()


if __name__ == '__main__':
    main(sys.argv)