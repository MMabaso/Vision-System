#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import math
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
#import matplotlib.patches as mpatches
from matplotlib.pylab import plot, show, ion
import matplotlib.pyplot as plt
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import pyzbar.pyzbar as pyzbar
from skimage import measure
import message_filters
import time

bridge = CvBridge()
# Get the camera calibration parameter for the rectified image
#msg = rospy.wait_for_message('.ros/camera/rgb/camera_info/rgb_1406030525.yaml', CameraInfo, timeout=None) 
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]

fx = 516.3324584960938
fy = 520.5577392578125
cx = 310.9759851787312
cy = 245.6342644189317

#fx = msg.P[0]
#fy = msg.P[5]
#cx = msg.P[2]
#cy = msg.P[6]



def image_callback(rgb_data, depth_data):
  print 'got an image'
  global bridge
  #convert ros_image into an opencv-compatible image
  try:
    cv_image = bridge.imgmsg_to_cv2(rgb_data, "bgr8")
    cv_image_depth = np.array(bridge.imgmsg_to_cv2(depth_data, '32FC1'), dtype = np.float32)
  except CvBridgeError as e:
      print(e)

  

  #plt.imshow(cv_image_depth)
  #plt.show()
  cnts = find_red_maker(cv_image)
  #cnts = contours
  #print(contours)
  predictedPosition = {}
  keys = predictedPosition
  #print("the keys are: ", keys)
  #print("The predicted position is: ", )

  #Anchors true positions 
  truePosition = {'Anchor 1': (344, 126), 'Anchor 2': (342, 115), 'Anchor 3': (342, 115), 'Anchor 4': (296, 202)}
  #print("The true position is :", truePosition)

  if len(cnts) > 0:
    c = max(cnts, key = cv2.contourArea)
    boundingBox = cv2.minAreaRect(c)
    #c = max(cnts, key = cv2.contourArea)
    ((xp, yp), radius) = cv2.minEnclosingCircle(c)
    #print('The radius is: ', radius)
    M = cv2.moments(c)
    center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    zc = cv_image_depth[int(yp)][int(xp)]/1000
    #print('The depth_value for point: ',(xp, yp),"The center is", center, "is: ", zc)
    #if not math.isnan(cv_image_depth[int(yp)][int(xp)]) and cv_image_depth[int(yp)][int(xp)] > 0.1 and cv_image_depth[int(yp)][int(xp)] < 10.0:
    #  zc = cv_image_depth[int(yp)][int(xp)]/1000
    #  print('The depth_value for point: ',(xp, yp), "is: ", zc)

    if radius > 0.5:
      barCode = code(cv_image) 
      colors = {'yellow':(0, 255, 217)}
      cv2.circle(cv_image, (int(xp), int(yp)), int(radius),(0, 255, 217), 2)
      cv2.putText(cv_image, barCode, (int(xp-radius), int(yp-radius)), cv2.FONT_HERSHEY_SIMPLEX,  0.6,(0, 0, 0), 3)
      #Get the depth value from the depth image
      xc,yc,zc = getXYZ(xp, yp, zc, fx,fy,cx,cy)
      #print(xc, yc, zc)
      predictedPosition[barCode] = center
      #predictedPosition.update([barCode, (center)])
      #print("THe predicted: ",predictedPosition)
      moved = matching(truePosition, predictedPosition, 40)

      if len(moved) == 0:
        print('The :',barCode, 'did not move')
      else:
        print('The :', barCode, 'moved')
        #cv2.circle(cv_image, (int(296), int(202)), int(radius),(0, 255, 217), 2)
        #cv2.putText(cv_image, 'Old position', (int(296-radius), int(202-radius)), cv2.FONT_HERSHEY_SIMPLEX,  0.6,(0, 0, 0), 1)
        #cv2.line(cv_image, (296, 202), (int(xp), int(yp)), (0, 255, 0), 2)
        #cv2.circle(cv_image, (int(296), int(202)), 1,(0, 255, 217), 2)
        #cv2.circle(cv_image, (int(xp), int(yp)), 1,(0, 255, 217), 2)
      print(moved)
      #print('The anchor has moved: ', matching(truePosition, predictedPosition, 30))

      #time.sleep(0.5)

  cv2.imshow("Frame", cv_image)
  cv2.waitKey(50)


def find_red_maker(cv_image):

    #Define the color yellow color range in HSV
  lower_yellow = np.array([120, 128, 84])
  upper_yellow = np.array([179, 255, 255])


  blurred = cv2.GaussianBlur(cv_image, (11, 11), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
  kernel = np.ones((9, 9), np.uint8)
  mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
  cv2.imshow("Mask", mask)
  cv2.waitKey(50)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  #Find the contours in the mask 
  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
  #if max(cnts) > 0:
    
  #c = max(cnts, key = cv2.contourArea)
  #print("The c is: ", c)
  #boundingBox = cv2.minAreaRect(c)
  return cnts

def HSVObjectDetector(cv_image):
  hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

  #Define the color yellow color range in HSV
  lower_yellow = np.array([149, 158, 84])
  upper_yellow = np.array([179, 255, 255])

  #THreshold the HSV image to get only the yellow color
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

def getXYZ(xp, yp, zc, fx,fy,cx,cy):
    ## 
    xn = (xp - cx) / fx
    yn = (yp - cy) / fy
    xc = xn * zc
    yc = yn * zc
    return (xc,yc,zc)


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



def main(args):
  rospy.init_node('object_localizer', anonymous=True)
  #for turtlebot3 waffle
  #image_topic="/camera/rgb/image_raw/compressed"
  #for usb cam
  #image_topic="/usb_cam/image_raw"
  image_sub = message_filters.Subscriber("camera/rgb/image_rect_color",Image)
  depth_sub = message_filters.Subscriber("camera/depth/image_raw",Image)
  #depth_sub = message_filters.Subscriber("camera/depth/points",Image)
  ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.5)
  ts.registerCallback(image_callback)
  #msg = message_filters.Subscriber('/camera/rgb/camera_info/', CameraInfo, timeout=None) 
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)