#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This import is for general library
import os
import threading

# This import is for ROS integration
import rospy
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes,BoundingBox
import cv2

class OutdoorDetector():
    def __init__(self):

        # cv_bridge handles
        self.cv_bridge = CvBridge()
        
        # outdoor object
        self.person_bbox        = BoundingBox()
        self.bicycle_bbox       = BoundingBox()
        self.car_bbox           = BoundingBox()
        self.motorcycle_bbox    = BoundingBox()
        self.bus_bbox           = BoundingBox()
        self.truck_bbox         = BoundingBox()
        self.traffic_light_bbox = BoundingBox()
        self.stop_sign_bbox     = BoundingBox()


        # ROS PARAM
        self.m_pub_threshold = rospy.get_param('~pub_threshold', 0.70)

        # detect width height
        self.WIDTH  = 50
        self.HEIGHT = 50

        # Subscribe
        sub_camera_rgb    =  rospy.Subscriber('/camera/color/image_raw', Image, self.CamRgbImageCallback)
        sub_camera_depth  =  rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.CamDepthImageCallback)
        sub_darknet_bbox  =  rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.DarknetBboxCallback)

        return

    def CamRgbImageCallback(self, rgb_image_data):
        try:
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_image_data, 'passthrough')
        except CvBridgeError, e:
            rospy.logerr(e)

        rgb_image.flags.writeable = True
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_image.shape

        # person exist
        if self.person_bbox.probability > 0.0:

            x1 = (w / 2) - self.WIDTH
            x2 = (w / 2) + self.WIDTH
            y1 = (h / 2) - self.HEIGHT
            y2 = (h / 2) + self.HEIGHT

            sum = 0.0
            for i in range(y1, y2):
                for j in range(x1, x2):
                    rgb_image.itemset((i, j, 0), 0)
                    rgb_image.itemset((i, j, 1), 0)
                    if self.m_depth_image.item(i,j) == self.m_depth_image.item(i,j):
                        sum += self.m_depth_image.item(i,j)
    
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            rospy.loginfo('Class : person, Score: %.2f, Dist: %dmm ' %(self.person_bbox.probability, ave))
            
        # bicycle exist
        elif self.bicycle_bbox.probability > 0.0:

            x1 = (w / 2) - self.WIDTH
            x2 = (w / 2) + self.WIDTH
            y1 = (h / 2) - self.HEIGHT
            y2 = (h / 2) + self.HEIGHT

            sum = 0.0
            for i in range(y1, y2):
                for j in range(x1, x2):
                    rgb_image.itemset((i, j, 0), 0)
                    rgb_image.itemset((i, j, 1), 0)
                    if self.m_depth_image.item(i,j) == self.m_depth_image.item(i,j):
                        sum += self.m_depth_image.item(i,j)
    
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            rospy.loginfo('Class : bicycle, Score: %.2f, Dist: %dmm ' %(self.bicycle_bbox.probability, ave))

        # car exist
        elif self.car_bbox.probability > 0.0:

            x1 = (w / 2) - self.WIDTH
            x2 = (w / 2) + self.WIDTH
            y1 = (h / 2) - self.HEIGHT
            y2 = (h / 2) + self.HEIGHT

            sum = 0.0
            for i in range(y1, y2_bbox):
                for j in range(x1, x2):
                    rgb_image.itemset((i, j, 0), 0)
                    rgb_image.itemset((i, j, 1), 0)
                    if self.m_depth_image.item(i,j) == self.m_depth_image.item(i,j):
                        sum += self.m_depth_image.item(i,j)
    
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            rospy.loginfo('Class : car, Score: %.2f, Dist: %dmm ' %(self.car_bbox.probability, ave))
        
        # motorcycle exist
        elif self.motorcycle_bbox.probability > 0.0:

            x1 = (w / 2) - self.WIDTH
            x2 = (w / 2) + self.WIDTH
            y1 = (h / 2) - self.HEIGHT
            y2 = (h / 2) + self.HEIGHT

            sum = 0.0
            for i in range(y1, y2):
                for j in range(x1, x2):
                    rgb_image.itemset((i, j, 0), 0)
                    rgb_image.itemset((i, j, 1), 0)
                    if self.m_depth_image.item(i,j) == self.m_depth_image.item(i,j):
                        sum += self.m_depth_image.item(i,j)
    
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            rospy.loginfo('Class : motorcycle, Score: %.2f, Dist: %dmm ' %(self.motorcycle_bbox.probability, ave))
        
        # bus exist
        elif self.bus_bbox.probability > 0.0:

            x1 = (w / 2) - self.WIDTH
            x2 = (w / 2) + self.WIDTH
            y1 = (h / 2) - self.HEIGHT
            y2 = (h / 2) + self.HEIGHT

            sum = 0.0
            for i in range(y1, y2):
                for j in range(x1, x2):
                    rgb_image.itemset((i, j, 0), 0)
                    rgb_image.itemset((i, j, 1), 0)
                    if self.m_depth_image.item(i,j) == self.m_depth_image.item(i,j):
                        sum += self.m_depth_image.item(i,j)
    
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            rospy.loginfo('Class : bus, Score: %.2f, Dist: %dmm ' %(self.bus_bbox.probability, ave))
        
        # truck exist
        elif self.truck_bbox.probability > 0.0:

            x1 = (w / 2) - self.WIDTH
            x2 = (w / 2) + self.WIDTH
            y1 = (h / 2) - self.HEIGHT
            y2 = (h / 2) + self.HEIGHT

            sum = 0.0
            for i in range(y1, y2):
                for j in range(x1, x2):
                    rgb_image.itemset((i, j, 0), 0)
                    rgb_image.itemset((i, j, 1), 0)
                    if self.m_depth_image.item(i,j) == self.m_depth_image.item(i,j):
                        sum += self.m_depth_image.item(i,j)
    
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            rospy.loginfo('Class : truck, Score: %.2f, Dist: %dmm ' %(self.truck_bbox.probability, ave))
        
        # traffic light exist
        elif self.traffic_light_bbox.probability > 0.0:

            x1 = (w / 2) - self.WIDTH
            x2 = (w / 2) + self.WIDTH
            y1 = (h / 2) - self.HEIGHT
            y2 = (h / 2) + self.HEIGHT

            sum = 0.0
            for i in range(y1, y2):
                for j in range(x1, x2):
                    rgb_image.itemset((i, j, 0), 0)
                    rgb_image.itemset((i, j, 1), 0)
                    if self.m_depth_image.item(i,j) == self.m_depth_image.item(i,j):
                        sum += self.m_depth_image.item(i,j)
    
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            rospy.loginfo('Class : traffic light, Score: %.2f, Dist: %dmm ' %(self.traffic_light_bbox.probability, ave))
        
        # stop sign exist
        elif self.stop_sign_bbox.probability > 0.0:

            x1 = (w / 2) - self.WIDTH
            x2 = (w / 2) + self.WIDTH
            y1 = (h / 2) - self.HEIGHT
            y2 = (h / 2) + self.HEIGHT

            sum = 0.0
            for i in range(y1, y2):
                for j in range(x1, x2):
                    rgb_image.itemset((i, j, 0), 0)
                    rgb_image.itemset((i, j, 1), 0)
                    if self.m_depth_image.item(i,j) == self.m_depth_image.item(i,j):
                        sum += self.m_depth_image.item(i,j)
    
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            rospy.loginfo('Class : stop sign, Score: %.2f, Dist: %dmm ' %(self.stop_sign_bbox.probability, ave))

            """
            cv2.normalize(self.m_depth_image, self.m_depth_image, 0, 1, cv2.NORM_MINMAX)
            cv2.namedWindow("color_image")
            cv2.namedWindow("depth_image")
            cv2.imshow("color_image", rgb_image)
            cv2.imshow("depth_image", self.m_depth_image)
            cv2.waitKey(10)
            """

            """
            cv2.rectangle(rgb_image, (self.person_bbox.xmin, self.person_bbox.ymin), (self.person_bbox.xmax, self.person_bbox.ymax),(0,0,255), 2)
            #rospy.loginfo('Class : person, Score: %.2f, Dist: %dmm ' %(self.person_bbox.probability, m_person_depth))
            text = "person " +('%dmm' % ave)
            text_top = (self.person_bbox.xmin, self.person_bbox.ymin - 10)
            text_bot = (self.person_bbox.xmin + 80, self.person_bbox.ymin + 5)
            text_pos = (self.person_bbox.xmin + 5, self.person_bbox.ymin)
            cv2.rectangle(rgb_image, text_top, text_bot, (0,0,0),-1)
            cv2.putText(rgb_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        cv2.namedWindow("rgb_image")
        cv2.imshow("rgb_image", rgb_image)
        cv2.waitKey(10)
        cv2.normalize(self.m_depth_image, self.m_depth_image, 0, 32768, cv2.NORM_MINMAX)
        cv2.namedWindow("depth_image")
        cv2.imshow("depth_image", self.m_depth_image)
        cv2.waitKey(10)
        """
        return

    def CamDepthImageCallback(self, depth_image_data):
        try:
            self.m_depth_image = self.cv_bridge.imgmsg_to_cv2(depth_image_data, 'passthrough')
        except CvBridgeError, e:
            rospy.logerr(e)
        self.m_camdepth_height, self.m_camdepth_width = self.m_depth_image.shape[:2]
        return

    def DarknetBboxCallback(self, darknet_bboxs):
        bboxs = darknet_bboxs.bounding_boxes

        person_bbox        = BoundingBox()
        bicycle_bbox       = BoundingBox()
        car_bbox           = BoundingBox()
        motorcycle_bbox    = BoundingBox()
        bus_bbox           = BoundingBox()
        truck_bbox         = BoundingBox()
        traffic_light_bbox = BoundingBox()
        stop_sign_bbox     = BoundingBox()

        if len(bboxs) != 0 :
            for i, bb in enumerate(bboxs) :
                if bboxs[i].Class == 'person' and bboxs[i].probability >= self.m_pub_threshold:
                    person_bbox = bboxs[i]        
                elif bboxs[i].Class == 'bicycle' and bboxs[i].probability >= self.m_pub_threshold:
                    bicycle_bbox = bboxs[i]    
                elif bboxs[i].Class == 'car' and bboxs[i].probability >= self.m_pub_threshold:
                    car_bbox = bboxs[i]    
                elif bboxs[i].Class == 'motorcycle' and bboxs[i].probability >= self.m_pub_threshold:
                    motorcycle_bbox = bboxs[i]    
                elif bboxs[i].Class == 'bus' and bboxs[i].probability >= self.m_pub_threshold:
                    bus_bbox = bboxs[i]    
                elif bboxs[i].Class == 'truck' and bboxs[i].probability >= self.m_pub_threshold:
                    truck_bbox = bboxs[i]    
                elif bboxs[i].Class == "traffic light" and bboxs[i].probability >= self.m_pub_threshold:
                    traffic_light_bbox = bboxs[i]    
                elif bboxs[i].Class == "stop sign" and bboxs[i].probability >= self.m_pub_threshold:
                    stop_sign_bbox = bboxs[i]    

        self.person_bbox        = person_bbox
        self.bicycle_bbox       = bicycle_bbox
        self.car_bbox           = car_bbox
        self.motorcycle_bbox    = motorcycle_bbox
        self.bus_bbox           = bus_bbox
        self.truck_bbox         = truck_bbox
        self.traffic_light_bbox = traffic_light_bbox
        self.stop_sign_bbox     = stop_sign_bbox

if __name__ == '__main__':
    try:
        rospy.init_node('person_detector', anonymous=True)
        idc = OutdoorDetector()
        rospy.loginfo('idc Initialized')
        rospy.spin()

    except rospy.ROSInterruptException:
        pass