#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This import is for general library
import os
import threading

# This import is for ROS integration
import rospy
import message_filters
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes,BoundingBox
import cv2

class PersonDetector():
    def __init__(self):

        # cv_bridge handles
        self.cv_bridge = CvBridge()

        self.person_bbox = BoundingBox()

        # ROS PARAM
        self.m_pub_threshold = rospy.get_param('~pub_threshold', 0.60)

        # detect width height
        self.WIDTH  = 100
        self.HEIGHT = 100

        # Subscribe
        sub_camera_rgb     =  message_filters.Subscriber("/camera/color/image_raw",Image)
        sub_camera_depth   =  message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",Image)
        #sub_camera_depth   =  rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.CamDepthImageCallback)
        sub_darknet_bbox   =  rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.DarknetBboxCallback)
        self.mf = message_filters.ApproximateTimeSynchronizer([sub_camera_rgb, sub_camera_depth], 100, 10.0)
        self.mf.registerCallback(self.ImageCallback)

        return

    def ImageCallback(self, rgb_image_data , depth_image_data):
        try:
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_image_data, 'passthrough')
            self.m_depth_image = self.cv_bridge.imgmsg_to_cv2(depth_image_data, 'passthrough')
        except CvBridgeError, e:
            rospy.logerr(e)

        rgb_image.flags.writeable = True
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_image.shape

        # 人がいる場合
        if self.person_bbox.probability > 0.0:

           # 一旦、BoundingBoxの中心位置の深度を取得 (今後改善予定）
            #m_person_depth = self.m_depth_image[(int)(self.person_bbox.ymax+self.person_bbox.ymin)/2][(int)(self.person_bbox.xmax+self.person_bbox.xmin)/2]

            """
            x1 = self.person_bbox.xmin + self.WIDTH 
            x2 = self.person_bbox.xmax - self.WIDTH 
            y1 = self.person_bbox.ymin + self.HEIGHT
            y2 = self.person_bbox.ymax - self.HEIGHT
            """

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
    
            #ave = sum / (((x2 - x1) * 2) * ((y2 - y1)* 2))
            ave = sum / ((self.WIDTH * 2) * (self.HEIGHT * 2))
            #print("%f [m]" % ave)
            rospy.loginfo('Class : person, Score: %.2f, Dist: %dmm ' %(self.person_bbox.probability, ave))

            #cv2.normalize(self.m_depth_image, self.m_depth_image, 0, 1, cv2.NORM_MINMAX)
            #cv2.namedWindow("color_image")
            #cv2.namedWindow("depth_image")
            #cv2.imshow("color_image", rgb_image)
            #cv2.imshow("depth_image", self.m_depth_image)
            #cv2.waitKey(10)

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
        person_bbox = BoundingBox()
        if len(bboxs) != 0 :
            for i, bb in enumerate(bboxs) :
                if bboxs[i].Class == 'person' and bboxs[i].probability >= self.m_pub_threshold:
                    person_bbox = bboxs[i]        
        self.person_bbox = person_bbox


if __name__ == '__main__':
    try:
        rospy.init_node('person_detector', anonymous=True)
        idc = PersonDetector()
        rospy.loginfo('idc Initialized')
        #idc.start()
        rospy.spin()
        #idc.finish()

    except rospy.ROSInterruptException:
        pass