cmake_minimum_required(VERSION 3.0.2)
project(Hector-SLAM-Navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  move_base
  actionlib
  geometry_msgs
  tf
  visualization_msgs
)

find_package(PkgConfig)
pkg_search_module(yaml-cpp REQUIRED yaml-cpp)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES hector-slam-navigation
  CATKIN_DEPENDS actionlib geometry_msgs move_base rospy roscpp tf visualization_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
)