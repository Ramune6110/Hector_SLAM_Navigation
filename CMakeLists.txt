cmake_minimum_required(VERSION 3.0.2)
project(Hector_SLAM_Navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  move_base
  actionlib
  geometry_msgs
  tf
  visualization_msgs
  std_msgs
  message_generation
  #Hector_SLAM_Navigation
)

find_package(PkgConfig)
pkg_search_module(yaml-cpp REQUIRED yaml-cpp)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  object_data.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES Hector_SLAM_Navigation
  CATKIN_DEPENDS actionlib geometry_msgs move_base rospy roscpp tf visualization_msgs std_msgs message_runtime
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
)