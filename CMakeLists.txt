cmake_minimum_required(VERSION 3.0.2)
project(Hector-SLAM-Navigation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  move_base
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES hector-slam-navigation
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)