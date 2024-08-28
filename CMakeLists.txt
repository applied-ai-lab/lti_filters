cmake_minimum_required(VERSION 3.0.2)
project(lti_filters)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package(
    CATKIN_DEPENDS sensor_msgs
)

# catkin_install_python(PROGRAMS
#     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

include_directories(${catkin_INCLUDE_DIRS})

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
FILE(GLOB python_examples "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*")