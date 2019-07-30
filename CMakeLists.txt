cmake_minimum_required(VERSION 3.8)
project(rankPoolingTest)

cmake_minimum_required(VERSION 2.8)

# Define project name
project(opencv_rankpooling_project)

# Find OpenCV, need to set OpenCV_DIR variable to absolute path 
# of  directory containing OpenCVConfig.cmake file
set (OpenCV_DIR "${OPENCV_CMAKE_INSTALL_PREFIX}")
find_package(OpenCV REQUIRED)

# If the package is found, several variables will
# be set. Full lists of variables in OpenCVConfig.cmake file
message(STATUS "OpenCV library status")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    cpp file to exec: ${EXEC_CPP_FILE}")

if (CMAKE_VERSION VERSION_LESS "2.8.11")
    # Add OpenCV headers location to include paths
    include_directories(${OPENCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from sources
if(NOT EXEC_CPP_FILE)
    message("EXEC_CPP_FILE must be declared! e.g. >> cmake -DEXEC_CPP_FILE=<YOUR_CPP.cpp>")
endif()
add_executable(main "${EXEC_CPP_FILE}")

# Link application with OpenCV Libraries
target_link_libraries(main ${OpenCV_LIBS})
