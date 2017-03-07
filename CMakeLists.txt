cmake_minimum_required(VERSION 3.6)
project(Visualizer)

set(CMAKE_CXX_STANDARD 14)


link_directories(/usr/local/lib/ /usr/lib/)
include_directories(/usr/local/include/)

set(SOURCE_FILES main.cpp)
add_executable(Visualizer ${SOURCE_FILES})
target_link_libraries(Visualizer opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
