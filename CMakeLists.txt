CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(IMAGE_INSPECTOR)

set (CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
add_executable(img_inspector img_inspector.cpp)
target_link_libraries(img_inspector ${OpenCV_LIBS})
