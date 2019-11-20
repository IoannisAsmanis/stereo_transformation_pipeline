CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(IMAGE_INSPECTOR)

set (CMAKE_CXX_STANDARD 11)

add_executable(img_inspector img_inspector.cpp)
add_executable(fov_extractor fov_extractor.cpp)

find_package(OpenCV REQUIRED)
find_package(Threads)
if (THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(img_inspector PUBLIC "-pthread")
endif()
if (CMAKE_THREAD_LIBS_INIT)
    target_compile_definitions(img_inspector PUBLIC "-DMULTI_THREADING")
    target_link_libraries(img_inspector ${CMAKE_THREAD_LIBS_INIT})
endif()

target_link_libraries(img_inspector ${OpenCV_LIBS})
target_link_libraries(fov_extractor ${OpenCV_LIBS})
