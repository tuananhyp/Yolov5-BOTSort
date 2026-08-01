CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

PROJECT(botsort_tracker VERSION 1.0 LANGUAGES CXX)

# Set Build Type if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()

# Define the executable
add_executable(${PROJECT_NAME} main.cpp)

# Find and link OpenCV
set(OpenCV_DIR "/opt/opencv-4.5.5/lib/cmake/opencv4")  
find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${botsort_INCLUDE_DIRS})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} botsort)

# Gstreamer 
find_package(PkgConfig REQUIRED)

find_package(Boost COMPONENTS filesystem program_options thread REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

pkg_check_modules(GLIB REQUIRED glib-2.0)

include_directories(
  ${GLIB_INCLUDE_DIRS}
)

link_directories(
  ${GLIB_LIBRARY_DIRS}
)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/x86_64-linux-gnu)

set(GSTREAMER_MINIMUM_VERSION 1.0.5)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavdevice
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)

pkg_check_modules(GST1_TEST gstreamer-1.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
    pkg_check_modules(GOBJECT REQUIRED gobject-2.0)
    pkg_check_modules(GSTREAMER-APP REQUIRED gstreamer-app-1.0)
    pkg_check_modules(GSTREAMER-VIDEO REQUIRED gstreamer-video-1.0)
    pkg_check_modules(GSTREAMER-BASE REQUIRED gstreamer-base-1.0)
    pkg_check_modules(GSTREAMER-AUDIO REQUIRED gstreamer-audio-1.0)
    pkg_check_modules(GSTREAMER-PBUTILS REQUIRED gstreamer-pbutils-1.0)
    pkg_check_modules(GSTREAMER-FFT REQUIRED gstreamer-fft-1.0)
    add_definitions(-DGST_API_VERSION_1=1)

include_directories(
  ${GSTREAMER_INCLUDE_DIRS}
${GOBJECT_INCLUDE_DIRS}
  ${GSTREAMER-APP_INCLUDE_DIRS}
  ${GSTREAMER-VIDEO_INCLUDE_DIRS}
  ${GSTREAMER-BASE_INCLUDE_DIRS}
  ${GSTREAMER-AUDIO_INCLUDE_DIRS}
  ${GSTREAMER-PBUTILS_INCLUDE_DIRS}
  ${GSTREAMER-FFT_INCLUDE_DIRS}
  ${GSTREAMER_RTSP_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
  ${GSTREAMER_LIBRARY_DIRS}
${GOBJECT_LIBRARY_DIRS}
  ${GSTREAMER-APP_LIBRARY_DIRS}
  ${GSTREAMER-VIDEO_LIBRARY_DIRS}
  ${GSTREAMER-BASE_LIBRARY_DIRS}
  ${GSTREAMER-AUDIO_LIBRARY_DIRS}
  ${GSTREAMER-PBUTILS_LIBRARY_DIRS}
  ${GSTREAMER-FFT_LIBRARY_DIRS}
)


set(gst_LIBRARIES
  ${GSTREAMER_LIBRARIES}
${GOBJECT_LIBRARIES}
  ${GSTREAMER-APP_LIBRARIES}
  ${GSTREAMER-VIDEO_LIBRARIES}
  ${GSTREAMER-BASE_LIBRARIES}
  ${GSTREAMER-AUDIO_LIBRARIES}
  ${GSTREAMER-PBUTILS_LIBRARIES}
  ${GSTREAMER-FFT_LIBRARIES}
    
  pthread
  ${GLIB_LIBRARIES}
  ${GLIB_GIO_LIBRARIES}
  ${GLIB_GOBJECT_LIBRARIES}
)

set(GSTREAMER_LIBS
  gstreamer-1.0
  gstbase-1.0
  gstrtp-1.0
  gstsdp-1.0
  gstrtsp-1.0
  gstapp-1.0
  gstnet-1.0
  gobject-2.0
  gmodule-2.0
  gthread-2.0
  glib-2.0
  gio-2.0
)

# Import .so library yolov5

add_library(yolov5trt SHARED IMPORTED)
set_target_properties(yolov5trt PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../yolov5/yolov5trt/lib/libyolov5trt.so)
target_link_libraries(${PROJECT_NAME} yolov5trt)

add_library(yolov5plugins SHARED IMPORTED)
set_target_properties(yolov5plugins PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../yolov5/yolov5trt/lib/libyolov5plugins.so)
target_link_libraries(${PROJECT_NAME} yolov5plugins)

add_library(nvidiatrt SHARED IMPORTED)
set_target_properties(nvidiatrt PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../yolov5/nvidia_trt/lib/libnvidia_trt.so)
target_link_libraries(${PROJECT_NAME} nvidiatrt)


target_link_libraries(${PROJECT_NAME} nvinfer cudart)
target_link_libraries(${PROJECT_NAME} "${gst_LIBRARIES}")
target_link_libraries(${PROJECT_NAME}  "${GSTREAMER_LIBS}")
target_link_libraries(${PROJECT_NAME}  PkgConfig::LIBAV)

# Include yolov5_trt.h

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../yolov5/yolov5trt/include)
