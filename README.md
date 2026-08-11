# Yolov5 BoTSORT Tracker

This repository contains a C++ implementation of a multi-object tracking pipeline using YOLOv5 for object detection and BoTSORT for tracking.
## Demo video

<https://drive.google.com/file/d/1MI5A7BKqeD_EL5gyVUe3aoVzLL04nEMf/view?usp=sharing>

## Project structure

- `CMakeLists.txt` - top-level build configuration
- `botsort/` - BoTSORT tracker implementation and supporting modules
- `source/` - example application that runs video inference and tracking
- `yolov5/` - detection model files, TensorRT engine, and configuration data
- `config/` - tracker and re-identification configuration files

## Dependencies

- CMake 3.16 or newer
- C++17 compatible compiler
- OpenCV
- Eigen3
- TensorRT (used by the YOLOv5 TensorRT inference module)

## Build instructions

1. Create a build directory:

```bash
mkdir -p build
cd build
```

2. Configure the project with CMake:

```bash
cmake ..
```

3. Build the project:

```bash
cmake --build .
```

If `BUILD_EXAMPLES` is enabled (default), the example application in `source/` is built.

## Run example

The example binary reads a video file, runs YOLOv5 detection, and tracks objects with BoTSORT.

From the build folder:

```bash
./bin/botsort_tracker_example /path/to/video.mp4
```
