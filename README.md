# stereo_transformation_pipeline

A C++ tool used to perform stereo image preprocessing.
Input stereo image locations on disk along with the location of an appropriately configured calibration file
and the program will automatically undistort, rectify and intelligently crop and resize your images to match
your output specification.

## Prerequisites
This software requires:
- Modern Linux kernel (tested on Ubuntu 16.04LTS)
- CMake (>= 2.8)
- C++11
- OpenCV (tested with 2.4.9.1)
- Pthread

## Building
Clone the repository on your system. Then run:

```
cd stereo_transformation_pipeline
mkdir build
cd build
cmake ..
make
```

This should generate an `img_inspector` executable in the `build` folder.

## Usage
To easily use the program use the `launch_img_inspector.sh` script, which comes ready with two examples.
Pay close attention to the notes on how the arguments should be configured.
Also look closely at the example_calibration.yaml file and follow this structure *exactly* for your calibration
files. If this is a major issue, edit the macros at the top of `img_inspector.cpp` and run make again.

## Additional executable
`fov_extractor` is an additional utility that allows you to estimate the Field of View of a camera given its intrinsic parameters.
The tool relies on an OpenCV function that can do more than just that, provided that the camera aperture dimensions are known.
If these are known, pass them in as extra CL args.
