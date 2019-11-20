## Remember:
# All arguments are mandatory
# Directory indeces are inclusive
# Paths must end in '/'
# Calibration files must adhere to the specified format, see macros of source code and example file!

# Example call of the executable in the ./build directory, converting 100 grayscale images
# (e.g.: <PATH>/raw/left/00087.pgm) to 700x500
./build/img_inspector 700 500 0 100 0 grayscale "/home/galar/Documents/morocco_SFR/merzouga/merzouga-minnie-trajectory21-1/front_cam/"\
    "raw/left/" "raw/right/" "" "" "%s%05d.pgm" "frontcam-calibration.yaml"\
    "/home/galar/Documents/morocco_SFR/merzouga/merzouga-minnie-trajectory21-1/front_cam/threaded/" "left/" "right/" "" "" "%s%05d.pgm"\
    "/home/galar/Documents/morocco_SFR/merzouga/merzouga-minnie-trajectory21-1/front_cam/threaded/" "updated_calibration_700x500.yaml"

# Example call of the same executable with all images in the SAME directory, but different names for left VS right, also different output naming style is used, mix and match according to what you need
# (e.g.: <COMMON_PATH>/IMG_leftcam_087_stereo.jpg)
#./build/img_inspector 700 500 0 100 0 grayscale "/home/galar/Documents/morocco_SFR/merzouga/merzouga-minnie-trajectory21-1/front_cam/"\
#    "" "" "leftcam" "rightcam" "IMG_%s_%03d_stereo.jpg" "frontcam-calibration.yaml"\
#    "/home/galar/Documents/morocco_SFR/merzouga/merzouga-minnie-trajectory21-1/front_cam/threaded/" "" "" "left" "right" "%s-%03d.jpg"\
#    "/home/galar/Documents/morocco_SFR/merzouga/merzouga-minnie-trajectory21-1/front_cam/threaded/" "updated_calibration_700x500.yaml"


