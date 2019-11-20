#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

// Definitions for reading calib file
#define CALIB_IN_IMG_WIDTH "image_width"
#define CALIB_IN_IMG_HEIGHT "image_height"
#define CALIB_IN_CAMERA_MAT_LEFT "camera_matrix_1"
#define CALIB_IN_CAMERA_MAT_RIGHT "camera_matrix_2"


// Utility definitions
#define MAX_STR_LEN 512

// Uncomment for debugging
//#define DEBUG

using namespace std;
using namespace cv;

/**
 * This function simply converts degrees to radians
 * */
inline double deg2rad (double degrees) {
    static const double pi_on_180 = M_PI / 180.0;
    return degrees * pi_on_180;
}


/**
 * This function performs the core optics calculations and pushes the results in an output
 * vector passed by reference
 * */
void processOptics(Mat K, double apertWidth, double apertHeight, Size sz, vector<double>& result)
{
    double fovx, fovy, focalLength, aspectRatio;
    Point2d pp;
    calibrationMatrixValues(K, sz, apertWidth, apertHeight, fovx, fovy, focalLength, pp, aspectRatio);
    result.push_back(fovx);
    result.push_back(fovy);
    result.push_back(focalLength);
    result.push_back(pp.x);
    result.push_back(pp.y);
    result.push_back(aspectRatio);
}


/**
 * This function reads in a calibration file and
 * returns the results of several optics-related
 * calculations.
 * */
void calcOptics(char calibFilePath[], double apertWidth, double apertHeight, vector<double> &out_optics_left, vector<double>& out_optics_right)
{
    int width, height;
    FileStorage fs(calibFilePath, FileStorage::READ);
    fs[CALIB_IN_IMG_WIDTH] >> width;
    fs[CALIB_IN_IMG_HEIGHT] >> height;
    Size sz(width, height);

    Mat K;
    fs[CALIB_IN_CAMERA_MAT_LEFT] >> K;
    processOptics(K, apertWidth, apertHeight, sz, out_optics_left);
    fs[CALIB_IN_CAMERA_MAT_RIGHT] >> K;
    processOptics(K, apertWidth, apertHeight, sz, out_optics_right);
}


/**
 * This function simply displays the results of the container vector
 * which is the output of the OpenCV processing occurring above.
 * If the aperture dimensions were provided, more output becomes
 * available, otherwise only FoV can be computed.
 * */
void dispFoV(vector<double> container, bool apertureValid)
{
    cout << "FoV - Horizontal:\t" << container[0] << " deg" << endl;
    cout << "FoV - Vertical:\t\t" << container[1] << " deg" << endl;
    if (apertureValid) {
        cout << "Focal length:\t\t" << container[2] << " mm" << endl;
        cout << "Principal point:\t" << "(" << container[3] << ", " << container[4] << ")" << " mm" << endl;
        cout << "Aspect ratio:\t\t" << container[5] << endl;
    }
}


/**
 * Driver code and CLI
 * */
int main(int argc, char **argv)
{
    char calibFilePath[MAX_STR_LEN];
    if (argc == 1 || argc > 4) {
        cerr << "Unexpected number of arguments, aborting!" << endl;
        exit(1);
    }
    strcpy(calibFilePath, argv[1]);

    bool apertValid=false;
    double apertWidth=0.0, apertHeight=0.0;
    if (argc > 2) {
        apertValid = true;
        apertWidth = atof(argv[2]);
        apertHeight = atof(argv[3]);
    }

#ifdef DEBUG
    cout << calibFilePath << endl;
    cout << apertWidth << endl;
    cout << apertHeight << endl;
#endif

    vector<double> optics_left, optics_right;
    calcOptics(calibFilePath, apertWidth, apertHeight, optics_left, optics_right);

    cout << "LEFT CAMERA:" << endl;
    dispFoV(optics_left, apertValid);
    cout << "RIGHT CAMERA:" << endl;
    dispFoV(optics_right, apertValid);

    return 0;
}
