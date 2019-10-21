#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

// Definitions for given field of view (in DEGREES)
#define FOV_HOR 110
#define FOV_VERT 96

// Definitions for reading calib file
#define CALIB_IN_IMG_WIDTH "image_width"
#define CALIB_IN_IMG_HEIGHT "image_height"
#define CALIB_IN_CAMERA_MAT_LEFT "camera_matrix_1"
#define CALIB_IN_CAMERA_MAT_RIGHT "camera_matrix_2"


// Utility definitions
#define MAX_STR_LEN 128


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
 * This function reads in a calibration file and
 * returns the results of several optics-related
 * calculations.
 * */
void calcOptics(char calibFilePath[], vector<double> &out_optics_left, vector<double>& out_optics_right)
{
    
    int width, height;
    double fovx, fovy, focalLength, aspectRatio;
    Mat K;
    Point2d pp;
    FileStorage fs(calibFilePath, FileStorage::READ);
    
    fs[CALIB_IN_IMG_WIDTH] >> width;
    fs[CALIB_IN_IMG_HEIGHT] >> height;
    Size sz(width, height);
    
    fs[CALIB_IN_CAMERA_MAT_LEFT] >> K;
    calibrationMatrixValues(K, sz, 0.0, 0.0, fovx, fovy, focalLength, pp, aspectRatio);
    out_optics_left.push_back(fovx);
    out_optics_left.push_back(fovy);

    fs[CALIB_IN_CAMERA_MAT_RIGHT] >> K;
    calibrationMatrixValues(K, sz, 0.0, 0.0, fovx, fovy, focalLength, pp, aspectRatio);
    out_optics_right.push_back(fovx);
    out_optics_right.push_back(fovy);
}

void dispFoV(vector<double> container)
{
    cout << "FoV - Horizontal: " << container[0] << endl;
    cout << "FoV - Vertical: " << container[1] << endl; 
}


int main(int argc, char **argv)
{
    char calibFilePath[MAX_STR_LEN];
    strcpy(calibFilePath, argv[1]);

    cout << calibFilePath << endl;
    vector<double> optics_left, optics_right;
    calcOptics(calibFilePath, optics_left, optics_right);

    cout << "LEFT CAMERA:" << endl;
    dispFoV(optics_left);
    cout << "RIGHT CAMERA:" << endl;
    dispFoV(optics_right);
    
    return 0;
}
