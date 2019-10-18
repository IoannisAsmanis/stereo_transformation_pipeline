#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// Functionality definitions
#define ORIG_WIDTH 1280
#define ORIG_HEIGHT 960
#define TAR_WIDTH 512
#define TAR_HEIGHT 512

#define SYMMETRIC_CROPPING 0
#define ASYMMETRIC_CROPPING 1
#define CROPPING_POLICY SYMMETRIC_CROPPING     // Change according to your logic

// File opening definitions
#define DATA_ROOT "/home/galar/Documents/morocco_SFR/merzouga/merzouga-minnie-trajectory21-1/front_cam/"
#define LEFT_IMG_PATH "raw/left/"
#define RIGHT_IMG_PATH "raw/right/"
#define PIC_FILE_IDX_LIM 0
#define IMG_FILE_FORMAT "%05d.pgm"
#define CALIB_FILE "frontcam-calibration.yaml"
#define IMG_DIR_START_IDX 2
#define IMG_DIR_END_IDX 2


// Utilities
#define MAX_STR_LEN 128

// Display
#define TITLE_ORIG_FORMAT "Original frame #%d"
#define TITLE_PROC_FORMAT "Processed frame #%d"

// Debugging
#define DEBUG

using namespace std;
using namespace cv;


/**
 * This function obtains a cv::Mat representation of an image file from disk
 * */
Mat getImgMat(int index, bool isLeft=true, bool isGrayscale=true)
{
    char fname[MAX_STR_LEN], file_template[MAX_STR_LEN];
    strcpy(file_template, DATA_ROOT);
    strcat(file_template, ((isLeft) ? LEFT_IMG_PATH : RIGHT_IMG_PATH));
    strcat(file_template, IMG_FILE_FORMAT);
    sprintf(fname, file_template, index);

    Mat res = imread(fname, ((isGrayscale)?IMREAD_GRAYSCALE:IMREAD_COLOR));
    if (!res.data) {
        cerr << "Error loading image file: "
            << fname
            << endl;
        exit(1);
    }

#ifdef DEBUG
    cout << fname << endl;
#endif

    return res;
}


/**
 * This function finds a cropping/scaling strategy to accomodate the requested resolutions
 * and then implements it, returning the updated camera parameters
 * */
//vector<double> autofix(int index)
//{
//    if (ORIG_WIDTH < TAR_WIDTH || ORIG_HEIGHT < TAR_HEIGHT) {
//        cerr << "Dimensions requested imply image enlargment, aborting" << endl;
//        exit(1);
//    }
//
//    Mat leftImg = getImgMat(index);
//    Mat rightImg = getImgMat(index, true);

//#if CROPPING_POLICY == NO_CROPPING
    
//#endif

    // If aspect ratios don't match, consider cropping if it is allowed
    // Compare aspect ratios using integers, doubles may have rounding errors
    //bool needs_crop = (ORIG_WIDTH * TAR_HEIGHT != TAR_WIDTH * ORIG_HEIGHT);

//}

void getFinalMatrices(Mat& p1, Mat&p2, Mat& out_left_map1, Mat& out_left_map2, Mat& out_right_map1, Mat& out_right_map2)
{
    // Read original calibration file
    char calib_fname[MAX_STR_LEN];
    strcpy(calib_fname, DATA_ROOT);
    strcat(calib_fname, CALIB_FILE);
    FileStorage fs(calib_fname, FileStorage::READ);
    Mat c1, c2, d1, d2, t, r;
    fs["camera_matrix_1"] >> c1;
    fs["camera_matrix_2"] >> c2;
    fs["distortion_coefficients_1"] >> d1;
    fs["distortion_coefficients_2"] >> d2;
    fs["rotation_matrix"] >> r;
    fs["translation_coefficients"] >> t;

#ifdef DEBUG
    cout << c1 << endl;
    cout << c2 << endl;
    cout << d1 << endl;
    cout << d2 << endl;
    cout << r << endl;
    cout << t << endl;
#endif

    // Get new matrices after rectification and resizing
    Size final_sz(TAR_WIDTH, TAR_HEIGHT);
    Mat r1, r2, q;
    stereoRectify(
            c1, d1, c2, d2, Size(ORIG_WIDTH, ORIG_HEIGHT), r, t, // inputs
            r1, r2, p1, p2, q,   // outputs
            CALIB_ZERO_DISPARITY, -1, final_sz   // defaults
    );

    initUndistortRectifyMap(
            c1, d1, r1, p1, final_sz, CV_32FC1, //inputs
            out_left_map1, out_left_map2  //outputs
    );
    initUndistortRectifyMap(
            c2, d2, r2, p2, final_sz, CV_32FC1, //inputs
            out_right_map1, out_right_map2  //outputs
    );

}

/**
 * Create a window and display an image in it, then wait for user action
 * */
void displayImage(Mat input_img, int index, bool isOriginal=true)
{
    char win_name[MAX_STR_LEN];
    sprintf(win_name, ((isOriginal) ? TITLE_ORIG_FORMAT : TITLE_PROC_FORMAT), index);
    namedWindow(win_name, CV_WINDOW_AUTOSIZE);
    imshow(win_name, input_img);
    waitKey();
}


void fixSinglePair(int idx, Mat lm1, Mat lm2, Mat rm1, Mat rm2)
{
    Mat left_img = getImgMat(idx), right_img = getImgMat(idx, false);
    Mat left_new, right_new;
    remap(left_img, left_new, lm1, lm2, INTER_LINEAR);
    remap(right_img, right_new, rm1, rm2, INTER_LINEAR);

    displayImage(left_img, idx);
    displayImage(left_new, idx, false);

    //TODO write new images to files
}


void loopOverImageDirectory(Mat left_map_1, Mat left_map_2, Mat right_map_1, Mat right_map_2)
{
    for (int i=IMG_DIR_START_IDX; i<=IMG_DIR_END_IDX; i++) {
        fixSinglePair(i, left_map_1, left_map_2, right_map_1, right_map_2);
    }
}


/**
 * Driver function for the above utilities
 * */
int main(int argc, char **argv)
{
    Mat p1, p2, lm1, lm2, rm1, rm2;
    getFinalMatrices(p1, p2, lm1, lm2, rm1, rm2);

#ifdef DEBUG
    cout << p1 << endl;
    cout << p2 << endl;
#endif

    loopOverImageDirectory(lm1, lm2, rm1, rm2);

    return 0;
}
