#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <thread>
#include <ctime>

#include <opencv2/opencv.hpp>

// CLI (2+3+1+6+1+8)
#define NUM_COMMAND_LINE_ARGS 21

// Utility
#define MAX_STR_LEN 512

// Display
#define TITLE_ORIG_FORMAT "Original frame #%d"
#define TITLE_PROC_FORMAT "Processed frame #%d"

// Calibration structure definitions
#define CALIB_IN_IMG_WIDTH "image_width"
#define CALIB_IN_IMG_HEIGHT "image_height"
#define CALIB_IN_CAMERA_MAT_LEFT "camera_matrix_1"
#define CALIB_IN_CAMERA_MAT_RIGHT "camera_matrix_2"
#define CALIB_IN_DIST_COEFFS_LEFT "distortion_coefficients_1"
#define CALIB_IN_DIST_COEFFS_RIGHT "distortion_coefficients_2"
#define CALIB_IN_ROT_MAT "rotation_matrix"
#define CALIB_IN_TRANS_COEFFS "translation_coefficients"

// Uncomment for debuggin output
//#define DEBUG


/* Global variables to simplify function signatures */
// NOTES:
// - Most of the global variables used in this program
// need to be supplied as command-line arguments. Remember
// that they ALL need to be present, including empty strings.
// - Also remember that system path variables must always end
// in a '/' character!
// - Finally, images can be named with any convention of the form:
// [optional characters]<STRING_MARK(possibly empty)>[optional characters]<FORMATTED_INTEGER_INDEX>[optional characters + file suffix]
// If you need to switch to index-before-string convention, then
// search through the code for calls to sprintf() and fix the ordering
// of its arguments where applicable.

// Below are the variables, with (x) indicating how many
// per block are found from CL args

// Functionality variables
int
    // Image dimensions (2)
    ORIG_WIDTH, // auto-detected
    ORIG_HEIGHT,    // auto-detected
    TAR_WIDTH,
    TAR_HEIGHT,

    // Directory indeces (3)
    IMG_DIR_START_IDX,     // inclusive
    IMG_DIR_END_IDX,    // inclusive
    OUTPUT_DIR_START_IDX;   // inclusive

// Type variables (1)
bool IMG_GRAYSCALE;

// File opening variables
char
    // Input (6)
    DATA_ROOT[MAX_STR_LEN],
    LEFT_IMG_PATH[MAX_STR_LEN],
    RIGHT_IMG_PATH[MAX_STR_LEN],
    LEFT_MARK[MAX_STR_LEN],
    RIGHT_MARK[MAX_STR_LEN],
    IMG_FILE_FORMAT[MAX_STR_LEN],

    // Calibration (1)
    CALIB_FILE[MAX_STR_LEN],

    // Output (8)
    OUTPUT_IMG_DIR[MAX_STR_LEN],
    OUTPUT_LEFT_IMG_PATH[MAX_STR_LEN],
    OUTPUT_RIGHT_IMG_PATH[MAX_STR_LEN],
    OUTPUT_LEFT_MARK[MAX_STR_LEN],
    OUTPUT_RIGHT_MARK[MAX_STR_LEN],
    OUTPUT_IMG_FILE_FORMAT[MAX_STR_LEN],
    OUTPUT_CALIB_DIR[MAX_STR_LEN],
    OUTPUT_CALIB_FILE[MAX_STR_LEN];


using namespace std;
using namespace cv;


/**
 * This function obtains a cv::Mat representation of an image file from disk
 * */
Mat getImgMat(int index, bool isLeft=true)
{
    char fname[MAX_STR_LEN], file_template[MAX_STR_LEN];
    strcpy(file_template, DATA_ROOT);
    strcat(file_template, ((isLeft) ? LEFT_IMG_PATH : RIGHT_IMG_PATH));
    strcat(file_template, IMG_FILE_FORMAT);
    sprintf(fname, file_template, ((isLeft) ? LEFT_MARK : RIGHT_MARK), index);

    Mat res = imread(fname, ((IMG_GRAYSCALE) ? IMREAD_GRAYSCALE : IMREAD_COLOR));
    if (!res.data) {
        cerr << "Error loading image file: "
            << fname
            << endl;
        exit(1);
    }

    ORIG_WIDTH = res.cols;
    ORIG_HEIGHT = res.rows;

#ifdef DEBUG
    cout << fname << endl;
#endif

    return res;
}


/**
 * Returns a reasonable size by minimally editing the size of the original
 * image to match the target aspect ratio
 * */
Size getIntermediateSize()
{
	double aspect_ratio_orig = ((double) ORIG_WIDTH) / ORIG_HEIGHT;
	double aspect_ratio_tar = ((double) TAR_WIDTH) / TAR_HEIGHT;

	double width = ORIG_WIDTH, height = ORIG_HEIGHT;

	// Only edit one dimension, e.g. if we're narrowing an image, only
	// width needs to change
	if (aspect_ratio_orig > aspect_ratio_tar) {
		width = height * aspect_ratio_tar;
	} else {
		height = width / aspect_ratio_tar;
	}

#ifdef DEBUG
    cout << "--- INTERM SZ ---" << endl;
    cout << cvRound(width) << ", " << cvRound(height) << endl;
#endif

	return Size(cvRound(width), cvRound(height));
}


/**
 * Computes matrices for the image transformations
 * */
void getFinalMatrices(Mat& p1, Mat&p2, Mat& out_left_map1, Mat& out_left_map2, Mat& out_right_map1, Mat& out_right_map2, Size& intermediate_sz)
{
    // Read original calibration file
    char calib_fname[MAX_STR_LEN];
    strcpy(calib_fname, DATA_ROOT);
    strcat(calib_fname, CALIB_FILE);
    FileStorage fs(calib_fname, FileStorage::READ);
    Mat c1, c2, d1, d2, t, r;
    fs[CALIB_IN_CAMERA_MAT_LEFT] >> c1;
    fs[CALIB_IN_CAMERA_MAT_RIGHT] >> c2;
    fs[CALIB_IN_DIST_COEFFS_LEFT] >> d1;
    fs[CALIB_IN_DIST_COEFFS_RIGHT] >> d2;
    fs[CALIB_IN_ROT_MAT] >> r;
    fs[CALIB_IN_TRANS_COEFFS] >> t;

#ifdef DEBUG
    cout << "--- INPUT FROM CALIB FILE ---" << endl;
    cout << c1 << endl;
    cout << c2 << endl;
    cout << d1 << endl;
    cout << d2 << endl;
    cout << r << endl;
    cout << t << endl;
#endif

    // Get matrices and maps for rectification and aspect ratio correction
	intermediate_sz = getIntermediateSize();
    Mat r1, r2, q;
    stereoRectify(
            c1, d1, c2, d2, Size(ORIG_WIDTH, ORIG_HEIGHT), r, t, // inputs
            r1, r2, p1, p2, q,   // outputs
            CALIB_ZERO_DISPARITY, 0, intermediate_sz   // defaults -> use alpha=0 to avoid black edges in the output image
    );

    initUndistortRectifyMap(
            c1, d1, r1, p1, intermediate_sz, CV_32FC1, //inputs
            out_left_map1, out_left_map2  //outputs
    );
    initUndistortRectifyMap(
            c2, d2, r2, p2, intermediate_sz, CV_32FC1, //inputs
            out_right_map1, out_right_map2  //outputs
    );

}

/**
 * Create a window and display an image in it, then optionally wait for user action
 * */
void displayImage(Mat input_img, int index, bool isOriginal, bool waitAfterDisplay=true)
{
    char win_name[MAX_STR_LEN];
    sprintf(win_name, ((isOriginal) ? TITLE_ORIG_FORMAT : TITLE_PROC_FORMAT), index);
    namedWindow(win_name, CV_WINDOW_AUTOSIZE);
    imshow(win_name, input_img);
    if (waitAfterDisplay) {
		waitKey();
	}
}


/**
 * Simple utility that constructs a file path and prints an image to it
 * */
void printImgToFile(Mat img, int idx_in, bool isLeft)
{
	// Calculate the desired output index
	int idx_out = idx_in - IMG_DIR_START_IDX + OUTPUT_DIR_START_IDX;

	// Create the file name using macros
	char of_template[MAX_STR_LEN], out_file[MAX_STR_LEN];
	strcpy(of_template, OUTPUT_IMG_DIR);
	strcat(of_template, ((isLeft) ? OUTPUT_LEFT_IMG_PATH : OUTPUT_RIGHT_IMG_PATH));
	strcat(of_template, OUTPUT_IMG_FILE_FORMAT);
	sprintf(out_file, of_template, ((isLeft) ? OUTPUT_LEFT_MARK : OUTPUT_RIGHT_MARK), idx_out);
	
#ifdef DEBUG
	cout << out_file << endl;
#endif

	// Write the image to disk
	imwrite(out_file, img);
}


/**
 * Function that accepts an index and the necessary matrices to
 * transform a left-right stereo image pair to the appropriate
 * format and store the results in the corresponding output
 * image files
 * */
void fixSinglePair(int idx, Mat lm1, Mat lm2, Mat rm1, Mat rm2)
{
	// Rectify images and simultaneously fix their aspect ratios
    Mat left_img = getImgMat(idx), right_img = getImgMat(idx, false);
    Mat left_new, right_new;
    remap(left_img, left_new, lm1, lm2, INTER_LINEAR);
    remap(right_img, right_new, rm1, rm2, INTER_LINEAR);

	// Resize them to their final size, using INTER_AREA interpolation to avoid Moire effects
    // when sizing down, INTER_CUBIC for better performance when sizing up
    bool zooming_in = (left_new.rows < TAR_HEIGHT);
	Mat left_final, right_final;
	resize(left_new, left_final, Size(TAR_WIDTH, TAR_HEIGHT), 0, 0,
            (zooming_in ? INTER_CUBIC : INTER_AREA));
	resize(right_new, right_final, Size(TAR_WIDTH, TAR_HEIGHT), 0, 0,
            (zooming_in ? INTER_CUBIC : INTER_AREA));

#ifdef DEBUG
    //displayImage(left_img, idx, true, false);
    //displayImage(left_final, idx, false);
#endif

    // Write images to a file
	printImgToFile(left_final, idx, true);
	printImgToFile(right_final, idx, false);
}


/**
 * This function loops over the image directories and fixes the stereo pairs
 * one at a time, storing the results in the corresponding output directories.
 * */
void loopOverImageDirectory(Mat left_map_1, Mat left_map_2, Mat right_map_1, Mat right_map_2, int range_start, int range_end)
{
    for (int i=range_start; i<=range_end; i++) {
        fixSinglePair(i, left_map_1, left_map_2, right_map_1, right_map_2);
    }
}


/**
 * This function stores the resulting calibration values to a .yaml
 * calibration file
 * */
void storeJointCalibrationFile(Mat pmat_with_baseline, Size interm_size)
{
	// First calculate the new calibration data
	// Start by computing the scaling factors for final resizing
	double xfactor = ((double) TAR_WIDTH) / interm_size.width,
		yfactor = ((double) TAR_HEIGHT) / interm_size.height;
	Mat final_K;
	pmat_with_baseline(Rect(0,0, 3,3)).copyTo(final_K);
	for (int j=0; j<3; j++) {
		final_K.at<double>(0, j) *= xfactor;
		final_K.at<double>(1, j) *= yfactor;
	}

	// Then use the rectification property to find
	// the equivalent baseline translation
	Mat dist_coeffs = Mat::zeros(Size(5, 1), CV_64FC1);
	Mat rot_mat = Mat::zeros(Size(3, 3), CV_64FC1);
	Mat trans_coeffs = Mat::zeros(Size(1, 3), CV_64FC1);
	double fx = pmat_with_baseline.at<double>(0, 0),
		dx = pmat_with_baseline.at<double>(0, 3);
	trans_coeffs.at<double>(0, 0) = dx / fx;

	// Finally store everything in the new calib file
	char calib_out[MAX_STR_LEN];
	strcpy(calib_out, OUTPUT_CALIB_DIR);
	strcat(calib_out, OUTPUT_CALIB_FILE);
    FileStorage fs(calib_out, FileStorage::WRITE);
	fs << CALIB_IN_IMG_WIDTH << TAR_WIDTH
		<< CALIB_IN_IMG_HEIGHT << TAR_HEIGHT
		<< CALIB_IN_CAMERA_MAT_LEFT << final_K
		<< CALIB_IN_DIST_COEFFS_LEFT << dist_coeffs
		<< CALIB_IN_CAMERA_MAT_RIGHT << final_K
		<< CALIB_IN_DIST_COEFFS_RIGHT << dist_coeffs
		<< CALIB_IN_ROT_MAT << rot_mat
		<< CALIB_IN_TRANS_COEFFS << trans_coeffs;
}


/**
 * Get a HW-dependent estimation on the optimal number of threads
 * to spawn on your system.
 * */
int getThreadCount()
{
#ifdef MULTI_THREADING
    unsigned result = std::thread::hardware_concurrency()/2;
    return (result == 0 ? 1 : result);
#else
    return 1;
#endif
}


/**
 * Get a vector containing the start indeces for each span of
 * image data to be fed to the worker threads of this program.
 * */
vector<int> getRangeOrigins(int nRanges)
{
    int start = IMG_DIR_START_IDX, end = IMG_DIR_END_IDX;
    int width = (end-start+1)/nRanges;
    vector<int> res;
    for (register unsigned i=0; i<nRanges; i++) {
        res.push_back(i*width+start);
    }
    return res;
}


/**
 * Driver function for the above utilities
 * */
int main(int argc, char **argv)
{

    if (argc != NUM_COMMAND_LINE_ARGS+1) {
        cerr << "Unexpected number of arguments, aborting!" << endl;
        exit(1);
    } else {
        unsigned arg_idx=1;
        TAR_WIDTH = atoi(argv[arg_idx++]);
        TAR_HEIGHT = atoi(argv[arg_idx++]);
        IMG_DIR_START_IDX = atoi(argv[arg_idx++]);
        IMG_DIR_END_IDX = atoi(argv[arg_idx++]);
        OUTPUT_DIR_START_IDX = atoi(argv[arg_idx++]);

        IMG_GRAYSCALE = !(strcmp(argv[arg_idx++], "grayscale"));

        strcpy(DATA_ROOT, argv[arg_idx++]);
        strcpy(LEFT_IMG_PATH, argv[arg_idx++]);
        strcpy(RIGHT_IMG_PATH, argv[arg_idx++]);
        strcpy(LEFT_MARK, argv[arg_idx++]);
        strcpy(RIGHT_MARK, argv[arg_idx++]);
        strcpy(IMG_FILE_FORMAT, argv[arg_idx++]);

        strcpy(CALIB_FILE, argv[arg_idx++]);

        strcpy(OUTPUT_IMG_DIR, argv[arg_idx++]);
        strcpy(OUTPUT_LEFT_IMG_PATH, argv[arg_idx++]);
        strcpy(OUTPUT_RIGHT_IMG_PATH, argv[arg_idx++]);
        strcpy(OUTPUT_LEFT_MARK, argv[arg_idx++]);
        strcpy(OUTPUT_RIGHT_MARK, argv[arg_idx++]);
        strcpy(OUTPUT_IMG_FILE_FORMAT, argv[arg_idx++]);
        strcpy(OUTPUT_CALIB_DIR, argv[arg_idx++]);
        strcpy(OUTPUT_CALIB_FILE, argv[arg_idx++]);
    }


	// p=projection matrices, lm=left image maps, rm=right image maps
    Mat p1, p2, lm1, lm2, rm1, rm2, query;
    // Use query to dynamically determine the dimensions of the input images
    query = getImgMat(IMG_DIR_START_IDX, true);

    // Compute maps and the intermediate size
	Size isz;
    getFinalMatrices(p1, p2, lm1, lm2, rm1, rm2, isz);

#ifdef DEBUG
    cout << "--- INTERM PMATS ---" << endl;
    cout << p1 << endl;
    cout << p2 << endl;
#endif

	// After the above process, the intrinsics for the matrices are identical
	// Also no intra-camera-frame rotation exists, only translation, and the
	// translation is confined only to the x axis due to rectification.
	storeJointCalibrationFile(p2, isz);

    // The implementation uses multithreading from here on
    // If you compiled OpenCV with parallelization support,
    // OpenCV tasks used here are already running in parallel
    // The additional parallelization introduced here helps
    // reduce disk I/O overheads, resulting in up to 20% acceleration
    // for a 4-core hyperthreaded i7
    int n_threads = getThreadCount();
    clock_t t;
    t = clock();

    if (n_threads <= 1 || IMG_DIR_END_IDX - IMG_DIR_START_IDX < 2*n_threads) {
        cout << "Executing single-thread solution... ";
        loopOverImageDirectory(lm1, lm2, rm1, rm2,
                IMG_DIR_START_IDX, IMG_DIR_END_IDX);
        cout << "DONE" << endl;
    } else {

        cout << "Multithreading factor: " << n_threads << "... ";

        vector<int> sectors = getRangeOrigins(n_threads);
        int range = sectors[1]-sectors[0];
        thread *ths[n_threads];

        for (register unsigned i=0; i<n_threads; i++) {
            ths[i] = new thread(loopOverImageDirectory, lm1, lm2, rm1, rm2,
                    sectors[i], ((i == n_threads-1) ? IMG_DIR_END_IDX : sectors[i]+range-1));
        }

        for (register unsigned i=0; i<n_threads; i++) {
            ths[i]->join();
#ifdef DEBUG
            cout << "Joining thread #" << i << endl;
#endif
            delete ths[i];
        }
        cout << "DONE" << endl;
    }

    t = clock()-t;
    cout << "Total execution time: " << ((float) t) / CLOCKS_PER_SEC << "s" << endl;

    return 0;
}
