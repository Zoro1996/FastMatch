#pragma once
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include <fstream>
#include <string>
#include <vector>
#include<iostream>
#include <iterator>
#include "MyFunction.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


static constexpr const float PI = 3.14;
static constexpr const int scale = 4;
static constexpr const float sampleRate = 0.001;

struct Contours
{
	/*轮廓图*/
	Mat dstImage;

	/*质心坐标*/
	int centerX;
	int centerY;

	/*点集形式*/
	vector<Point2f> PointSet;
};

struct TransVariable
{
	int tx;
	int ty;
	float theta;
};
struct CurrentBestReusult
{
	tuple<int, int, float> currentBestTrans;
	float currentBestDistance;
};


Point2f AfterTrans(tuple<int, int, float>&curTrans, Point2f curPoint, float centerX, float centerY);

float SingleTransEvaluation(Contours& maskStruct, Mat &srcImage, vector<Point2f> &subMaskPiontSet,
	tuple<int, int, float>&curTrans, float epsilon);

vector<tuple<int, int, float>> ConstructNet(Mat&srcImage, float delta);

CurrentBestReusult GetBestTrans(Contours& maskStruct, Mat&srcImage, vector<Point2f>&subMaskPiontSet,
	vector<tuple<int, int, float>>&TransNet, float delta, float epsilon);

vector <tuple<int, int, float >> GetNextNet(Mat&srcImage, vector <tuple<int, int, float >> &GoodTransNet,
	vector<Point2f>&subMaskPointSet, float centerX, float centerY, float delta);

tuple<int, int, float> FastMatch(Contours &maskStruct, Mat &srcImage, float delta, float epsilon, float factor);