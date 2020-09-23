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

static constexpr const double PI = 3.14;
struct Contours
{
	/*轮廓图*/
	Mat dstImage;

	/*质心坐标*/
	int centerX;
	int centerY;

	/*点集形式*/
	vector<Point> PointSet;
};




Point AfterTrans(tuple<int, int, double>&curTrans, Point curPoint, double centerX, double centerY);

double SingleTransEvaluation(Contours& maskStruct, Mat &srcImage, vector<Point> &subMaskPiontSet,
	tuple<int, int, double>&curTrans, double epsilon);

vector<tuple<int, int, double>> ConstructNet(Mat&srcImage, double delta);

tuple<int, int, double> GetBestTrans(Contours& maskStruct, Mat&srcImage, vector<Point>&subMaskPiontSet,
	vector<tuple<int, int, double>>&TransNet, double delta, double epsilon);

vector <tuple<int, int, double >> GetNextNet(Mat&srcImage, vector <tuple<int, int, double >> &GoodTransNet,
	vector<Point>&subMaskPointSet, double centerX, double centerY, double delta);

tuple<int, int, double> FastMatch(Contours &maskStruct, Mat &srcImage, double delta, double epsilon, double factor);