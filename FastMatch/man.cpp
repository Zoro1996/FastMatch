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



Contours GetMaskImage(Mat &maskImage)
{
	int centerX = 0;
	int centerY = 0;
	double size = 0.01;
	Contours maskStruct;
	Mat cannyOutput;
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	vector<Point> PointSet;

	Canny(maskImage, cannyOutput, 150, 255, 3, false);

	findContours(cannyOutput, contours, hierachy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

	Mat dstImg = Mat::zeros(maskImage.size(), CV_8UC1);
	//»­ÂÖÀª
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		double length = arcLength(contours[i], true);
		if (area > 100 && length > 100)
		{
			drawContours(dstImg, contours, i, Scalar(255), 1, 8, hierachy, 1, Point(0, 0));
			for (int j = 0; j < contours[i].size(); j++)
			{
				PointSet.push_back(contours[i][j]);
				centerX += contours[i][j].x;
				centerY += contours[i][j].y;
				size++;
			}
		}
	}
	centerX /= size;
	centerY /= size;

	maskStruct.dstImage = dstImg;
	maskStruct.centerX = centerX;
	maskStruct.centerY = centerY;
	maskStruct.PointSet = PointSet;

	return maskStruct;
}


int main(int argc, char *argv[])
{
	int index = 2;
	char  srcImagePath[100], maskImagePath[100];

	sprintf_s(srcImagePath, "E:\\dataset\\0901\\02\\%d.bmp", index);
	sprintf_s(maskImagePath, "E:\\dataset\\0901\\01\\%d.bmp", index);

	Mat srcImage = imread(srcImagePath, 0);
	Mat srcImageRGB = imread(srcImagePath, 1);
	Mat maskImage = imread(maskImagePath, 0);

	int row = int(srcImage.rows / scale);
	int col = int(srcImage.cols / scale);
	resize(srcImage, srcImage, Size(col, row));
	resize(srcImageRGB, srcImageRGB, Size(col, row));
	resize(maskImage, maskImage, Size(col, row));

	Contours maskStruct = GetMaskImage(maskImage);

	if (srcImage.empty())
	{
		cout << "load image failed !" << endl;
	}

	/*Ä£°åÆ¥Åä¿ìËÙËÑË÷Ëã·¨*/
	//Normalize
	srcImage /= 255;
	maskImage /= 255;
	double delta = 0.2;
	double epsilon = 1;
	double factor = 0.5;
	tuple<int, int, double> bestTrans = FastMatch(maskStruct, srcImage, delta, epsilon, factor);
	//tuple<int, int, double> bestTrans{ -54,-42,-0.54 };
	int transX = get<0>(bestTrans);
	int transY = get<1>(bestTrans);
	double theta = get<2>(bestTrans);

	srcImage *= 255;
	maskImage *= 255;
	double centerX = maskStruct.centerX;
	double centerY = maskStruct.centerY;


	vector<Point>maskPointSet = maskStruct.PointSet;
	for (int i = 0; i < maskPointSet.size(); i++)
	{
		int x = (int)(cos(theta)*(maskPointSet[i].x - centerX) - sin(theta)*(maskPointSet[i].y - centerY)) + centerX + transX;
		int y = (int)(sin(theta)*(maskPointSet[i].x - centerX) + cos(theta)*(maskPointSet[i].y - centerY)) + centerY + transY;
		if (x >= 0 && x < srcImage.cols && y >= 0 && y < srcImage.rows)
		{
			srcImageRGB.at<Vec3b>(y, x)[0] = 0;
			srcImageRGB.at<Vec3b>(y, x)[1] = 0;
			srcImageRGB.at<Vec3b>(y, x)[2] = 255;
		}
	}

	waitKey(0);
	return 0;
}