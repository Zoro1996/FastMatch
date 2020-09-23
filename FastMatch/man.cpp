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



Contours GetMaskImage(char *maskImagePath)
{
	int centerX = 0;
	int centerY = 0;
	double size = 0.01;
	Contours maskStruct;
	Mat cannyOutput;
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	vector<Point> PointSet;

	Mat maskImage = imread(maskImagePath, 0);
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
	int index = 1;
	char  srcImagePath[100], maskImagePath[100];

	sprintf_s(srcImagePath, "E:\\dataset\\0901\\02\\%d.bmp", index);
	sprintf_s(maskImagePath, "E:\\dataset\\0901\\01\\%d.bmp", index);

	Mat srcImage = imread(srcImagePath, 0);
	Mat srcImageRGB = imread(srcImagePath, 1);

	Contours maskStruct = GetMaskImage(maskImagePath);
	Mat maskImage = maskStruct.dstImage;

	if (srcImage.empty())
	{
		cout << "load image failed !" << endl;
	}

	/*Ä£°åÆ¥Åä¿ìËÙËÑË÷Ëã·¨*/
	//Normalize
	srcImage /= 255;
	maskImage /= 255;
	double delta = 0.2;
	double epsilon = 0.5;
	double factor = 0.9;
	tuple<int, int, double> bestTrans = FastMatch(maskStruct, srcImage, delta, epsilon, factor);

	int transX = get<0>(bestTrans);
	int transY = get<1>(bestTrans);
	double theta = get<2>(bestTrans);

	srcImage *= 255;
	maskImage *= 255;
	double centerX = 0;
	double centerY = 0;
	vector<Point2d> maskPoint;
	for (int x = 0; x < maskImage.cols; x++)
	{
		for (int y = 0; y < maskImage.rows; y++)
		{
			if (maskImage.at<uchar>(y, x) != 0)
			{
				maskPoint.push_back(Point2d(x, y));
				centerX += x;
				centerY += y;
			}
		}
	}
	centerX /= maskPoint.size();
	centerY /= maskPoint.size();

	for (int i = 0; i < maskPoint.size(); i++)
	{
		int x = (int)(centerX + cos(theta)*(maskPoint[i].x - centerX) - sin(theta)*(maskPoint[i].y - centerY) + transX);
		int y = (int)(centerY + sin(theta)*(maskPoint[i].x - centerX) + cos(theta)*(maskPoint[i].y - centerY) + transY);
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