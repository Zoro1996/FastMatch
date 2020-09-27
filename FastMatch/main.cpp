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
	float size = 0.01;
	Contours maskStruct;
	Mat cannyOutput;
	vector<vector<Point2f>> contours;
	vector<Vec4f> hierachy;
	vector<Point2f> PointSet;

	Canny(maskImage, cannyOutput, 150, 255, 3, false);

	findContours(cannyOutput, contours, hierachy, RETR_TREE, CHAIN_APPROX_NONE, Point2f(0, 0));
	//cvtColor(maskImage, maskImage, CV_GRAY2BGR);
	Mat dstImg = Mat::zeros(maskImage.size(), CV_8UC1);
	//»­ÂÖÀª
	for (int i = 0; i < contours.size(); i++)
	{
		float area = contourArea(contours[i]);
		float length = arcLength(contours[i], true);
		if (area > 100 && length > 100)
		{
			drawContours(dstImg, contours, i, Scalar(255), -1, 8, hierachy, 1, Point(0, 0));
			for (int j = 0; j < contours[i].size(); j++)
			{
				PointSet.push_back(contours[i][j]);
				centerX += contours[i][j].x;
				centerY += contours[i][j].y;
				size++;
			}
		}
	}

	//for (int x = 0; x < dstImg.cols; x++)
	//{
	//	for (int y = 0; y < dstImg.rows; y++)
	//	{
	//		if (dstImg.at<uchar>(y,x)!=0)
	//		{
	//			maskImage.at<Vec3b>(y,x)[0] = 0;
	//			maskImage.at<Vec3b>(y,x)[1] = 0;
	//			maskImage.at<Vec3b>(y,x)[2] = 255;
	//		}
	//	}
	//}
	centerX /= size;
	centerY /= size;

	maskStruct.dstImage = dstImg;
	maskStruct.centerX = centerX;
	maskStruct.centerY = centerY;
	maskStruct.PointSet = PointSet;

	return maskStruct;
}



Contours ExitedGetMaskImage(Mat &maskImage)
{
	int centerX = 0;
	int centerY = 0;
	vector<Point2f> PointSet;
	Contours maskStruct;

	float size = 0.01;

	for (int row = 0; row < maskImage.rows; row++)
	{
		for (int col = 0; col < maskImage.cols; col++)
		{
			if (maskImage.at<float>(row, col)!= 0)
			{
				centerX += col;
				centerY += row;
				PointSet.push_back(Point2f(col, row));

				size++;
			}
		}
	}

	centerX /= size;
	centerY /= size;

	maskStruct.dstImage = maskImage;
	maskStruct.centerX = centerX;
	maskStruct.centerY = centerY;
	maskStruct.PointSet = PointSet;

	return maskStruct;
}




int main(int argc, char *argv[])
{
	int index = 1;
	char  srcPath[100], maskPath[100];

	sprintf_s(srcPath, "E:\\dataset\\0901\\02\\%d.bmp", index);
	sprintf_s(maskPath, "E:\\dataset\\0901\\01\\mask_4.bmp");

	Mat src = imread(srcPath, 0);
	Mat srcRGB = imread(srcPath, 1);
	Mat mask = imread(maskPath, 0);
	Mat maskRGB = imread(maskPath, 1);

	Mat srcImage = src.clone();
	Mat maskImage = mask.clone();
	Mat srcImageRGB;

	/*Ëõ·Å*/
	int row = int(srcImage.rows / scale);
	int col = int(srcImage.cols / scale);
	resize(srcImage, srcImage, Size(col, row));
	resize(srcRGB, srcImageRGB, Size(col, row));
	resize(maskImage, maskImage, Size(col, row));

	//Contours maskStruct = GetMaskImage(maskImage);
	Contours maskStruct = ExitedGetMaskImage(maskImage);

	if (srcImage.empty())
	{
		cout << "load image failed !" << endl;
	}

	/*Ä£°åÆ¥Åä¿ìËÙËÑË÷Ëã·¨*/
	//Normalize
	srcImage.convertTo(srcImage, CV_32FC1);
	//maskImage.convertTo(maskImage, CV_32FC1);
	srcImage /= 255;
	maskImage /= 255;
	float delta = 0.2;
	float epsilon = 1;
	float factor = 0.5;
	tuple<int, int, float> bestTrans = FastMatch(maskStruct, srcImage, delta, epsilon, factor);

	//tuple<int, int, float> bestTrans{ -54,-42,-0.54 };
	//tuple<int, int, float> bestTrans{ 70,57,-2.79 };
	int transX = get<0>(bestTrans);
	int transY = get<1>(bestTrans);
	float theta = get<2>(bestTrans);

	//srcImage *= 255;
	//maskImage *= 255;
	float centerX = maskStruct.centerX;
	float centerY = maskStruct.centerY;


	vector<Point2f>maskPointSet = maskStruct.PointSet;
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

	for (int i = 0; i < maskPointSet.size(); i++)
	{
		int x = (int)(maskPointSet[i].x * 8);
		int y = (int)(maskPointSet[i].y * 8);
		maskRGB.at<Vec3b>(y, x)[0] = 0;
		maskRGB.at<Vec3b>(y, x)[1] = 0;
		maskRGB.at<Vec3b>(y, x)[2] = 255;
	}

	waitKey(0);
	return 0;
}