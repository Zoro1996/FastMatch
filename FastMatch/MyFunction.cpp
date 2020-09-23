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


Point AfterTrans(tuple<int, int, double>&curTrans, Point curPoint, double centerX, double centerY)
{
	Point resultPoint;
	int transX = get<0>(curTrans);
	int transY = get<1>(curTrans);
	double theta = get<2>(curTrans);

	resultPoint.x = (int)(cos(theta)*(curPoint.x - centerX) - sin(theta)*(curPoint.y - centerY)) + centerX + transX;
	resultPoint.y = (int)(sin(theta)*(curPoint.x - centerX) + cos(theta)*(curPoint.y - centerY)) + centerY + transY;

	return resultPoint;
}


double SingleTransEvaluation(Contours& maskStruct, Mat &srcImage, vector<Point> &subMaskPiontSet,
	tuple<int, int, double>&curTrans, double epsilon)
{
	Mat maskmage = maskStruct.dstImage;
	double centerX = maskStruct.centerX;
	double centerY = maskStruct.centerY;

	int pixelStep = int(10/scale);
	double curPointValue;
	double curPointValue_UR1;
	double curPointValue_UR2;
	double curPointValue_UL1;
	double curPointValue_UL2;
	double curPointValue_DR1;
	double curPointValue_DR2;
	double curPointValue_DL1;
	double curPointValue_DL2;

	Point curPoint, transPoint;
	int sampleStep = (int)(1 / pow(epsilon, 2)), num = 0;
	double distance = 0, score = 0;
	for (int i = 0; i < subMaskPiontSet.size(); i += sampleStep)
	{
		num++;
		curPoint = subMaskPiontSet[i];
		transPoint = AfterTrans(curTrans, subMaskPiontSet[i],centerX,centerY);
		if (transPoint.x >= 0 && transPoint.x < srcImage.cols && transPoint.y >= 2*pixelStep && transPoint.y < srcImage.rows-2*pixelStep)
		{
			//penalty += abs(maskImage.at<uchar>(curPoint) - srcImage.at<uchar>(transPoint));
			curPointValue = srcImage.at<uchar>(transPoint.y, transPoint.x);
			curPointValue_UR1 = srcImage.at<uchar>(transPoint.y - 1 * pixelStep, transPoint.x + 1 * pixelStep);
			curPointValue_UR2 = srcImage.at<uchar>(transPoint.y - 2 * pixelStep, transPoint.x + 2 * pixelStep);
			curPointValue_UL1 = srcImage.at<uchar>(transPoint.y - 1 * pixelStep, transPoint.x - 1 * pixelStep);
			curPointValue_UL2 = srcImage.at<uchar>(transPoint.y - 2 * pixelStep, transPoint.x - 2 * pixelStep);
			curPointValue_DR1 = srcImage.at<uchar>(transPoint.y + 1 * pixelStep, transPoint.x + 1 * pixelStep);
			curPointValue_DR2 = srcImage.at<uchar>(transPoint.y + 2 * pixelStep, transPoint.x + 2 * pixelStep);
			curPointValue_DL1 = srcImage.at<uchar>(transPoint.y + 1 * pixelStep, transPoint.x - 1 * pixelStep);
			curPointValue_DL2 = srcImage.at<uchar>(transPoint.y + 2 * pixelStep, transPoint.x - 2 * pixelStep);
			score += (8 * curPointValue - curPointValue_DL1 - curPointValue_DL2 - curPointValue_DR1
				- curPointValue_DR2 - curPointValue_UL1 - curPointValue_UL2 - curPointValue_UR1 - curPointValue_UR2) / 8;
		}		
	}

	//for (int i = 0; i < subMaskPiontSet.size(); i += sampleStep)
	//{
	//	num++;
	//	curPoint = subMaskPiontSet[i];
	//	transPoint = AfterTrans(curTrans, curPoint = subMaskPiontSet[i], centerX, centerY);
	//	if (transPoint.x >= 0 && transPoint.x < srcImage.cols && transPoint.y >= 0 && transPoint.y < srcImage.rows)
	//	{
	//		penalty += 1.0 - srcImage.at<uchar>(transPoint);
	//	}
	//	else
	//	{
	//		penalty += 1;
	//	}
	//}

	distance =1-(score / num);

	return distance;
}


vector<tuple<int, int, double>> ConstructNet(Mat&srcImage, double delta)
{
	
	int lowX = -srcImage.cols;//-5472
	int highX = srcImage.cols;//-3648
	int lowY = -srcImage.rows;
	int highY = srcImage.rows;
	//double lowX = -10;
	//double highX = 10;
	//double lowY = -10;
	//double highY = 10;
	double lowR = -PI;
	double highR = PI;

	double attenuationFactor = 0.1;
	int tx_step = int(attenuationFactor * delta * srcImage.rows);
	int ty_step = int(attenuationFactor * delta * srcImage.rows);
	//double tx_step = delta * srcImage.rows;
	//double ty_step = delta * srcImage.rows;
	double r_step = delta;

	int netSize = (int)(highX - lowX)*(highY - lowY)*(highR - lowR) / (tx_step*ty_step*r_step);

	int tx, ty;
	double r;
	vector<tuple<int, int, double>> Trans;

	for (int tx_index = lowX; tx_index < highX; tx_index += tx_step)
	{
		tx = tx_index;
		for (int ty_index = lowY; ty_index < highY; ty_index += ty_step)
		{
			ty = ty_index;
			for (double r_index = lowR; r_index < highR; r_index += r_step)
			{
				r = r_index;
				tuple<int, int, double>curTrans{ tx,ty,r };
				Trans.push_back(curTrans);
			}
		}
	}

	return Trans;
}


/*计算当前TransNet下的最佳变换*/
CurrentBestReusult GetBestTrans(Contours& maskStruct, Mat&srcImage, vector<Point>&subMaskPiontSet,
	vector<tuple<int, int, double>>&TransNet, double delta, double epsilon)
{
	Mat maskImage = maskStruct.dstImage;
	double centerX = maskStruct.centerX;
	double centerY = maskStruct.centerY;

	double distance = DBL_MAX;
	double temp_distance;
	tuple<int, int, double> bestTrans;
	for (int i = 0; i < TransNet.size(); i++)
	{
		tuple<int, int, double> curTrans = TransNet[i];
		temp_distance = SingleTransEvaluation(maskStruct, srcImage, subMaskPiontSet, curTrans, epsilon);
		//cout << "SingleTransEvaluation's temp_distance is:" << temp_distance << endl;
		if (distance > temp_distance)
		{
			distance = temp_distance;
			bestTrans = curTrans;
		}
	}

	CurrentBestReusult bestResult;
	bestResult.currentBestTrans = bestTrans;
	bestResult.currentBestDistance = distance;

	return bestResult;
}


vector <tuple<int, int, double >> GetNextNet(Mat&srcImage, vector<tuple<int, int, double >> &GoodTransNet,
	vector<Point>&subMaskPointSet, double centerX, double centerY, double delta)
{

	int lowX = -srcImage.cols;
	int highX = srcImage.cols;
	int lowY = -srcImage.rows;
	int highY = srcImage.rows;
	double lowR = 0;
	double highR = 2 * PI;

	double tx_step = delta * srcImage.rows;
	double ty_step = delta * srcImage.rows;
	double r_step = delta;

	int netSize = (int)(highX - lowX)*(highY - lowY)*(highR - lowR) / (tx_step*ty_step*r_step);

	double thetaL, thetaR;
	bool FLAG = true;
	double distanceCurToGood = 0;//计算L(∞)
	vector<tuple<int, int, double>> nextTransNet;
	tuple<int, int, double> extendedTrans;
	for (int i = 0; i < GoodTransNet.size(); i++)
	{
		nextTransNet.push_back(GoodTransNet[i]);
		for (int outerX = -2; outerX <= 2; outerX++)
		{
			for (int outerY = -2; outerY <= 2; outerY++)
			{
				for (int outerTheta = -2; outerTheta <= 2; outerTheta++)
				{
					if (outerX==0 && outerY==0 && outerTheta==0)
					{
						continue;
					}
					get<0>(extendedTrans) = get<0>(GoodTransNet[i]) + outerX * delta * srcImage.rows;
					get<1>(extendedTrans) = get<1>(GoodTransNet[i]) + outerY * delta * srcImage.rows;
					get<2>(extendedTrans) = get<2>(GoodTransNet[i]) + outerTheta * delta;
					nextTransNet.push_back(extendedTrans);
				}
			}
		}
	}

	cout << "next Net's size is :" << nextTransNet.size() << endl;
	return nextTransNet;
}


/* I1 : mask; I2 : src*/
tuple<int, int, double> FastMatch(Contours &maskStruct, Mat &srcImage, double delta, double epsilon, double factor)
{
	Mat maskImage = maskStruct.dstImage;
	double centerX = maskStruct.centerX;
	double centerY = maskStruct.centerY;

	int n1 = maskImage.rows;
	int n2 = srcImage.rows;
	double bestAngle;
	double sampleRate = 0.005;


	/*Step 0 : Get the mask's point*/
	cout << "Step 0:Prepare work : Sample subMaskPontSet from the whole maskImage's pointSet." << endl;
	clock_t t1 = clock();

	//Get the whole PointSet
	vector<Point> PointSet = maskStruct.PointSet;

	//Sample the subMaskPointSet
	vector<Point> subMaskPointSet;
	for (int i = 0; i < PointSet.size(); i += (int)(sampleRate * PointSet.size()))
	{
		subMaskPointSet.push_back(PointSet[i]);
	}

	clock_t t2 = clock();
	cout << "Step 0:PointSet's size is :" << PointSet.size() << endl;
	cout << "Step 0:subMaskPointSet's size is :" << subMaskPointSet.size() << endl;
	cout << "Step 0:Prepare work has been finished !" << endl;
	cout << "Time is :" << (t2 - t1)* 1.0 / CLOCKS_PER_SEC << "s\n" << endl;


	/*Step 1 : Construct the N(δ) net */
	//T : [translationX, translationY, Rtate1, Rotate2, ScaleX, ScaleY] 
	cout << "Step 1:Construct the N(δ) net." << endl;
	clock_t t3 = clock();
	vector<tuple<int, int, double>> TransNet, GoodTransNet;

	//建立初始网络
	TransNet = ConstructNet(srcImage, delta);

	clock_t t4 = clock();
	cout << "Step 1:δis:" << delta << endl;
	cout << "Step 1:Size of the N(δ) net is:" << TransNet.size() << endl;
	cout << "Step 1:Construct the N(δ) net has been finished !" << endl;
	cout << "Time is :" << (t4 - t3)* 1.0 / CLOCKS_PER_SEC << "s\n" << endl;


	/*Step 2: Iterate, update and calculate the best translation.*/
	cout << "Step 2:Iterate, update and calculate the best translation." << endl;
	double distance = 0, alpha = 0.1, beta = 0;
	double L_Delta; 
	double curDistance;
	double bestDistance = DBL_MAX;
	int index = 0;
	vector<double>bestDistanceSet;
	tuple<int, int, double>  bestTrans;
	CurrentBestReusult bestResult;
	clock_t t5 = clock();

	while (true)
	{
		index++;
		L_Delta = alpha * delta + beta;

		/*计算当前变换网络下的最佳变换bestTrans + bestDistance*/
		bestResult = GetBestTrans(maskStruct, srcImage, subMaskPointSet, TransNet, delta, epsilon);

		if (bestDistance > bestResult.currentBestDistance)
		{
			bestDistance = bestResult.currentBestDistance;
			bestTrans = bestResult.currentBestTrans;
		}
		cout << "bestDistance is :" << bestDistance << endl;

		bestDistanceSet.push_back(bestDistance);

		if (bestDistanceSet.size() >= 3 && abs(bestDistanceSet[index - 1] - bestDistanceSet[index - 3]) < 0.1)
		{
			break;
		}

		/*计算和最佳变换相近的次优解集合GoodTransNet*/
		for (int i = 0; i < TransNet.size(); i++)
		{
			tuple<int, int, double> curTrans = TransNet[i];
			curDistance = SingleTransEvaluation(maskStruct, srcImage, subMaskPointSet, curTrans, epsilon);

			if (abs(curDistance - bestDistance) < L_Delta)
			{
				GoodTransNet.push_back(curTrans);
			}
		}
		cout << "the " << index << "th's GoodTransNet's size is :" << GoodTransNet.size() << endl;

		/*更新δ*/
		delta = delta * factor;

		/*根据新的δ和GoodTransNet更新变换网络TransNet*/
		vector<tuple<int, int, double>>().swap(TransNet);
		TransNet = GetNextNet(srcImage, GoodTransNet, subMaskPointSet, centerX, centerY, delta);

		/*清空vector*/
		vector<double>().swap(bestDistanceSet);
		vector<tuple<int, int, double>>().swap(GoodTransNet);

		if (delta < 0.0005/* || bestDistance < 0.2*/)
		{
			break;
		}

	}
	clock_t t6 = clock();
	cout << "Step 2:Size of the next_N(δ) net is:" << TransNet.size() << endl;
	cout << "Step 2:delta is :" << delta << "; bestDistance is :" << bestDistance << endl;
	cout << "Step 2:Iterate, update and calculate the best translation has been finished !" << endl;
	cout << "Time is :" << (t6 - t5)* 1.0 / CLOCKS_PER_SEC << "s\n" << endl;

	return bestTrans;
}
