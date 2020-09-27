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





Point2f AfterTrans(tuple<int, int, float>&curTrans, Point2f curPoint, float centerX, float centerY)
{
	Point2f resultPoint;
	int transX = get<0>(curTrans);
	int transY = get<1>(curTrans);
	float theta = get<2>(curTrans);

	resultPoint.x = (int)(cos(theta)*(curPoint.x - centerX) - sin(theta)*(curPoint.y - centerY)) + centerX + transX;
	resultPoint.y = (int)(sin(theta)*(curPoint.x - centerX) + cos(theta)*(curPoint.y - centerY)) + centerY + transY;

	return resultPoint;
}


float SingleTransEvaluation(Contours& maskStruct, Mat &srcImage, vector<Point2f> &subMaskPiontSet,
	tuple<int, int, float>& curTrans, float epsilon)
{
	Mat maskmage = maskStruct.dstImage;
	float centerX = maskStruct.centerX;
	float centerY = maskStruct.centerY;

	int pixelStep = int(10/scale);
	float curPointValue;
	float curPointValue_UR1;
	float curPointValue_UR2;
	float curPointValue_UL1;
	float curPointValue_UL2;
	float curPointValue_DR1;
	float curPointValue_DR2;
	float curPointValue_DL1;
	float curPointValue_DL2;

	Point2f curPoint, transPoint;
	int sampleStep = (int)(1 / pow(epsilon, 2)), num = 0;
	float distance = 0, score = 0, temp = 0;
	for (int i = 0; i < subMaskPiontSet.size(); i += sampleStep)
	{
		num++;
		curPoint = subMaskPiontSet[i];
		transPoint = AfterTrans(curTrans, subMaskPiontSet[i],centerX,centerY);
		if (transPoint.x >= 0 && transPoint.x < srcImage.cols && transPoint.y >= 2*pixelStep && transPoint.y < srcImage.rows-2*pixelStep)
		{
			curPointValue = srcImage.at<float>(transPoint.y, transPoint.x);
			curPointValue_UR1 = srcImage.at<float>(transPoint.y - 1 * pixelStep, transPoint.x + 1 * pixelStep);
			curPointValue_UR2 = srcImage.at<float>(transPoint.y - 2 * pixelStep, transPoint.x + 2 * pixelStep);
			curPointValue_UL1 = srcImage.at<float>(transPoint.y - 1 * pixelStep, transPoint.x - 1 * pixelStep);
			curPointValue_UL2 = srcImage.at<float>(transPoint.y - 2 * pixelStep, transPoint.x - 2 * pixelStep);
			curPointValue_DR1 = srcImage.at<float>(transPoint.y + 1 * pixelStep, transPoint.x + 1 * pixelStep);
			curPointValue_DR2 = srcImage.at<float>(transPoint.y + 2 * pixelStep, transPoint.x + 2 * pixelStep);
			curPointValue_DL1 = srcImage.at<float>(transPoint.y + 1 * pixelStep, transPoint.x - 1 * pixelStep);
			curPointValue_DL2 = srcImage.at<float>(transPoint.y + 2 * pixelStep, transPoint.x - 2 * pixelStep);
			temp = (8 * curPointValue - curPointValue_DL1 - curPointValue_DL2 - curPointValue_DR1
				- curPointValue_DR2 - curPointValue_UL1 - curPointValue_UL2 - curPointValue_UR1 - curPointValue_UR2);
			score += (temp / 2 + 4)/8;//Normalize->[0,1]
		}		
	}

	distance =1-(score / num);

	return distance;
}


vector<tuple<int, int, float>> ConstructNet(Mat&srcImage, float delta)
{
	
	int lowX = -srcImage.cols;//-5472
	int highX = srcImage.cols;//-3648
	int lowY = -srcImage.rows;
	int highY = srcImage.rows;
	//float lowX = -10;
	//float highX = 10;
	//float lowY = -10;
	//float highY = 10;
	float lowR = -PI;
	float highR = PI;

	float attenuationFactor = 0.1;
	int tx_step = int(attenuationFactor * delta * srcImage.rows);
	int ty_step = int(attenuationFactor * delta * srcImage.rows);
	//float tx_step = delta * srcImage.rows;
	//float ty_step = delta * srcImage.rows;
	float r_step = delta;

	int netSize = (int)(highX - lowX)*(highY - lowY)*(highR - lowR) / (tx_step*ty_step*r_step);

	int tx, ty;
	float r;
	vector<tuple<int, int, float>> Trans;

	for (int tx_index = lowX; tx_index < highX; tx_index += tx_step)
	{
		tx = tx_index;
		for (int ty_index = lowY; ty_index < highY; ty_index += ty_step)
		{
			ty = ty_index;
			for (float r_index = lowR; r_index < highR; r_index += r_step)
			{
				r = r_index;
				tuple<int, int, float>curTrans{ tx,ty,r };
				Trans.push_back(curTrans);
			}

		}
	}

	return Trans;
}


/*计算当前TransNet下的最佳变换*/
CurrentBestReusult GetBestTrans(Contours& maskStruct, Mat& srcImage, vector<Point2f>& subMaskPiontSet,
	vector<tuple<int, int, float>>& TransNet, float delta, float epsilon)
{
	Mat maskImage = maskStruct.dstImage;
	float centerX = maskStruct.centerX;
	float centerY = maskStruct.centerY;

	float distance = DBL_MAX;
	float temp_distance;
	tuple<int, int, float> bestTrans;
	for (int i = 0; i < TransNet.size(); i++)
	{
		tuple<int, int, float> curTrans = TransNet[i];
		temp_distance = SingleTransEvaluation(maskStruct, srcImage, subMaskPiontSet, curTrans, epsilon);
		//cout << "SingleTransEvaluation's temp_distance is:" << temp_distance << endl;
		if (abs(distance) > abs(temp_distance))
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


vector <tuple<int, int, float >> GetNextNet(Mat&srcImage, vector<tuple<int, int, float >> &GoodTransNet,
	vector<Point2f>&subMaskPointSet, float centerX, float centerY, float delta)
{

	int lowX = -srcImage.cols;
	int highX = srcImage.cols;
	int lowY = -srcImage.rows;
	int highY = srcImage.rows;
	float lowR = 0;
	float highR = 2 * PI;

	float tx_step = delta * srcImage.rows;
	float ty_step = delta * srcImage.rows;
	float r_step = delta;

	int netSize = (int)(highX - lowX)*(highY - lowY)*(highR - lowR) / (tx_step*ty_step*r_step);

	float thetaL, thetaR;
	bool FLAG = true;
	float distanceCurToGood = 0;//计算L(∞)
	vector<tuple<int, int, float>> nextTransNet;
	tuple<int, int, float> extendedTrans;
	for (int i = 0; i < GoodTransNet.size(); i++)
	{
		nextTransNet.push_back(GoodTransNet[i]);
		for (int outerX = -2; outerX <= 2; outerX++)
		{
			for (int outerY = -2; outerY <= 2; outerY++)
			{
				for (int outerTheta = -2; outerTheta <= 2; outerTheta++)
				{
					if (outerX == 0 && outerY == 0 && outerTheta == 0)
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
tuple<int, int, float> FastMatch(Contours &maskStruct, Mat &srcImage, float delta, float epsilon, float factor)
{
	float bestAngle;
	float sampleRate = 0.005;


	/*Step 0 :Sample the subMaskPointSet*/
	cout << "Step 0:Prepare work : Sample subMaskPontSet from the whole maskImage's pointSet." << endl;
	clock_t t1 = clock();

	Mat maskImage = maskStruct.dstImage;
	vector<Point2f> PointSet = maskStruct.PointSet;
	float centerX = maskStruct.centerX;
	float centerY = maskStruct.centerY;

	int n1 = maskImage.rows;
	int n2 = srcImage.cols;

	vector<Point2f> subMaskPointSet;
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
	//T : [translationX, translationY, Rtate] 
	cout << "Step 1:Construct the N(δ) net." << endl;
	clock_t t3 = clock();
	vector<tuple<int, int, float>> TransNet, GoodTransNet;

	//建立初始网络
	TransNet = ConstructNet(srcImage, delta);

	clock_t t4 = clock();
	cout << "Step 1:δis:" << delta << endl;
	cout << "Step 1:Size of the N(δ) net is:" << TransNet.size() << endl;
	cout << "Step 1:Construct the N(δ) net has been finished !" << endl;
	cout << "Time is :" << (t4 - t3)* 1.0 / CLOCKS_PER_SEC << "s\n" << endl;


	/*Step 2: Iterate, update and calculate the best translation.*/
	cout << "Step 2:Iterate, update and calculate the best translation." << endl;
	float distance = 0, alpha = 0.1, beta = 0.01;
	float L_Delta; 
	float curDistance;
	float bestDistance = DBL_MAX;
	int index = 0;
	vector<float>bestDistanceSet;
	tuple<int, int, float>  bestTrans;
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
			tuple<int, int, float> curTrans = TransNet[i];
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
		vector<tuple<int, int, float>>().swap(TransNet);
		TransNet = GetNextNet(srcImage, GoodTransNet, subMaskPointSet, centerX, centerY, delta);
		if (TransNet.size()>10000)
		{
			break;
		}
		/*清空vector*/
		vector<float>().swap(bestDistanceSet);
		vector<tuple<int, int, float>>().swap(GoodTransNet);

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
