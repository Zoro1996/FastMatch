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

	int pixelStep = 10;
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
	double distance = 0, penalty = 0;
	//for (int i = 0; i < subMaskPiontSet.size(); i += sampleStep)
	//{
	//	num++;
	//	curPoint = subMaskPiontSet[i];
	//	transPoint = AfterTrans(curTrans, curPoint = subMaskPiontSet[i],centerX,centerY);
	//	if (transPoint.x >= 0 && transPoint.x < srcImage.cols && transPoint.y >= 2*pixelStep && transPoint.y < srcImage.rows-2*pixelStep)
	//	{
	//		//penalty += abs(maskImage.at<uchar>(curPoint) - srcImage.at<uchar>(transPoint));
	//		curPointValue = srcImage.at<uchar>(transPoint.y, transPoint.x);
	//		curPointValue_UR1 = srcImage.at<uchar>(transPoint.y - 1 * pixelStep, transPoint.x + 1 * pixelStep);
	//		curPointValue_UR2 = srcImage.at<uchar>(transPoint.y - 2 * pixelStep, transPoint.x + 2 * pixelStep);
	//		curPointValue_UL1 = srcImage.at<uchar>(transPoint.y - 1 * pixelStep, transPoint.x - 1 * pixelStep);
	//		curPointValue_UL2 = srcImage.at<uchar>(transPoint.y - 2 * pixelStep, transPoint.x - 2 * pixelStep);
	//		curPointValue_DR1 = srcImage.at<uchar>(transPoint.y + 1 * pixelStep, transPoint.x + 1 * pixelStep);
	//		curPointValue_DR2 = srcImage.at<uchar>(transPoint.y + 2 * pixelStep, transPoint.x + 2 * pixelStep);
	//		curPointValue_DL1 = srcImage.at<uchar>(transPoint.y + 1 * pixelStep, transPoint.x - 1 * pixelStep);
	//		curPointValue_DL2 = srcImage.at<uchar>(transPoint.y + 2 * pixelStep, transPoint.x - 2 * pixelStep);
	//		penalty += abs(8 * curPointValue - curPointValue_DL1 - curPointValue_DL2 - curPointValue_DR1
	//			- curPointValue_DR2 - curPointValue_UL1 - curPointValue_UL2 - curPointValue_UR1 - curPointValue_UR2);
	//	}
	//	
	//}


	for (int i = 0; i < subMaskPiontSet.size(); i += sampleStep)
	{
		num++;
		curPoint = subMaskPiontSet[i];
		transPoint = AfterTrans(curTrans, curPoint = subMaskPiontSet[i], centerX, centerY);
		if (transPoint.x >= 0 && transPoint.x < srcImage.cols && transPoint.y >= 0 && transPoint.y < srcImage.rows)
		{
			penalty += 1.0 - srcImage.at<uchar>(transPoint);
		}
		else
		{
			penalty += 1;
		}
	}

	distance = penalty / num;

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


/*���㵱ǰTransNet�µ���ѱ任*/
tuple<int, int, double> GetBestTrans(Contours& maskStruct, Mat&srcImage, vector<Point>&subMaskPiontSet,
	vector<tuple<int, int, double>>&TransNet, double delta, double epsilon)
{
	Mat maskImage = maskStruct.dstImage;
	double centerX = maskStruct.centerX;
	double centerY = maskStruct.centerY;

	double distance = 0;
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

	return bestTrans;
}


vector <tuple<int, int, double >> GetNextNet(Mat&srcImage, vector<tuple<int, int, double >> &GoodTransNet,
	vector<Point>&subMaskPointSet, double centerX, double centerY, double delta)
{
	double lowX = -srcImage.cols;
	double highX = srcImage.cols;
	double lowY = -srcImage.rows;
	double highY = srcImage.rows;
	double lowR = 0;
	double highR = 2 * PI;

	double tx_step = delta * srcImage.rows;
	double ty_step = delta * srcImage.rows;
	double r_step = delta;

	int netSize = (int)(highX - lowX)*(highY - lowY)*(highR - lowR) / (tx_step*ty_step*r_step);

	int tx, ty;
	double r;
	bool FLAG = true;
	double distanceCurToGood = 0;//����L(��)
	vector<tuple<int, int, double>> TransNet;
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

				for (int i = 0; i < subMaskPointSet.size(); i++)
				{
					/*ģ��㼯������һ��curPoint*/
					Point curPoint = subMaskPointSet[i];
					/*curPoint������ǰ�任��ĵ�transPoint*/
					Point transPoint = AfterTrans(curTrans, curPoint, centerX, centerY);
					/*����GoodTransNet������ max|transPoint - curGoodTransNet(curPoint)|^2*/
					for (int j = 0; j < GoodTransNet.size(); j++)
					{
						FLAG = true;
						tuple<int, int, double> curGoodTrans = GoodTransNet[i];
						Point curGoodTransPoint = AfterTrans(curGoodTrans, curPoint, centerX, centerY);
						if (curGoodTransPoint.x >= 0 && curGoodTransPoint.x < srcImage.cols &&curGoodTransPoint.y >= 0 && curGoodTransPoint.y < srcImage.rows)
						{
							double distance = sqrt((transPoint.x - curGoodTransPoint.x) ^ 2 + (transPoint.y - curGoodTransPoint.y) ^ 2);
							if (distance > 0.05*delta*srcImage.rows)
							{
								i = subMaskPointSet.size();
								j = GoodTransNet.size();
								FLAG = false;
							}
						}
					}
				}

				/*���GoodTransNet�д��ڱ任curGoodTransNet����curTrans֮��ľ���С������ֵ���洢��ǰcurTrans*/
				if (FLAG)
				{
					TransNet.push_back(curTrans);
				}
			}
		}
	}

	cout << "next Net's size is :" << TransNet.size() << endl;
	return TransNet;
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
	double sampleRate = 0.01;


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


	/*Step 1 : Construct the N(��) net */
	//T : [translationX, translationY, Rtate1, Rotate2, ScaleX, ScaleY] 
	cout << "Step 1:Construct the N(��) net." << endl;
	clock_t t3 = clock();
	vector<tuple<int, int, double>> TransNet, GoodTransNet;

	//������ʼ����
	TransNet = ConstructNet(srcImage, delta);

	clock_t t4 = clock();
	cout << "Step 1:��is:" << delta << endl;
	cout << "Step 1:Size of the N(��) net is:" << TransNet.size() << endl;
	cout << "Step 1:Construct the N(��) net has been finished !" << endl;
	cout << "Time is :" << (t4 - t3)* 1.0 / CLOCKS_PER_SEC << "s\n" << endl;


	/*Step 2: Iterate, update and calculate the best translation.*/
	cout << "Step 2:Iterate, update and calculate the best translation." << endl;
	double distance = 0, alpha = 0.99, beta = 0.02;
	double L_Delta = alpha * delta + beta;
	double curDistance;
	double bestDistance = DBL_MAX;
	int index = 0;
	vector<double>bestDistanceSet;
	tuple<int, int, double>  bestTrans, tempBestTrans;
	clock_t t5 = clock();

	while (true)
	{
		index++;

		/*���㵱ǰ�任�����µ���ѱ任bestTrans*/
		tempBestTrans = GetBestTrans(maskStruct, srcImage, subMaskPointSet, TransNet, delta, epsilon);

		/*������ѱ任�����µ�ģ��ƥ�����*/
		double tempBestDistance = SingleTransEvaluation(maskStruct, srcImage, subMaskPointSet, tempBestTrans, epsilon);

		if (bestDistance > tempBestDistance)
		{
			bestDistance = tempBestDistance;
			bestTrans = tempBestTrans;
		}
		cout << "bestDistance is :" << bestDistance << endl;

		bestDistanceSet.push_back(bestDistance);
		if (bestDistanceSet.size() >= 3 && abs(bestDistanceSet[index - 1] - bestDistanceSet[index - 3]) < 0.1)
		{
			break;
		}

		/*�������ѱ任����Ĵ��Ž⼯��GoodTransNet*/
		for (int i = 0; i < TransNet.size(); i++)
		{
			tuple<int, int, double> curTrans = TransNet[i];
			curDistance = SingleTransEvaluation(maskStruct, srcImage, subMaskPointSet, curTrans, epsilon);

			if (abs(curDistance - bestDistance) < L_Delta)
			{
				GoodTransNet.push_back(curTrans);
			}
		}

		/*���¦�*/
		delta = delta * factor;

		/*�����µĦĺ�GoodTransNet���±任����TransNet*/
		TransNet = GetNextNet(srcImage, GoodTransNet, subMaskPointSet, centerX, centerY, delta);


		if (delta < 0.05 || bestDistance < 0.2)
		{
			break;
		}

	}
	clock_t t6 = clock();
	cout << "Step 1:Size of the next_N(��) net is:" << TransNet.size() << endl;
	cout << "Step 1:delta is :" << delta << "; bestDistance is :" << bestDistance << endl;
	cout << "Step 2:Iterate, update and calculate the best translation has been finished !" << endl;
	cout << "Time is :" << (t6 - t5)* 1.0 / CLOCKS_PER_SEC << "s\n" << endl;

	return bestTrans;
}
