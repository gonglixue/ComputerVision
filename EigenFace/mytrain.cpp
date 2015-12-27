#include <cv.h>
#include <highgui.h>
#include <vector>
#include <iostream>

using namespace std;

int main()
{
	int M=5;
	int cols, rows;
	int N; //单幅图片像素
	//vector<IplImage> images; //定义容器来存放M张训练图片
	IplImage* temp = cvLoadImage("./train/1.jpg");
	cols = cvGetSize(temp).height;
	rows = cvGetSize(temp).width;
	N = cols*rows;
	CvMat* S = cvCreateMat(N, M, CV_64FC1);
	cvShowImage("1", temp);
	//循环载入训练图片，构建出N*M矩阵S
	for (int i = 1; i <= M; i++)
	{
		char filename[50];
		sprintf(filename, "./train/%d.jpg", i);
		IplImage* image_gray = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_8U, 1); //单通道灰度
		IplImage* image;
		image = cvLoadImage(filename);
		cvCvtColor(image, image_gray, CV_BGR2GRAY);    //出错
		CvMat* img_mat = cvCreateMat(N, 1, CV_16UC1);  //载入图片的矩阵形式，便于reshape
		// 模板预处理

		//灰度均衡，直方图均衡化
		cvEqualizeHist(image_gray, image_gray);

		cvConvert(image_gray, img_mat);  //把图片转化为矩阵		
		//reshape拉伸
		CvMat* oneCol,mathdr;
		oneCol = cvReshape(img_mat, &mathdr, 1, N); //把矩阵reshape，oneCol是reshape后的N行列向量
		for (int j = 0; j < rows; j++){
			int value = cvGetReal2D(oneCol, j, 0);
			cvSetReal2D(S, j, i, value);  //S的一列就是一张图片。
		}
	}

	//对S中列向量求和，进而求出平均人脸
	CvMat* mean_face = cvCreateMat(N, 1, CV_64FC1);
	for (int i = 0; i < M; i++){
		CvMat* temp = cvCreateMat(N,1,CV_64FC1);
		cvGetCol(S, temp, i);
		cvAdd(temp, mean_face, mean_face);
	}
	// 求均值,数乘1/M
	cvConvertScale(mean_face, mean_face, 1.0 / M);
	// S中的每列减去平均人脸...这复杂度。。。
	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			int S_value = cvGetReal2D(S, j, i);
			int mean_value = cvGetReal2D(mean_face, j, 0);
			cvSetReal2D(S, j, i, S_value - mean_value);
		}
	}
	// 求S的转置矩阵
	CvMat* ST = cvCreateMat(M, N, CV_64FC1);
	cvTranspose(S, ST);
	//求协方差矩阵S'S，大小为MM
	CvMat* Col_mat = cvCreateMat(M, M, CV_64FC1);
	cvMatMul(ST, S, Col_mat);  //矩阵相乘
	//求矩阵Col_mat的特征值与特征向量，有M-1个特征向量？
	//构造输出特征向量矩阵
	CvMat* ProVector = cvCreateMat(M, M, CV_64FC1);
	//构造输出特征值矩阵
	CvMat* ProValue = cvCreateMat(M, 1, CV_64FC1);
	cvEigenVV(Col_mat, ProVector, ProValue, 1.0e-6F);
	//特征向量矩阵左乘S
	CvMat* eigenface = cvCreateMat(N, M, CV_64FC1);
	cvMatMul(S, ProVector, eigenface);

	//对eigenface矩阵每一列归一化
	CvMat* tempVector = cvCreateMat(M,M,CV_64FC1);
	cvPow(eigenface,tempVector,2);  //元素平方
	CvMat* temp2 = cvCreateMat(1,M,CV_64FC1);
	cvReduce(tempVector, temp2, 1, CV_REDUCE_SUM);  //平方后求和，合并成一行
	for (int i = 0; i < M; i++){   //列
		double sum;
		sum = cvGetReal2D(temp2, 0, i); //第0行，第i列
		for (int j = 0; j < M; j++){    //行
			double value;
			value = cvGetReal2D(eigenface, j, i);
			value = value / (sqrt(sum));
			cvSetReal2D(eigenface, j, i, value);
		}
	}
	//基底构建完了？tempVector的每一列是特征人脸？

	//计算训练集中的照片在基底上的投影坐标。

	//CvMat* Mat1 = cvCreateMat(2, 2, CV_64FC1);   //对每个像素 构造一个2*2大小的M
	//cvSetReal2D(Mat1, 0, 0, 1);
	//cvSetReal2D(Mat1, 0, 1, 2);
	//cvSetReal2D(Mat1, 1, 0, 3);
	//cvSetReal2D(Mat1, 1, 1, 4);
	//CvMat* Mat2, mathdr;
	//Mat2 = cvReshape(Mat1, &mathdr, 1, 4);
	//cout << "height:" << cvGetSize(Mat2).height;
	//cout << "width" << cvGetSize(Mat2).width;

	cvWaitKey(0);
	cvReleaseImage(&temp);

}
