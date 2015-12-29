#include <cv.h>
#include <highgui.h>
#include <iostream>
using namespace std;

int main()
{
	CvMat* eigenface = (CvMat*)cvLoad("eigenface.txt");
	//cvSave("eigenface2.txt", eigenface);
	CvMat* traincoeff = (CvMat*)cvLoad("traincoeff.txt");
	CvMat* mean = (CvMat*)cvLoad("meanface.txt");  //N*1
	int N = cvGetSize(eigenface).height;
	int M = cvGetSize(eigenface).width;
	char filename[50] = "5.pgm";

	// 需要把input处理成与训练集人脸同大小吗？
	// 假设input与训练集人脸同大小；
	IplImage* input = cvLoadImage(filename);
	IplImage* input_gray = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	cvCvtColor(input, input_gray, CV_BGR2GRAY);

	cvEqualizeHist(input_gray, input_gray); //均衡化
	int rows = cvGetSize(input).height;
	int cols = cvGetSize(input).width;
	CvMat* input_mat = cvCreateMat(rows, cols, CV_32FC1);
	cvConvert(input_gray, input_mat);

	// 把input_mat拉伸成列向量
	CvMat* input_n, mathdr;
	input_n = cvReshape(input_mat, &mathdr, 1, N);
	// 拉伸后的列向量减去平均人脸,使用元素级相减函数
	cvSub(input_n, mean, input_n);

	//计算输入照片在基底上的投影坐标
	CvMat* input_n_T = cvCreateMat(1, N, CV_32FC1);
	cvTranspose(input_n, input_n_T);

	CvMat* coeff = cvCreateMat(1, M, CV_32FC1);
	cvMatMul(input_n_T, eigenface, coeff);

	CvMat* dist = cvCreateMat(1, M, CV_32FC1); //用来存放输入图像与每张训练图像的欧式距离
	for (int i = 0; i < M; i++){
		//计算输入图片的坐标与traincoeff中的每列的内积
		
		CvMat* train_each = cvCreateMat(M, 1, CV_32FC1);

		for (int j = 0; j < M; j++){
			float value = cvGetReal2D(traincoeff, j, i);
			cvSetReal2D(train_each, j, 0, value);
		}
		CvMat* coeffT = cvCreateMat(M, 1, CV_32FC1);
		cvTranspose(coeff, coeffT);
		cvSub(coeffT, train_each, coeffT);
		float distance = 0;
		for (int j = 0; j < M; j++){
			float value = cvGetReal2D(coeffT, j, 0);
			distance += value*value;
		}
		cvSetReal2D(dist, 0, i, sqrt(distance));
	}

	// 找到距离最小值
	CvPoint min_loc = cvPoint(0,0);
	CvPoint max_loc = cvPoint(0,0);
	double min_value, max_value;
	cvMinMaxLoc(dist, &min_value, &max_value, &min_loc, &max_loc);

	int id = (&min_loc)->x;  // x对应第几列吗？
	cout << "和第" << id << "张最像" << endl;


}
