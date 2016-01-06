/*
Name: EigenFace_train
Author: Gong Lixue 3130104153@zju.edu.cn
Description: Without using relevant eigenface function in opencv to realize EigenFace algorithm based on PCA.
			 The main part of this algorithm is self-completed except for some maxtrix calculation.
Date: 2016-01-06
*/

#include <cv.h>
#include <highgui.h>
#include <iostream>
using namespace std;

int main(int argc,char* argv[])
{
	if (argc != 5){
		cout << "wrong argument" << endl;
		return -1;
	}

	char* filename = argv[1];
	CvMat* eigenface = (CvMat*)cvLoad(argv[2]);
	//cvSave("eigenface2.txt", eigenface);
	CvMat* traincoeff = (CvMat*)cvLoad(argv[3]);
	CvMat* mean = (CvMat*)cvLoad(argv[4]);  //N*1
	int N = cvGetSize(eigenface).height;
	int M = cvGetSize(eigenface).width;
	

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

	CvMat* coeff = cvCreateMat(1, M, CV_32FC1); //投影后坐标
	cvMatMul(input_n_T, eigenface, coeff);
	//输出坐标测试
	/*
	for (int i = 0; i < M; i++)
	{
		float value = cvGetReal2D(coeff, 0, i);
		cout << value << " ";
	}
	cout << endl;
	*/
	CvMat* dist = cvCreateMat(1, M, CV_32FC1); //用来存放输入图像与每张训练图像的欧式距离
	CvMat* coeffT = cvCreateMat(M, 1, CV_32FC1);
	cvTranspose(coeff, coeffT);
	for (int i = 0; i < M; i++){
		//计算输入图片投影后的坐标与traincoeff中的每列的欧式举例
		
		CvMat* train_each = cvCreateMat(M, 1, CV_32FC1);  //读取traincoeff中的每一列

		for (int j = 0; j < M; j++){
			float value = cvGetReal2D(traincoeff, j, i);
			cvSetReal2D(train_each, j, 0, value);
		}
		CvMat* coeffTtemp = cvCreateMat(M, 1, CV_32FC1);
		cvSub(coeffT, train_each, coeffTtemp);
		float distance = 0;
		for (int j = 0; j < M; j++){
			float value = cvGetReal2D(coeffTtemp, j, 0);
			distance += value*value;
		}
		cvSetReal2D(dist, 0, i, sqrt(distance));
	}

	// 找到距离最小值
	CvPoint min_loc = cvPoint(0,0);
	CvPoint max_loc = cvPoint(0,0);
	double min_value, max_value;
	cvMinMaxLoc(dist, &min_value, &max_value, &min_loc, &max_loc);

	int id = (&min_loc)->x;  // 最小距离位于第几列，就是人脸库中最相近的人脸
	cout << "和第" << id << "张最像" << endl;

	// 使用投影坐标coeff 重构出一张人脸
	CvMat* reconstruct = cvCreateMat(N, 1, CV_32FC1);
	cvMatMul(eigenface, coeffT, reconstruct);
	cvAdd(reconstruct, mean, reconstruct);  //再加回平均人脸
	// 系数*基底，有M项相加
	/*
	for (int i = 0; i < M; i++)
	{
		float c = cvGetReal2D(coeff, 0, i); //读取系数
		CvMat* each_face = cvCreateMat(N, 1, CV_32FC1);
		// 读取eigenface中的一列，也就是一张特征脸
		for (int j = 0; j < N; j++){
			float value = cvGetReal2D(eigenface, j, i);  //第j行 第i列
			cvSetReal2D(each_face, j, 0, value*c); //将系数与对应基底（特征人脸）数乘
		}
		
		//把系数*基底加到最终重构结果上
		cvAdd(each_face, reconstruct, reconstruct);

	}
	*/


	// 把reconstruct映射到0-255上
	cvNormalize(reconstruct, reconstruct, 255, 0, CV_MINMAX);
	//for (int i = 0; i < N; i++)
	//{
	//	float value = cvGetReal2D(reconstruct, i, 0);
	//	cout << value << "/";
	//}
	// 把reconstruct reshape成rows*cols;
	CvMat* reconstruct_reshape = cvCreateMat(rows, cols, CV_32FC1);
	CvMat reconstruct_hdr;
	reconstruct_reshape = cvReshape(reconstruct, &reconstruct_hdr,1, rows);
	IplImage* reconstruct_show = cvCreateImage(cvSize(rows, cols), IPL_DEPTH_8U, 1);
	IplImage reconstruct_show_hdr;
	reconstruct_show = cvGetImage(&reconstruct_hdr, &reconstruct_show_hdr);
	//cvShowImage("reconstruct", reconstruct_show);
	cvSaveImage("./output/reconstruct.jpg", reconstruct_show);
	//reconstruct_show = cvLoadImage("./output/reconstruct.jpg");
	//cvShowImage("reconstruct", reconstruct_show);
	cvShowImage("input", input);

	// 将重构结果叠加到输入的人脸上
	IplImage* reconstruct_add_input = cvCreateImage(cvGetSize(reconstruct_show), IPL_DEPTH_8U, 1);
	cvAddWeighted(input_gray, 0.5, reconstruct_show, 0.5, 0, reconstruct_add_input);
	cvShowImage("addweight", reconstruct_add_input);

	//显示最相近的图片
	IplImage* most_likely;
	char likely_filename[50];
	sprintf(likely_filename, "./train2/BioID_%04d.pgm", id);
	most_likely = cvLoadImage(likely_filename);
	cvShowImage("most likely image", most_likely);

	cvWaitKey(0);
	cvReleaseImage(&reconstruct_show);
	cvReleaseImage(&input);
	cvReleaseImage(&input_gray);
	cvReleaseImage(&reconstruct_show);
	cvReleaseImage(&reconstruct_add_input);





}
