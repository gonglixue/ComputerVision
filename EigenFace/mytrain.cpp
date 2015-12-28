#include <cv.h>
#include <highgui.h>
#include <vector>
#include <iostream>

using namespace std;

int main()
{
	int M=4;
	int cols, rows;
	int N; //单幅图片像素
	//vector<IplImage> images; //定义容器来存放M张训练图片
	IplImage* temp = cvLoadImage("./train/1.jpg");
	rows = cvGetSize(temp).height;
	cols = cvGetSize(temp).width;
	N = cols*rows;
	CvMat* S = cvCreateMat(N, M,CV_32FC1);
	
	//循环载入训练图片，构建出N*M矩阵S
	for (int i = 1; i <= M; i++)
	{
		char filename[50];
		sprintf(filename, "./train/%d.jpg", i);
		IplImage* image;
		image = cvLoadImage(filename);
		IplImage* image_gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1); //单通道灰度
		//cvShowImage(filename, image); 每张图片导入正常
		
		cvCvtColor(image, image_gray, CV_BGR2GRAY);   
		CvMat* img_mat = cvCreateMat(rows, cols, CV_16UC1);  //载入图片的矩阵形式，便于reshape
		// 模板预处理

		//灰度均衡，直方图均衡化
		cvEqualizeHist(image_gray, image_gray);

		cvConvert(image_gray, img_mat);  //把图片转化为矩阵		
		//reshape拉伸
		CvMat* oneCol,mathdr;
		oneCol = cvReshape(img_mat, &mathdr, 1, N); //把矩阵reshape，oneCol是reshape后的N行列向量
		for (int j = 0; j < N; j++){
			int value = cvGetReal2D(oneCol, j, 0);
			cvSetReal2D(S, j, i-1, value);  //S的一列就是一张图片。
			//cout << value << "/";  //这里value都挺正常的
		}
	}

	//对S中列向量求和，进而求出平均人脸
	CvMat* mean_face = cvCreateMat(N, 1,CV_16UC1);
	cvSetZero(mean_face);
	for (int i = 0; i < M; i++){
		CvMat* temp = cvCreateMat(N,1,CV_16UC1);  //为毛64位出错
		cvGetCol(S, temp, i);
		cvAdd(temp, mean_face, mean_face);
	}	
	//cvReduce(S, mean_face, 1, CV_REDUCE_SUM);  //合并成一列
	// 求均值,数乘1/M
	cvConvertScale(mean_face, mean_face, 1.0 / M);

	/*取出S的一列进行重构测试*/
	IplImage* mean_show = cvCreateImage(cvSize(cols, rows),IPL_DEPTH_8U,1);
	//IplImage* mean_show;
	IplImage mean_show_hdr;
	CvMat* mean_reshape = cvCreateMat(rows, cols, CV_8UC1);
	CvMat mean_hdr;
	//CvMat* s_1 = cvCreateMat(N, 1, CV_64FC1);
	//cvGetCol(S, s_1, 0);
	mean_reshape = cvReshape(mean_face, &mean_hdr, 1, rows);  //测试输出meanface都是0
	mean_show = cvGetImage(&mean_hdr, &mean_show_hdr);
	cvShowImage("mean", mean_show);
	cvSaveImage("./output/mean.jpg", mean_show);  //窗口显示不出来，但是保存后是正确的


	// S中的每列减去平均人脸...这复杂度。。。
	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			int S_value = cvGetReal2D(S, j, i);
			int mean_value = cvGetReal2D(mean_face, j, 0);
			cvSetReal2D(S, j, i, S_value - mean_value);
		}
	}
	// 求S的转置矩阵
	CvMat* ST = cvCreateMat(M, N, CV_32FC1);
	cvTranspose(S, ST);
	//求协方差矩阵S'S，大小为MM
	CvMat* Col_mat = cvCreateMat(M, M,CV_32FC1);
	cvMatMul(ST, S, Col_mat);  //矩阵相乘
	//求矩阵Col_mat的特征值与特征向量，有M-1个特征向量？
	//构造输出特征向量矩阵
	CvMat* ProVector = cvCreateMat(M, M, CV_32FC1);
	//构造输出特征值矩阵
	CvMat* ProValue = cvCreateMat(M, 1, CV_32FC1);
	cvEigenVV(Col_mat, ProVector, ProValue, 1.0e-6F);
	//特征向量矩阵左乘S
	CvMat* eigenface = cvCreateMat(N, M, CV_32FC1);
	cvMatMul(S, ProVector, eigenface);
	/*求出来的特征向量矩阵ProVector已经是归一化的了貌似，但做成了S之后的特征人脸空间并不是。。。*/
	/*输出特征向量矩阵测试*/
	for (int i = 0; i < M; i++){
		for (int j = 0; j < M; j++){
			cout << cvGetReal2D(eigenface, j, i) << " ";
		}
		cout << endl;
	}
	
	cout << "----------";
	//对eigenface矩阵每一列归一化
	CvMat* tempVector = cvCreateMat(N,M,CV_32FC1);
	cvPow(eigenface,tempVector,2);  //元素平方
	CvMat* temp2 = cvCreateMat(1,M,CV_32FC1);
	cvReduce(tempVector, temp2, 0, CV_REDUCE_SUM);  //平方后求和，合并成一行 ??? dim到底。。。
	for (int i = 0; i < M; i++){   //列
		double sum;
		sum = cvGetReal2D(temp2, 0, i); //第0行，第i列
		//cout << "sum:" << sum << endl;
		for (int j = 0; j < N; j++){    //行
			double value;
			value = cvGetReal2D(eigenface, j, i);
			//cout << value << " ";  //好多value都是0
			value = value / (sqrt(sum));
			cvSetReal2D(eigenface, j, i, value);
			
		}
	}
	/*测试是不是真的归一化了 测试第一列结果正确：)*/
	
	double testsum = 0;
	for (int i = 0; i < N; i++){
		double value;
		value = cvGetReal2D(eigenface, i, 0);
		testsum += value*value;
	}
	testsum = sqrt(testsum);
	cout << " test sum:" << testsum;
	
	//eigenface空间构建完毕

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
